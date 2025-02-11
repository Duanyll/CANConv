import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from .kmeans import KMeans, get_cluster_centers
from .pwac import filter_indice, dispatch_indice, permute, inverse_permute, batched_matmul_conv
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


class CANConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cluster_num=32,
                 kernel_size=3,
                 mlp_inner_dims=16,
                 bias="cluster",  # or "global_param" or "global_adaptive" or "none"
                 detach_centroid=False,
                 cluster_source="channel",  # "spatial" or "pixel"
                 kernel_generator="low_rank",  # or "weighted_sum" or "low_rank"
                 kernel_count=8,  # required when kernel_generator is "weighted_sum"
                 cluster_ablation="none",  # or "global" or "pixelwise"
                 filter_threshold=0,
                 ) -> None:
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            cluster_num: Number of clusters
            kernel_size: Kernel size
            mlp_inner_dims: Number of hidden units in for the MLP that generates the kernel
            bias: "none" for no bias, "cluster" for bias for each cluster, "global_param" use a uniform bias like nn.Conv2d, 
                  "global_adaptive" generates global bias like LAGConv
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2)
        self.kernel_area = kernel_size ** 2
        self.cluster_num = cluster_num
        self.bias_mode = bias
        self.detatch_centroid = detach_centroid
        self.cluster_source = cluster_source
        self.kernel_generator = kernel_generator
        self.cluster_ablation = cluster_ablation
        self.filter_threshold = filter_threshold

        self.kmeans = KMeans(cluster_num)

        if self.kernel_generator == "spatial":
            self.centroid_to_kernel = nn.Sequential(
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=self.kernel_area),
                nn.Sigmoid()
            )
            self.kernels = nn.parameter.Parameter(
                torch.randn(self.in_channels, self.kernel_area, self.out_channels))
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        elif self.kernel_generator == "weighted_sum":
            self.centroid_to_kernel = nn.Sequential(
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=kernel_count),
                nn.Softmax()
            )
            self.kernels = nn.parameter.Parameter(
                torch.randn(kernel_count, self.in_channels * self.kernel_area, self.out_channels))
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        elif self.kernel_generator == "low_rank":
            self.kernel_head = nn.Sequential(
                nn.Linear(self.in_channels * self.kernel_area, mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(mlp_inner_dims, mlp_inner_dims),
                nn.ReLU(),
            )
            self.to_area = nn.Linear(mlp_inner_dims, self.kernel_area)
            self.to_cin = nn.Linear(mlp_inner_dims, self.in_channels)
            self.to_cout = nn.Linear(mlp_inner_dims, self.out_channels)
            self.kernels = nn.parameter.Parameter(
                torch.randn(self.in_channels, self.kernel_area, self.out_channels))
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        else:
            raise ValueError(
                "kernel_generator must be either 'spatial' or 'weighted_sum' or 'low_rank'")

        if bias == "cluster":
            self.centroid_to_bias = nn.Sequential(
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=self.out_channels),
            )
        elif bias == "global_param":
            self.bias = nn.parameter.Parameter(
                torch.randn(self.out_channels))
        elif bias == "global_adaptive":
            self.global_bias = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1)
            )
        elif bias == "none":
            self.bias = None

    def generate_kernel(self, centroids: torch.Tensor):
        """
        Args:
            centroids: (batch_size, cluster_num, patch_dims)
        Returns:
            kernel_by_cluster: (batch_size, cluster_num, in_channels * kernel_area, out_channels)
        """
        if self.kernel_generator == "spatial":
            spatial_weights = rearrange(
                self.centroid_to_kernel(centroids), 'b k area -> b k 1 area 1')
            kernel_by_cluster = rearrange(
                spatial_weights * self.kernels, 'b k cin area cout -> b k (cin area) cout')
        elif self.kernel_generator == "weighted_sum":
            kernel_weights = rearrange(
                self.centroid_to_kernel(centroids), 'b k n -> b k n 1 1')
            kernel_by_cluster = reduce(
                kernel_weights * self.kernels, 'b k n cinarea cout -> b k cinarea cout', 'sum')
        else:
            kf = self.kernel_head(centroids)
            w_cin = rearrange(F.sigmoid(self.to_cin(kf)),
                              'b k cin -> b k cin 1 1')
            w_area = rearrange(F.sigmoid(self.to_area(kf)),
                               'b k area -> b k 1 area 1')
            w_cout = rearrange(F.sigmoid(self.to_cout(kf)),
                               'b k cout -> b k 1 1 cout')
            kernel_by_cluster = (w_cin * w_area * w_cout) * self.kernels
            kernel_by_cluster = rearrange(
                kernel_by_cluster, 'b k cin area cout -> b k (cin area) cout')

        return kernel_by_cluster

    def generate_bias(self, centroids: torch.Tensor, x):
        """
        Args:
            centroids: (batch_size, cluster_num, patch_dims)
        Returns:
            bias_by_cluster: (batch_size, cluster_num, out_channels)
        """
        if self.bias_mode == "cluster":
            return self.centroid_to_bias(centroids)
        elif self.bias_mode == "global_param":
            return self.bias
        elif self.bias_mode == "global_adaptive":
            return rearrange(self.global_bias(x), 'b cout 1 1 -> b 1 cout')
        elif self.bias_mode == "none":
            return None

    def downsample_to_cluster_feature(self, x: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch_size, patch_num, patch_dims)
        Returns:
            res: (batch_size, patch_num, cluster_feature_dims)
        """
        if self.cluster_source == "channel":
            return reduce(patches, 'b s (cin area) -> b s cin', 'mean', cin=self.in_channels, area=self.kernel_area)
        elif self.cluster_source == "spatial":
            return reduce(patches, 'b s (cin area) -> b s area', 'mean', area=self.kernel_area)
        elif self.cluster_source == "pixel":
            return rearrange(x, 'b cin h w -> b (h w) cin')
        else:
            raise ValueError(
                "cluster_source must be either 'channel', 'spatial', or 'pixel'")

    def convolution_by_cluster(self, patches: torch.Tensor, indice: torch.Tensor, weight: torch.Tensor, bias=None):
        """
        Args:
            patches: (batch_size, patch_num, patch_dims)
            indice: (batch_size, patch_num)
            weight: (batch_size, cluster_num, in_channels * kernel_area, out_channels)
            bias: (batch_size, cluster_num, out_channels)
        Returns:
            res: (batch_size, patch_num, out_channels)
        """
        b = patches.shape[0]
        k = weight.shape[1]

        patches = rearrange(patches, "b s f -> (b s) f")
        weight = rearrange(weight, "b k f cout -> (b k) f cout")
        indice = indice + torch.arange(b, device=indice.device).view(-1, 1) * k
        indice = rearrange(indice, "b hw -> (b hw)")
        if bias is not None:
            bias = rearrange(bias, "b k cout -> (b k) cout")

        indice_perm, padded_patch_num, cluster_size_sorted, permuted_offset, cluster_perm, batch_height = dispatch_indice(
            indice, b * k)
        input_permuted = permute(patches, indice_perm, padded_patch_num)
        output_permuted = batched_matmul_conv(
            input_permuted, weight, permuted_offset, cluster_perm, batch_height, bias)
        output = inverse_permute(output_permuted, indice_perm)

        return rearrange(output, "(b hw) cout -> b hw cout", b=b)

    def cluster_ablation_global_forward(self, x: torch.Tensor):
        b, cin, h, w = x.shape
        patches = self.unfold(x)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=h, w=w)
        centroids = reduce(patches, 'b s f -> b 1 f', 'mean')
        kernel = self.generate_kernel(centroids)
        bias = self.generate_bias(centroids, x)
        result = torch.matmul(patches, rearrange(
            kernel, 'b 1 f cout -> b f cout')) + bias
        return rearrange(result, 'b (h w) cout -> b cout h w', h=h, w=w), torch.zeros(b, h*w, dtype=torch.long, device=x.device)

    def cluster_ablation_pixelwise_forward(self, x: torch.Tensor):
        b, cin, h, w = x.shape
        patches = self.unfold(x)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=h, w=w)
        kernel = self.generate_kernel(patches)
        bias = self.generate_bias(patches, x)
        result = torch.matmul(rearrange(patches, 'b s f -> b s 1 f'),
                              kernel) + rearrange(bias, 'b s cout -> b s 1 cout')
        return rearrange(result, 'b (h w) 1 cout -> b cout h w', h=h, w=w), repeat(torch.arange(h*w, device=x.device), 's -> b s', b=b)

    # When cluster_override is given, the module will not perform clustering and use the given indice instead
    # When cluster_override is not persent and cache_indice is given, the module will try to use the cached indice
    # When both cluster_override and cache_indice are not present, the module will always perform clustering
    # The second return value is the indice used for clustering
    def forward(self, x: torch.Tensor, cache_indice=None, cluster_override=None):
        if self.cluster_ablation == "global":
            return self.cluster_ablation_global_forward(x)
        elif self.cluster_ablation == "pixelwise":
            return self.cluster_ablation_pixelwise_forward(x)

        batch_size = x.shape[0]
        in_channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        # Step 1: Unfold x into patches and cluster them
        patches = self.unfold(x)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=height, w=width)

        if cluster_override is not None:
            cluster_indice = cluster_override
        else:
            cluster_indice = self.kmeans(self.downsample_to_cluster_feature(
                x, patches), cache_indice=cache_indice)
            if self.filter_threshold > 0:
                cluster_indice = filter_indice(
                    cluster_indice, self.cluster_num, self.filter_threshold).to(x.device)

        # Step 2: Calculate centroids for each cluster
        centroids = get_cluster_centers(
            patches, cluster_indice, self.cluster_num + 1 if self.filter_threshold > 0 else self.cluster_num)
        if self.filter_threshold > 0:
            global_center = reduce(patches, 'b s f -> b f', 'mean')
            centroids[:, self.cluster_num, :] = global_center

        if self.detatch_centroid:
            centroids = centroids.detach()
        # centroids = reduce(centroids, 'b k (cin area) -> b k cin',
        #                    'mean', cin=self.in_channels, area=self.kernel_area)

        # Step 3: Generate kernels from each centroid
        kernel_by_cluster = self.generate_kernel(centroids)
        bias = self.generate_bias(centroids, x)

        # Step 4: Apply Convolution
        # result = self.convolution_by_cluster(
        #     patches, cluster_indice, kernel_by_cluster)
        if self.bias_mode == "cluster":
            result = self.convolution_by_cluster(
                patches, cluster_indice, kernel_by_cluster, bias)
        else:
            result = self.convolution_by_cluster(
                patches, cluster_indice, kernel_by_cluster)
            if self.bias_mode == "global_param" or self.bias_mode == "global_adaptive":
                result += bias
        return rearrange(result, 'b (h w) cout -> b cout h w', h=height, w=width), cluster_indice


def test_kmconv_layer():
    dev = torch.device("cuda:0")
    module = CANConv(32, 32).to(dev)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step in range(10):
            with record_function('single_run'):
                x = torch.randn(1, 32, 64, 64, device=dev)
                y = module(x)
                # print(y.shape)
            prof.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    test_kmconv_layer()
