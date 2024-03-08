import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

# x: (B, Cin, H, W)
# weight: (B, k, Cout, Cin, 3, 3)
# indice: (B, H, W) long
# This implementation is not memory efficient and takes too much time, but it is easy to understand.
def conv3x3s1p1_cluster_masked(x, weight, indice, bias=None):
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out_channels = weight.shape[2]
    cluster_num = weight.shape[1]

    result = torch.zeros(batch_size, out_channels,
                         height, width, device=x.device, dtype=x.dtype)
    zero_bias = torch.zeros(out_channels, device=x.device, dtype=x.dtype)
    for b in range(batch_size):
        output = torch.cat([F.conv2d(x[b], weight[b, i], zero_bias if bias is None else bias[b, i], padding=1).unsqueeze(0)
                            for i in range(cluster_num)], dim=0)
        mask = rearrange(F.one_hot(indice[b], cluster_num), 'H W K -> K 1 H W')
        output = reduce(output * mask, 'K C H W -> C H W', 'sum')
        result[b] = output
    return result


# This implementation is not memory efficient.
def conv3x3s1p1_cluster_gather(x, weight, indice):
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    out_channels = weight.shape[2]

    patches = F.unfold(x, kernel_size=3, padding=1)
    patches = rearrange(
        patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=9, h=height, w=width)
    kernel_by_cluster = rearrange(
        weight, 'b k cout cin kh kw -> b k (cin kh kw) cout')
    kernel_by_patch = torch.gather(kernel_by_cluster, dim=1, index=repeat(
        indice, 'b h w -> b (h w) (cin kh kw) cout', cin=in_channels, kh=3, kw=3, cout=out_channels))
    patches = rearrange(
        patches, 'b s (cin kh kw) -> b s 1 (cin kh kw)', cin=in_channels, kh=3, kw=3)
    patches = torch.matmul(patches, kernel_by_patch)
    patches = rearrange(
        patches, 'b (h w) 1 cout -> b cout h w', h=height, w=width)
    return patches
