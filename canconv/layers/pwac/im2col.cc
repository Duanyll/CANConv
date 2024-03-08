#include "im2col.h"
#include "utils.h"

using namespace torch::autograd;

/// @brief Cluster convolution by im2col
/// @param input Unfolded input tensor [B, (H * W), (Cin * Kh * Kw)]
/// @param weight Clustered weight tensor [B, K, (Cin * Kh * Kw), Cout]
/// @param indice Cluster indice for each input patch [B, (H * W)]
/// @return Output tensor [B, (H * W), Cout]
torch::Tensor conv_by_im2col_cluster_nested(torch::Tensor input, torch::Tensor weight, torch::Tensor indice) {
  int batch_size = input.size(0);
  int patch_num = input.size(1);
  int feature_size = input.size(2);
  int out_channels = weight.size(3);
  int cluster_num = weight.size(1);

  TORCH_CHECK(weight.size(0) == batch_size, "Weight batch size does not match input batch size");
  TORCH_CHECK(weight.size(2) == feature_size, "Weight feature size does not match input feature size");
  TORCH_CHECK(indice.size(0) == batch_size, "Indice batch size does not match input batch size");
  TORCH_CHECK(indice.size(1) == patch_num, "Indice patch num does not match input patch num");
  TORCH_CHECK(input.device() == weight.device(), "Input and weight are not on the same device");

  int total_cluster = batch_size * cluster_num;
  int total_patch = batch_size * patch_num;

  indice = indice.to(torch::kCPU);
  indice += torch::arange(0, batch_size, torch::kLong).view({batch_size, 1}) * cluster_num;
  indice = indice.reshape({-1}).contiguous();

  input = input.reshape({total_patch, feature_size});
  weight = weight.reshape({total_cluster, feature_size, out_channels});

  // Reorder input so that each cluster is contiguous
  // [1, 3, 2, 2, 1, 3] -> [1, 1, 2, 2, 3, 3]
  // Indice permutation: [0, 4, 2, 3, 1, 5]
  // Inverse indice permutation: [0, 5, 2, 3, 1, 4]

  auto cluster_size = torch::zeros({total_cluster}, torch::kLong);
  auto cluster_offset = torch::zeros({total_cluster}, torch::kLong);
  auto indice_perm = torch::zeros({total_patch}, torch::kLong);
  auto indice_inv_perm = torch::zeros({total_patch}, torch::kLong);
  auto h_indice = indice.data_ptr<int64_t>();
  auto h_cluster_size = cluster_size.data_ptr<int64_t>();
  auto h_cluster_offset = cluster_offset.data_ptr<int64_t>();
  auto h_indice_perm = indice_perm.data_ptr<int64_t>();
  auto h_indice_inv_perm = indice_inv_perm.data_ptr<int64_t>();

  for (int i = 0; i < total_patch; i++) {
    h_cluster_size[h_indice[i]]++;
  }

  for (int i = 1; i < total_cluster; i++) {
    h_cluster_offset[i] = h_cluster_offset[i - 1] + h_cluster_size[i - 1];
  }

  for (int i = 0; i < total_patch; i++) {
    int out_idx = h_cluster_offset[h_indice[i]]++;
    h_indice_perm[out_idx] = i;
    h_indice_inv_perm[i] = out_idx;
  }

  indice_perm = indice_perm.to(input.device());
  indice_inv_perm = indice_inv_perm.to(input.device());

  auto input_permuted = input.index({indice_perm});
  std::vector<torch::Tensor> input_split(total_cluster);
  for (int i = 0; i < total_cluster; i++) {
    input_split[i] = input_permuted.narrow(0, h_cluster_offset[i] - h_cluster_size[i], h_cluster_size[i]);
  }
  auto input_nested = torch::nested::as_nested_tensor(input_split);
  auto weight_nested = torch::nested::as_nested_tensor(weight.unbind());
  auto output_nested = torch::matmul(input_nested, weight_nested);
  auto output_permuted = torch::cat(output_nested.unbind(), 0);
  auto output = output_permuted.index({indice_inv_perm});

  return output.view({batch_size, patch_num, out_channels});
}