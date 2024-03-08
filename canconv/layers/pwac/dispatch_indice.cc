#include "dispatch_indice.h"
#include "permute_kernel.cuh"

torch::Tensor filter_indice(torch::Tensor indice, int cluster_num, float threshold_ratio) {
  int batch_size = indice.size(0);
  int patch_num = indice.size(1);
  indice = indice.cpu();
  int threshold = static_cast<int>(threshold_ratio * patch_num);

  auto cluster_size = torch::zeros({batch_size, cluster_num + 1}, torch::kLong);
  auto h_cluster_size = cluster_size.data_ptr<int64_t>();
  auto h_indice = indice.data_ptr<int64_t>();
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < patch_num; i++) {
      h_cluster_size[b * (cluster_num + 1) + h_indice[b * patch_num + i]]++;
    }
  }

  auto res_cluster_size = torch::zeros({batch_size, cluster_num + 1}, torch::kLong);
  auto res_indice = torch::zeros({batch_size, patch_num}, torch::kLong);
  auto h_res_indice = res_indice.data_ptr<int64_t>();
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < patch_num; i++) {
      int cluster_idx = h_indice[b * patch_num + i];
      if (h_cluster_size[b * (cluster_num + 1) + cluster_idx] <= threshold) {
        h_res_indice[b * patch_num + i] = cluster_num;
      } else {
        h_res_indice[b * patch_num + i] = cluster_idx;
      }
    }
  }

  return res_indice;
}

std::tuple<torch::Tensor, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatch_indice(torch::Tensor indice, int cluster_num) {
  int patch_num = indice.size(0);
  indice = indice.cpu();

  TORCH_CHECK(indice.dtype() == torch::kLong, "indice must be long");

  // Compute cluster size
  auto cluster_size = torch::zeros({cluster_num}, torch::kLong);
  auto cluster_offset = torch::zeros({cluster_num}, torch::kLong);
  auto h_indice = indice.data_ptr<int64_t>();
  auto h_cluster_size = cluster_size.data_ptr<int64_t>();
  auto h_cluster_offset = cluster_offset.data_ptr<int64_t>();
  for (int i = 0; i < patch_num; i++) {
    h_cluster_size[h_indice[i]]++;
  }

  std::tuple<torch::Tensor, torch::Tensor> sort_result = torch::sort(cluster_size, 0, true);
  auto cluster_size_sorted = std::get<0>(sort_result);
  auto cluster_perm = std::get<1>(sort_result);
  auto batch_height = torch::zeros({DIV_CEIL(cluster_num, MATRIX_MUL_TILE)}, torch::kLong);
  auto permuted_offset = torch::zeros({cluster_num}, torch::kLong);
  auto h_cluster_size_sorted = cluster_size_sorted.data_ptr<int64_t>();
  auto h_cluster_perm = cluster_perm.data_ptr<int64_t>();
  auto h_batch_height = batch_height.data_ptr<int64_t>();
  auto h_permuted_offset = permuted_offset.data_ptr<int64_t>();
  int padded_patch_num = 0;
  for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
    int batchCount = std::min(cluster_num - i, MATRIX_MUL_TILE);
    int batchHeight = h_cluster_size_sorted[i];
    h_batch_height[i / MATRIX_MUL_TILE] = batchHeight;
    for (int j = 0; j < batchCount; j++) {
      h_cluster_offset[h_cluster_perm[i + j]] = padded_patch_num;
      h_permuted_offset[i + j] = padded_patch_num;
      padded_patch_num += batchHeight;
    }
  }

  auto cluster_cursor = torch::clone(cluster_offset);
  auto indice_perm = torch::full({patch_num}, -1, torch::kLong);
  auto h_cluster_cursor = cluster_cursor.data_ptr<int64_t>();
  auto h_indice_perm = indice_perm.data_ptr<int64_t>();

  for (int i = 0; i < patch_num; i++) {
    int out_idx = h_cluster_cursor[h_indice[i]]++;
    h_indice_perm[i] = out_idx;
  }

  return std::make_tuple(std::move(indice_perm), padded_patch_num, std::move(cluster_size_sorted),
                         std::move(permuted_offset), std::move(cluster_perm), std::move(batch_height));
  // indice_perm: [patch_num]
  //     input_permuted[indice_perm] = input
  // padded_patch_num: int
  //     input_permuted should be allocated with size [padded_patch_num, ...]
  // cluster_size_sorted: [cluster_num]
  //     the size of each cluster in input_permuted
  // permuted_offset: [cluster_num]
  //     the start offset of each cluster in input_permuted
  // cluster_perm: [cluster_num]
  //     the index in original input of each cluster in input_permuted
  // batch_height: [DIV_CEIL(cluster_num, MATRIX_MUL_TILE)]
  //     the height of matrices in each batch in input_permuted
}

std::tuple<int, torch::Tensor> dispatch_attn_offset(torch::Tensor batch_height, int cluster_num) {
  batch_height = batch_height.cpu().contiguous();
  auto h_batch_height = batch_height.data_ptr<int64_t>();

  auto attn_offset = torch::zeros({cluster_num}, torch::kLong);
  auto h_attention_offset = attn_offset.data_ptr<int64_t>();

  int cur = 0;
  for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
    int batch_count = std::min(cluster_num - i, MATRIX_MUL_TILE);
    int batch_height = h_batch_height[i / MATRIX_MUL_TILE];
    for (int j = i; j < i + batch_count; j++) {
      h_attention_offset[j] = cur;
      cur += batch_height * batch_height;
    }
  }

  return std::make_tuple(cur, std::move(attn_offset));
}