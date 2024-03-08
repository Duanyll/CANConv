#include "permute_kernel.cuh"
#define DIV_CEIL(x, y) (((x) + (y)-1) / (y))

__global__ void inverse_permute_kernel(float *input, float *output, int64_t *index, int64_t indice_num, int64_t feature_size) {
  float *input_ptr = input + index[blockIdx.x] * feature_size;
  float *output_ptr = output + blockIdx.x * feature_size;
  for (int i = threadIdx.x; i < feature_size; i += WARP_SIZE) {
    output_ptr[i] = input_ptr[i];
  }
}

__global__ void permute_kernel(float *input, float *output, int64_t *index, int64_t indice_num, int64_t feature_size) {
  float *input_ptr = input + blockIdx.x * feature_size;
  float *output_ptr = output + index[blockIdx.x] * feature_size;
  for (int i = threadIdx.x; i < feature_size; i += WARP_SIZE) {
    output_ptr[i] = input_ptr[i];
  }
}

__global__ void fill_bias_kernel(float *output_permuted, float *bias, int64_t *permuted_offest, int64_t *cluster_perm,
                                 int64_t *batch_height, int64_t total_cluster, int64_t out_channels) {
  int64_t cluster_idx = blockIdx.x;
  if (cluster_idx >= total_cluster) {
    return;
  }
  float *input_ptr = bias + cluster_perm[cluster_idx] * out_channels;
  float *output_ptr = output_permuted + permuted_offest[cluster_idx] * out_channels;
  int64_t height = batch_height[cluster_idx / MATRIX_MUL_TILE];
  int64_t feature_idx = blockIdx.y * WARP_SIZE + threadIdx.x;
  if (feature_idx >= out_channels) {
    return;
  }
  float bias_value = input_ptr[feature_idx];
  for (int i = 0; i < height; i++) {
    output_ptr[i * out_channels + feature_idx] = bias_value;
  }
}

__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int64_t *permuted_offest,
                                     int64_t *cluster_perm, int64_t *batch_height, int64_t total_cluster,
                                     int64_t out_channels) {
  int64_t cluster_idx = blockIdx.x;
  if (cluster_idx >= total_cluster) {
    return;
  }
  float *input_ptr = grad_output + permuted_offest[cluster_idx] * out_channels;
  float *output_ptr = grad_bias + cluster_perm[cluster_idx] * out_channels;
  int64_t height = batch_height[cluster_idx / MATRIX_MUL_TILE];
  int64_t feature_idx = blockIdx.y * WARP_SIZE + threadIdx.x;
  if (feature_idx >= out_channels) {
    return;
  }
  float sum = 0.0f;
  for (int i = 0; i < height; i++) {
    sum += input_ptr[i * out_channels + feature_idx];
  }
  output_ptr[feature_idx] = sum;
}

void inverse_permute_impl(float *input, float *output, int64_t *index, int64_t indice_num, int64_t feature_size) {
  inverse_permute_kernel<<<indice_num, WARP_SIZE>>>(input, output, index, indice_num, feature_size);
}

void permute_impl(float *input, float *output, int64_t *index, int64_t indice_num, int64_t feature_size) {
  permute_kernel<<<indice_num, WARP_SIZE>>>(input, output, index, indice_num, feature_size);
}

void fill_bias_impl(float *output_permuted, float *bias, int64_t *permuted_offest, int64_t *cluster_perm,
                    int64_t *batch_height, int64_t total_cluster, int64_t out_channels) {
  dim3 grid(total_cluster, DIV_CEIL(out_channels, WARP_SIZE));
  fill_bias_kernel<<<grid, WARP_SIZE>>>(output_permuted, bias, permuted_offest, cluster_perm, batch_height,
                                        total_cluster, out_channels);
}

void bias_backward_impl(float *grad_output, float *grad_bias, int64_t *permuted_offest, int64_t *cluster_perm,
                        int64_t *batch_height, int64_t total_cluster, int64_t out_channels) {
  dim3 grid(total_cluster, DIV_CEIL(out_channels, WARP_SIZE));
  bias_backward_kernel<<<grid, WARP_SIZE>>>(grad_output, grad_bias, permuted_offest, cluster_perm, batch_height,
                                            total_cluster, out_channels);
}