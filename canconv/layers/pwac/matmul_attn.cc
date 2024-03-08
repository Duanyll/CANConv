#include "matmul_attn.h"
#include "permute_kernel.cuh"
#include "utils.h"

using namespace torch::autograd;

class BatchedMatmulQKFunction : public Function<BatchedMatmulQKFunction> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor q, torch::Tensor k, torch::Tensor permuted_offset,
                               torch::Tensor batch_height, torch::Tensor attn_offset, int attn_size) {
    q = q.cuda().contiguous();
    k = k.cuda().contiguous();
    permuted_offset = permuted_offset.cuda().contiguous();
    batch_height = batch_height.cpu().contiguous();
    attn_offset = attn_offset.cuda().contiguous();

    int d = q.size(1);
    int cluster_num = permuted_offset.size(0);

    auto output = torch::full({attn_size}, std::numeric_limits<float>::lowest(), q.options());

    ctx->saved_data["q"] = q;
    ctx->saved_data["k"] = k;
    ctx->saved_data["permuted_offset"] = permuted_offset;
    ctx->saved_data["batch_height"] = batch_height;
    ctx->saved_data["attn_offset"] = attn_offset;

    auto q_matrix_ptr = int64_t(q.data_ptr<float>()) + permuted_offset * d * 4;
    auto k_matrix_ptr = int64_t(k.data_ptr<float>()) + permuted_offset * d * 4;
    auto output_matrix_ptr = int64_t(output.data_ptr<float>()) + attn_offset * 4;

    auto d_q_matrix_ptr = (float **)q_matrix_ptr.data_ptr<int64_t>();
    auto d_k_matrix_ptr = (float **)k_matrix_ptr.data_ptr<int64_t>();
    auto d_output_matrix_ptr = (float **)output_matrix_ptr.data_ptr<int64_t>();
    auto h_batch_height = batch_height.data_ptr<int64_t>();

    // Output(n x n) = Q(n x d) @ K(n x d).T
    // Output(n x n).T = K(n x d) @ Q(n x d).T
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
      int batchCount = std::min(cluster_num - i, MATRIX_MUL_TILE);
      int n = h_batch_height[i / MATRIX_MUL_TILE];
      if (n == 0) continue;
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &alpha, d_k_matrix_ptr + i, d,
                                            d_q_matrix_ptr + i, d, &beta, d_output_matrix_ptr + i, n, batchCount));
    }

    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto q = ctx->saved_data["q"].toTensor();
    auto k = ctx->saved_data["k"].toTensor();
    auto permuted_offset = ctx->saved_data["permuted_offset"].toTensor();
    auto batch_height = ctx->saved_data["batch_height"].toTensor();
    auto attn_offset = ctx->saved_data["attn_offset"].toTensor();

    int d = q.size(1);
    int cluster_num = permuted_offset.size(0);

    auto grad_output = grad_outputs[0].cuda().contiguous();
    auto grad_q = torch::zeros_like(q);
    auto grad_k = torch::zeros_like(k);

    auto q_matrix_ptr = int64_t(q.data_ptr<float>()) + permuted_offset * d * 4;
    auto k_matrix_ptr = int64_t(k.data_ptr<float>()) + permuted_offset * d * 4;
    auto grad_q_matrix_ptr = int64_t(grad_q.data_ptr<float>()) + permuted_offset * d * 4;
    auto grad_k_matrix_ptr = int64_t(grad_k.data_ptr<float>()) + permuted_offset * d * 4;
    auto grad_output_matrix_ptr = int64_t(grad_output.data_ptr<float>()) + attn_offset * 4;

    auto d_q_matrix_ptr = (float **)q_matrix_ptr.data_ptr<int64_t>();
    auto d_k_matrix_ptr = (float **)k_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_q_matrix_ptr = (float **)grad_q_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_k_matrix_ptr = (float **)grad_k_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_output_matrix_ptr = (float **)grad_output_matrix_ptr.data_ptr<int64_t>();

    auto h_batch_height = batch_height.data_ptr<int64_t>();

    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
      int batch_count = std::min(cluster_num - i, MATRIX_MUL_TILE);
      int n = h_batch_height[i / MATRIX_MUL_TILE];
      if (n == 0)
        continue;
      // Output = Q @ K.T
      // dQ = dOutput @ K
      // dQ.T = K.T @ dOutput.T
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, n, n, &alpha, d_k_matrix_ptr + i, d,
                                            d_grad_output_matrix_ptr + i, n, &beta, d_grad_q_matrix_ptr + i, d,
                                            batch_count));
      // dK.T = Q.T @ dOutput
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, n, n, &alpha, d_q_matrix_ptr + i, d,
                                            d_grad_output_matrix_ptr + i, n, &beta, d_grad_k_matrix_ptr + i, d,
                                            batch_count));
    }

    return {grad_q, grad_k, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor batched_matmul_qk(torch::Tensor q, torch::Tensor k, torch::Tensor permuted_offset,
                                torch::Tensor batch_height, torch::Tensor attn_offset, int attn_size) {
  return BatchedMatmulQKFunction::apply(q, k, permuted_offset, batch_height, attn_offset, attn_size);
}

class BatchedMatmulQKVFunction : public Function<BatchedMatmulQKVFunction> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor attn_map, torch::Tensor v,
                               torch::Tensor permuted_offset, torch::Tensor batch_height, torch::Tensor attn_offset,
                               int padded_patch_num) {
    attn_map = attn_map.cuda().contiguous();
    v = v.cuda().contiguous();
    permuted_offset = permuted_offset.cuda().contiguous();
    batch_height = batch_height.cpu().contiguous();
    attn_offset = attn_offset.cuda().contiguous();

    int d = v.size(1);
    int cluster_num = permuted_offset.size(0);

    auto output = torch::zeros({padded_patch_num, d}, v.options());

    ctx->saved_data["attn_map"] = attn_map;
    ctx->saved_data["v"] = v;
    ctx->saved_data["permuted_offset"] = permuted_offset;
    ctx->saved_data["batch_height"] = batch_height;
    ctx->saved_data["attn_offset"] = attn_offset;

    auto attn_map_matrix_ptr = int64_t(attn_map.data_ptr<float>()) + attn_offset * 4;
    auto v_matrix_ptr = int64_t(v.data_ptr<float>()) + permuted_offset * d * 4;
    auto output_matrix_ptr = int64_t(output.data_ptr<float>()) + permuted_offset * d * 4;

    auto d_attn_map_matrix_ptr = (float **)attn_map_matrix_ptr.data_ptr<int64_t>();
    auto d_v_matrix_ptr = (float **)v_matrix_ptr.data_ptr<int64_t>();
    auto d_output_matrix_ptr = (float **)output_matrix_ptr.data_ptr<int64_t>();
    auto h_batch_height = batch_height.data_ptr<int64_t>();

    // Output(n x d) = AttnMap(n x n) @ V(n x d)
    // Output.T = V.T @ AttnMap.T
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
      int batch_count = std::min(cluster_num - i, MATRIX_MUL_TILE);
      int n = h_batch_height[i / MATRIX_MUL_TILE];
      if (n == 0)
        continue;
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, n, n, &alpha, d_v_matrix_ptr + i, d,
                                            d_attn_map_matrix_ptr + i, n, &beta, d_output_matrix_ptr + i, d,
                                            batch_count));
    }

    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto attn_map = ctx->saved_data["attn_map"].toTensor();
    auto v = ctx->saved_data["v"].toTensor();
    auto permuted_offset = ctx->saved_data["permuted_offset"].toTensor();
    auto batch_height = ctx->saved_data["batch_height"].toTensor();
    auto attn_offset = ctx->saved_data["attn_offset"].toTensor();

    int d = v.size(1);
    int cluster_num = permuted_offset.size(0);

    auto grad_output = grad_outputs[0].cuda().contiguous();
    auto grad_attn_map = torch::zeros_like(attn_map);
    auto grad_v = torch::zeros_like(v);

    auto attn_map_matrix_ptr = int64_t(attn_map.data_ptr<float>()) + attn_offset * 4;
    auto v_matrix_ptr = int64_t(v.data_ptr<float>()) + permuted_offset * d * 4;
    auto grad_attn_map_matrix_ptr = int64_t(grad_attn_map.data_ptr<float>()) + attn_offset * 4;
    auto grad_v_matrix_ptr = int64_t(grad_v.data_ptr<float>()) + permuted_offset * d * 4;
    auto grad_output_matrix_ptr = int64_t(grad_output.data_ptr<float>()) + permuted_offset * d * 4;

    auto d_attn_map_matrix_ptr = (float **)attn_map_matrix_ptr.data_ptr<int64_t>();
    auto d_v_matrix_ptr = (float **)v_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_attn_map_matrix_ptr = (float **)grad_attn_map_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_v_matrix_ptr = (float **)grad_v_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_output_matrix_ptr = (float **)grad_output_matrix_ptr.data_ptr<int64_t>();

    auto h_batch_height = batch_height.data_ptr<int64_t>();

    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
      int batch_count = std::min(cluster_num - i, MATRIX_MUL_TILE);
      int n = h_batch_height[i / MATRIX_MUL_TILE];
      if (n == 0)
        continue;
      // Output = AttnMap @ V
      // dAttnMap = dOutput @ V.T
      // dAttnMap.T = V @ dOutput.T
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &alpha, d_v_matrix_ptr + i, d,
                                            d_grad_output_matrix_ptr + i, d, &beta, d_grad_attn_map_matrix_ptr + i, n,
                                            batch_count));
      // dV = AttnMap.T @ dOutput
      // dV.T = dOutput.T @ AttnMap
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, n, n, &alpha,
                                            d_grad_output_matrix_ptr + i, d, d_attn_map_matrix_ptr + i, n, &beta,
                                            d_grad_v_matrix_ptr + i, d, batch_count));
    }

    return {grad_attn_map, grad_v, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor batched_matmul_qkv(torch::Tensor attn_map, torch::Tensor v, torch::Tensor permuted_offset,
                                 torch::Tensor batch_height, torch::Tensor attn_offset, int padded_patch_num) {
  return BatchedMatmulQKVFunction::apply(attn_map, v, permuted_offset, batch_height, attn_offset, padded_patch_num);
}

torch::Tensor batched_softmax(torch::Tensor x, torch::Tensor batch_height, torch::Tensor attn_offset) {
  batch_height = batch_height.cpu().contiguous();
  attn_offset = attn_offset.cpu().contiguous();
  int batch_count = batch_height.size(0);

  auto h_batch_height = batch_height.data_ptr<int64_t>();

  auto list = torch::tensor_split(x, attn_offset.slice(0, MATRIX_MUL_TILE, c10::nullopt, MATRIX_MUL_TILE), 0);
  for (int i = 0; i < batch_count; i++) {
    if (h_batch_height[i] == 0) continue;
    auto slice = list[i].view({-1, h_batch_height[i]});
    list[i] = torch::softmax(slice, 1).view({-1});
  }
  auto output = torch::cat(list, 0);

  return output;
}