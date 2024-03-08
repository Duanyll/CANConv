#include "matmul_conv.h"
#include "permute_kernel.cuh"
#include "utils.h"

using namespace torch::autograd;

class MatmulFunction : public Function<MatmulFunction> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kFloat64 && b.dtype() == torch::kFloat64, "matmul requires float32 tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul requires 2D tensors");
    TORCH_CHECK(a.size(1) == b.size(0), "matmul requires size match");

    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    a = a.cuda().contiguous();
    b = b.cuda().contiguous();
    ctx->saved_data["a"] = a;
    ctx->saved_data["b"] = b;
    auto c = torch::zeros({m, n}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

    double alpha = 1.0f;
    double beta = 0.0f;
    // C = A @ B
    // C^T = B^T @ A^T
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b.data_ptr<double>(), n, a.data_ptr<double>(), k,
                &beta, c.data_ptr<double>(), n);

    return c;
  }

  static torch::autograd::tensor_list backward(AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
    auto a = ctx->saved_data["a"].toTensor().cuda().contiguous();
    auto b = ctx->saved_data["b"].toTensor().cuda().contiguous();
    auto grad_output = grad_outputs[0].cuda().contiguous();

    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    auto grad_a = torch::zeros_like(a);
    auto grad_b = torch::zeros_like(b);

    double alpha = 1.0f;
    double beta = 0.0f;

    // grad_a = grad_output @ B^T
    // grad_a^T = B @ grad_output^T
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, b.data_ptr<double>(), n,
                grad_output.data_ptr<double>(), n, &beta, grad_a.data_ptr<double>(), k);

    // grad_b = A^T @ grad_output
    // grad_b^T = grad_output^T @ A
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output.data_ptr<double>(), n,
                a.data_ptr<double>(), k, &beta, grad_b.data_ptr<double>(), n);

    return {grad_a, grad_b};
  }
};

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) { return MatmulFunction::apply(a, b); }

// Run batched matmul with shape specified by dispatch_indice between input and kernel
class BatchedMatmulConvFunction : public Function<BatchedMatmulConvFunction> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input_permuted, torch::Tensor weight,
                               torch::Tensor permuted_offset, torch::Tensor cluster_perm, torch::Tensor batch_height,
                               torch::Tensor bias, bool has_bias) {
    input_permuted = input_permuted.cuda().contiguous();   // patch_num * feature_size
    weight = weight.cuda().contiguous();                   // cluster_num * feature_size * out_channels
    permuted_offset = permuted_offset.cuda().contiguous(); // cluster_num
    cluster_perm = cluster_perm.cuda().contiguous();       // cluster_num
    batch_height = batch_height.cpu().contiguous();        // ceil(cluster_num / MATRIX_MUL_TILE)

    int patch_num = input_permuted.size(0);
    int feature_size = input_permuted.size(1);
    int out_channels = weight.size(2);
    int cluster_num = permuted_offset.size(0);

    auto output_permuted = torch::zeros({patch_num, out_channels}, input_permuted.options());
    if (has_bias) {
      bias = bias.cuda().contiguous();                           // cluster_num * out_channels
      auto batch_height_cuda = batch_height.cuda().contiguous(); // ceil(cluster_num / MATRIX_MUL_TILE)
      fill_bias_impl(output_permuted.data_ptr<float>(), bias.data_ptr<float>(), permuted_offset.data_ptr<int64_t>(),
                     cluster_perm.data_ptr<int64_t>(), batch_height_cuda.data_ptr<int64_t>(), cluster_num,
                     out_channels);
    }

    ctx->saved_data["input_permuted"] = input_permuted;
    ctx->saved_data["weight"] = weight;
    ctx->saved_data["permuted_offset"] = permuted_offset;
    ctx->saved_data["cluster_perm"] = cluster_perm;
    ctx->saved_data["batch_height"] = batch_height;
    ctx->saved_data["has_bias"] = has_bias;
    if (has_bias) {
      ctx->saved_data["bias"] = bias;
    }

    // Prepare pointers for batched matrix multiplication
    auto input_matrix_ptr = int64_t(input_permuted.data_ptr<float>()) + permuted_offset * feature_size * 4;
    auto weight_matrix_ptr = int64_t(weight.data_ptr<float>()) + cluster_perm * feature_size * out_channels * 4;
    auto output_matrix_ptr = int64_t(output_permuted.data_ptr<float>()) + permuted_offset * out_channels * 4;

    auto d_input_matrix_ptr = (float **)input_matrix_ptr.data_ptr<int64_t>();
    auto d_weight_matrix_ptr = (float **)weight_matrix_ptr.data_ptr<int64_t>();
    auto d_output_matrix_ptr = (float **)output_matrix_ptr.data_ptr<int64_t>();
    auto h_batch_height = batch_height.data_ptr<int64_t>();

    // Cublas batched matrix multiplication
    // C(m x n) = A(m x k) @ B(k x n)
    // m = cluster_size_sorted[i]
    // n = out_channels
    // k = feature_size
    float alpha = 1.0f;
    float beta = has_bias ? 1.0f : 0.0f;
    for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
      int batchCount = std::min(cluster_num - i, MATRIX_MUL_TILE);
      if (h_batch_height[i / MATRIX_MUL_TILE] == 0) {
        continue;
      }
      // Cublas expects column-major matrices, however we use row-major matrices in PyTorch.
      // We may utilize that C = A @ B is equivalent to C^T = B^T @ A^T.
      // See https://blog.csdn.net/qq_25147897/article/details/70806487
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle,
                                            CUBLAS_OP_N,                             // transa
                                            CUBLAS_OP_N,                             // transb
                                            out_channels,                            // m => n
                                            h_batch_height[i / MATRIX_MUL_TILE],     // n => m
                                            feature_size,                            // k
                                            &alpha,                                  // alpha
                                            (const float **)d_weight_matrix_ptr + i, // A
                                            out_channels,                            // lda => n
                                            (const float **)d_input_matrix_ptr + i,  // B
                                            feature_size,                            // ldb => k
                                            &beta,                                   // beta
                                            d_output_matrix_ptr + i,                 // C
                                            out_channels,                            // ldc => n
                                            batchCount                               // batchCount
                                            ));
    }

    return output_permuted;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto input_permuted = ctx->saved_data["input_permuted"].toTensor();
    auto weight = ctx->saved_data["weight"].toTensor();
    auto permuted_offset = ctx->saved_data["permuted_offset"].toTensor();
    auto cluster_perm = ctx->saved_data["cluster_perm"].toTensor();
    auto batch_height = ctx->saved_data["batch_height"].toTensor();
    auto has_bias = ctx->saved_data["has_bias"].toBool();
    torch::Tensor bias;
    if (has_bias) {
      bias = ctx->saved_data["bias"].toTensor();
    }

    auto grad_output_permuted = grad_outputs[0].cuda().contiguous();

    int patch_num = input_permuted.size(0);
    int feature_size = input_permuted.size(1);
    int out_channels = weight.size(2);
    int cluster_num = permuted_offset.size(0);

    // Prepare pointers for batched matrix multiplication
    auto grad_input_permuted = torch::zeros_like(input_permuted);
    auto grad_weight = torch::zeros_like(weight);

    auto grad_output_matrix_ptr = int64_t(grad_output_permuted.data_ptr<float>()) + permuted_offset * out_channels * 4;
    auto grad_input_matrix_ptr = int64_t(grad_input_permuted.data_ptr<float>()) + permuted_offset * feature_size * 4;
    auto grad_weight_matrix_ptr =
        int64_t(grad_weight.data_ptr<float>()) + cluster_perm * feature_size * out_channels * 4;
    auto input_matrix_ptr = int64_t(input_permuted.data_ptr<float>()) + permuted_offset * feature_size * 4;
    auto weight_matrix_ptr = int64_t(weight.data_ptr<float>()) + cluster_perm * feature_size * out_channels * 4;

    auto d_grad_output_matrix_ptr = (float **)grad_output_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_input_matrix_ptr = (float **)grad_input_matrix_ptr.data_ptr<int64_t>();
    auto d_grad_weight_matrix_ptr = (float **)grad_weight_matrix_ptr.data_ptr<int64_t>();
    auto d_input_matrix_ptr = (float **)input_matrix_ptr.data_ptr<int64_t>();
    auto d_weight_matrix_ptr = (float **)weight_matrix_ptr.data_ptr<int64_t>();

    auto h_batch_height = batch_height.data_ptr<int64_t>();

    // C(m x n) = A(m x k) @ B(k x n)
    // m = cluster_size_sorted[i]
    // n = out_channels
    // k = feature_size
    // A' = C' @ B^T
    // B' = A^T @ C'
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < cluster_num; i += MATRIX_MUL_TILE) {
      int batchCount = std::min(cluster_num - i, MATRIX_MUL_TILE);
      if (h_batch_height[i / MATRIX_MUL_TILE] == 0) {
        continue;
      }
      // grad_input = grad_output @ weight^T
      // in cublas transpose, A is weight and B is grad_output
      // grad_input^T = weight @ grad_output^T
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle,
                                            CUBLAS_OP_T,                                  // transa
                                            CUBLAS_OP_N,                                  // transb
                                            feature_size,                                 // m => k
                                            h_batch_height[i / MATRIX_MUL_TILE],          // n => m
                                            out_channels,                                 // k => n
                                            &alpha,                                       // alpha
                                            (const float **)d_weight_matrix_ptr + i,      // A
                                            out_channels,                                 // lda => n
                                            (const float **)d_grad_output_matrix_ptr + i, // B
                                            out_channels,                                 // ldb => n
                                            &beta,                                        // beta
                                            d_grad_input_matrix_ptr + i,                  // C
                                            feature_size,                                 // ldc => k
                                            batchCount                                    // batchCount
                                            ));

      // grad_weight^T = grad_output^T @ input
      CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle,
                                            CUBLAS_OP_N,                                  // transa
                                            CUBLAS_OP_T,                                  // transb
                                            out_channels,                                 // m => n
                                            feature_size,                                 // n => k
                                            h_batch_height[i / MATRIX_MUL_TILE],          // k => m
                                            &alpha,                                       // alpha
                                            (const float **)d_grad_output_matrix_ptr + i, // A
                                            out_channels,                                 // lda => n
                                            (const float **)d_input_matrix_ptr + i,       // B
                                            feature_size,                                 // ldb => k
                                            &beta,                                        // beta
                                            d_grad_weight_matrix_ptr + i,                 // C
                                            out_channels,                                 // ldc => n
                                            batchCount                                    // batchCount
                                            ));
    }

    torch::Tensor grad_bias;
    if (has_bias) {
      grad_bias = torch::zeros_like(bias);
      auto batch_height_cuda = batch_height.to(torch::kCUDA);
      bias_backward_impl(grad_output_permuted.data_ptr<float>(), grad_bias.data_ptr<float>(),
                         permuted_offset.data_ptr<int64_t>(), cluster_perm.data_ptr<int64_t>(),
                         batch_height_cuda.data_ptr<int64_t>(), cluster_num, out_channels);
    }

    // std::cout << "grad_output_permuted" << std::endl;
    // std::cout << grad_output_permuted << std::endl;
    // std::cout << "grad_input_permuted" << std::endl;
    // std::cout << grad_input_permuted << std::endl;

    return {grad_input_permuted, grad_weight, torch::Tensor(), torch::Tensor(),
            torch::Tensor(),     grad_bias,   torch::Tensor()};
  }
};

torch::Tensor batched_matmul_conv(torch::Tensor input_permuted, torch::Tensor weight, torch::Tensor permuted_offset,
                                  torch::Tensor cluster_perm, torch::Tensor batch_height) {
  return BatchedMatmulConvFunction::apply(input_permuted, weight, permuted_offset, cluster_perm, batch_height,
                                          torch::empty(0), false);
}

torch::Tensor batched_matmul_conv_bias(torch::Tensor input_permuted, torch::Tensor weight,
                                       torch::Tensor permuted_offset, torch::Tensor cluster_perm,
                                       torch::Tensor batch_height, torch::Tensor bias) {
  return BatchedMatmulConvFunction::apply(input_permuted, weight, permuted_offset, cluster_perm, batch_height, bias,
                                          true);
}