#include "permute.h"
#include "permute_kernel.cuh"

using namespace torch::autograd;

class PermuteFunction : public Function<PermuteFunction> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input, torch::Tensor indice_perm,
                               int padded_patch_num) {
    indice_perm = indice_perm.to(torch::kCUDA).contiguous();
    input = input.to(torch::kCUDA).contiguous();
    ctx->saved_data["indice_perm"] = indice_perm;
    int indice_num = indice_perm.size(0);
    int feature_num = input.size(1);
    auto output = torch::zeros({padded_patch_num, feature_num}, input.options());
    permute_impl(input.data_ptr<float>(), output.data_ptr<float>(), indice_perm.data_ptr<int64_t>(), indice_num,
                 feature_num);
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto indice_perm = ctx->saved_data["indice_perm"].toTensor();
    auto grad_output = grad_outputs[0].to(torch::kCUDA).contiguous();
    int indice_num = indice_perm.size(0);
    int feature_num = grad_output.size(1);
    auto grad_input = torch::zeros({indice_num, feature_num}, grad_output.options());
    inverse_permute_impl(grad_output.data_ptr<float>(), grad_input.data_ptr<float>(),
                         indice_perm.data_ptr<int64_t>(), indice_num, feature_num);
    return {grad_input, torch::Tensor(), torch::Tensor()};
  }
};

class InversePermuteFunction : public Function<InversePermuteFunction> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input, torch::Tensor indice_perm) {
    indice_perm = indice_perm.to(torch::kCUDA).contiguous();
    input = input.to(torch::kCUDA).contiguous();
    ctx->saved_data["indice_perm"] = indice_perm;
    ctx->saved_data["padded_patch_num"] = input.size(0);
    int indice_num = indice_perm.size(0);
    int feature_num = input.size(1);
    auto output = torch::zeros({indice_num, feature_num}, input.options());
    inverse_permute_impl(input.data_ptr<float>(), output.data_ptr<float>(), indice_perm.data_ptr<int64_t>(),
                         indice_num, feature_num);
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto indice_perm = ctx->saved_data["indice_perm"].toTensor();
    auto grad_output = grad_outputs[0].to(torch::kCUDA).contiguous();
    int indice_num = indice_perm.size(0);
    int feature_num = grad_output.size(1);
    auto grad_input = torch::zeros({ctx->saved_data["padded_patch_num"].toInt(), feature_num}, grad_output.options());
    permute_impl(grad_output.data_ptr<float>(), grad_input.data_ptr<float>(), indice_perm.data_ptr<int64_t>(),
                 indice_num, feature_num);
    return {grad_input, torch::Tensor()};
  }
};

torch::Tensor permute(torch::Tensor input, torch::Tensor indice_perm, int padded_patch_num) {
  return PermuteFunction::apply(input, indice_perm, padded_patch_num);
}

torch::Tensor inverse_permute(torch::Tensor input, torch::Tensor indice_perm) {
  return InversePermuteFunction::apply(input, indice_perm);
}