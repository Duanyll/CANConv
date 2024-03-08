#pragma once

#include "utils.h"

torch::Tensor matmul(torch::Tensor a, torch::Tensor b);
torch::Tensor batched_matmul_conv(torch::Tensor input_permuted, torch::Tensor weight, torch::Tensor permuted_offset,
                                  torch::Tensor cluster_perm, torch::Tensor batch_height);
torch::Tensor batched_matmul_conv_bias(torch::Tensor input_permuted, torch::Tensor weight,
                                       torch::Tensor permuted_offset, torch::Tensor cluster_perm,
                                       torch::Tensor batch_height, torch::Tensor bias);