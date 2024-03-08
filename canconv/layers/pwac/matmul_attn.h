#pragma once

#include "utils.h"

torch::Tensor batched_matmul_qk(torch::Tensor q, torch::Tensor k, torch::Tensor permuted_offset,
                                torch::Tensor batch_height, torch::Tensor attn_offset, int attn_size);

torch::Tensor batched_matmul_qkv(torch::Tensor attn_map, torch::Tensor v, torch::Tensor permuted_offset,
                                 torch::Tensor batch_height, torch::Tensor attn_offset, int padded_patch_num);

torch::Tensor batched_softmax(torch::Tensor x, torch::Tensor batch_height, torch::Tensor attn_offset);