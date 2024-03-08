#pragma once

#ifdef _DEBUG
#undef _DEBUG
#include <torch/extension.h>
#define _DEBUG 1
#else
#include <torch/extension.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

torch::Tensor conv3x3s1p1_forward_cpu(torch::Tensor input, torch::Tensor weights);
std::vector<torch::Tensor> conv3x3s1p1_backward_cpu(torch::Tensor input, torch::Tensor weights,
                                                    torch::Tensor grad_output);
torch::Tensor conv3x3s1p1_cluster_forward_cpu(torch::Tensor input, torch::Tensor weights, torch::Tensor indice);
std::vector<torch::Tensor> conv3x3s1p1_cluster_backward_cpu(torch::Tensor input, torch::Tensor weights,
                                                            torch::Tensor indice, torch::Tensor grad_output);
torch::Tensor conv3x3s1p1_forward_cuda_naive(torch::Tensor input, torch::Tensor weights);
std::vector<torch::Tensor> conv3x3s1p1_backward_cuda_naive(torch::Tensor input, torch::Tensor weights,
                                                           torch::Tensor grad_output);
torch::Tensor conv3x3s1p1_cluster_forward_cuda_naive(torch::Tensor input, torch::Tensor weights, torch::Tensor indice);
std::vector<torch::Tensor> conv3x3s1p1_cluster_backward_cuda_naive(torch::Tensor input, torch::Tensor weights,
                                                                   torch::Tensor indice, torch::Tensor grad_output);