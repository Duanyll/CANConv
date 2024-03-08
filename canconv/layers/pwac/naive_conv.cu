#include "naive_conv.cuh"

/// @brief Convolution forward pass on CPU
/// @param input Input tensor (N x Cin x H x W) fp64
/// @param weights Weights tensor (Cout x Cin x Kh x Kw) fp64
/// @return Output tensor (N x Cout x H x W) fp64
torch::Tensor conv3x3s1p1_forward_cpu(torch::Tensor input, torch::Tensor weights) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(0);
  int height = input.size(2);
  int width = input.size(3);
  TORCH_CHECK(in_channels == weights.size(1), "input and weights channels must match");
  TORCH_CHECK(weights.size(2) == 3 && weights.size(3) == 3, "weights must be 3x3");
  TORCH_CHECK(input.device().is_cpu(), "input must be on CPU");
  TORCH_CHECK(weights.device().is_cpu(), "weights must be on CPU");

  torch::Tensor output = torch::zeros({batch_size, out_channels, height, width}, torch::kFloat64);
  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int oh = 0; oh < height; ++oh) {
        for (int ow = 0; ow < width; ++ow) {
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              int ih = oh - 1 + kh;
              int iw = ow - 1 + kw;
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                  output[b][oc][oh][ow] += input[b][ic][ih][iw] * weights[oc][ic][kh][kw];
                }
              }
            }
          }
        }
      }
    }
  }

  return output;
}

std::vector<torch::Tensor> conv3x3s1p1_backward_cpu(torch::Tensor input, torch::Tensor weights,
                                                    torch::Tensor grad_output) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(0);
  int height = input.size(2);
  int width = input.size(3);

  torch::Tensor grad_input = torch::zeros_like(input);
  torch::Tensor grad_weights = torch::zeros_like(weights);

  for (int b = 0; b < batch_size; ++b) {
    for (int ic = 0; ic < in_channels; ++ic) {
      for (int ih = 0; ih < height; ++ih) {
        for (int iw = 0; iw < width; ++iw) {
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              int oh = ih - kh + 1;
              int ow = iw - kw + 1;
              if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                for (int oc = 0; oc < out_channels; ++oc) {
                  grad_input[b][ic][ih][iw] += grad_output[b][oc][oh][ow] * weights[oc][ic][kh][kw];
                  grad_weights[oc][ic][kh][kw] += grad_output[b][oc][oh][ow] * input[b][ic][ih][iw];
                }
              }
            }
          }
        }
      }
    }
  }

  return {grad_input, grad_weights};
}

/// @brief Convolution forward pass on CPU with cluster indices
/// @param input (N x Cin x H x W) fp64
/// @param weights (N x K x Cout x Cin x Kh x Kw) fp64
/// @param indice (N x H x W) int64
/// @return (N x Cout x H x W) fp64
torch::Tensor conv3x3s1p1_cluster_forward_cpu(torch::Tensor input, torch::Tensor weights, torch::Tensor indice) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(2);
  int height = input.size(2);
  int width = input.size(3);
  TORCH_CHECK(in_channels == weights.size(2), "input and weights channels must match");
  TORCH_CHECK(weights.size(4) == 3 && weights.size(5) == 3, "weights must be 3x3");
  TORCH_CHECK(indice.size(0) == batch_size && indice.size(1) == height && indice.size(2) == width,
              "indice size must match input size");
  TORCH_CHECK(input.device().is_cpu(), "input must be on CPU");
  TORCH_CHECK(weights.device().is_cpu(), "weights must be on CPU");
  TORCH_CHECK(indice.device().is_cpu(), "indice must be on CPU");

  torch::Tensor output = torch::zeros({batch_size, out_channels, height, width}, torch::kFloat64);
  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int oh = 0; oh < height; ++oh) {
        for (int ow = 0; ow < width; ++ow) {
          int64_t idx = indice[b][oh][ow].item<int64_t>();
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              int ih = oh - 1 + kh;
              int iw = ow - 1 + kw;
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                  output[b][oc][oh][ow] += input[b][ic][ih][iw] * weights[b][idx][oc][ic][kh][kw];
                }
              }
            }
          }
        }
      }
    }
  }

  return output;
}

/// @brief Convolution backward pass on CPU with cluster indices
/// @param input (N x Cin x H x W) fp64
/// @param weights (N x K x Cout x Cin x Kh x Kw) fp64
/// @param indice (N x H x W) int64
/// @param grad_output (N x Cout x H x W) fp64
/// @return (N x Cin x H x W) fp64, (N x K x Cout x Cin x Kh x Kw) fp64
std::vector<torch::Tensor> conv3x3s1p1_cluster_backward_cpu(torch::Tensor input, torch::Tensor weights,
                                                            torch::Tensor indice, torch::Tensor grad_output) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(2);
  int height = input.size(2);
  int width = input.size(3);

  torch::Tensor grad_input = torch::zeros_like(input);
  torch::Tensor grad_weights = torch::zeros_like(weights);

  for (int b = 0; b < batch_size; ++b) {
    for (int ic = 0; ic < in_channels; ++ic) {
      for (int ih = 0; ih < height; ++ih) {
        for (int iw = 0; iw < width; ++iw) {
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              int oh = ih - kh + 1;
              int ow = iw - kw + 1;
              if (oh >= 0 && oh < height && ow >= 0 && ow < width) {
                int64_t idx = indice[b][oh][ow].item<int64_t>();
                for (int oc = 0; oc < out_channels; ++oc) {
                  grad_input[b][ic][ih][iw] += grad_output[b][oc][oh][ow] * weights[b][idx][oc][ic][kh][kw];
                  grad_weights[b][idx][oc][ic][kh][kw] += grad_output[b][oc][oh][ow] * input[b][ic][ih][iw];
                }
              }
            }
          }
        }
      }
    }
  }

  return {grad_input, grad_weights};
}

template <typename value_t = float>
__global__ void conv3x3s1p1_forward_kernel_naive(value_t *input, value_t *weight, value_t *output, int N, int Cin,
                                                 int H, int W, int Cout) {
  int b = blockIdx.z;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  if (oh < H && ow < W) {
    for (int oc = 0; oc < Cout; oc++) {
      value_t sum = 0;
      for (int ic = 0; ic < Cin; ic++) {
        for (int kh = 0; kh < 3; kh++) {
          for (int kw = 0; kw < 3; kw++) {
            int ih = oh + kh - 1;
            int iw = ow + kw - 1;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
              // output[b][oc][oh][ow] += input[b][ic][ih][iw] * weights[oc][ic][kh][kw];
              sum += input[b * Cin * H * W + ic * H * W + ih * W + iw] *
                     weight[oc * Cin * 3 * 3 + ic * 3 * 3 + kh * 3 + kw];
            }
          }
        }
      }
      output[b * Cout * H * W + oc * H * W + oh * W + ow] = sum;
    }
  }
}

template <typename value_t = float>
__global__ void conv3x3s1p1_backward_kernel_input_naive(value_t *input, value_t *weight, value_t *grad_output,
                                                        value_t *grad_input, int N, int Cin, int H, int W, int Cout) {
  int b = blockIdx.z;
  int ih = blockIdx.y * blockDim.y + threadIdx.y;
  int iw = blockIdx.x * blockDim.x + threadIdx.x;
  if (ih < H && iw < W) {
    for (int ic = 0; ic < Cin; ic++) {
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          int oh = ih - kh + 1;
          int ow = iw - kw + 1;
          if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
            for (int oc = 0; oc < Cout; oc++) {
              // grad_input[b][ic][ih][iw] += grad_output[b][oc][oh][ow] * weights[oc][ic][kh][kw];
              grad_input[b * Cin * H * W + ic * H * W + ih * W + iw] +=
                  grad_output[b * Cout * H * W + oc * H * W + oh * W + ow] *
                  weight[oc * Cin * 3 * 3 + ic * 3 * 3 + kh * 3 + kw];
            }
          }
        }
      }
    }
  }
}

template <typename value_t = float>
__global__ void conv3x3s1p1_backward_kernel_weight_naive(value_t *input, value_t *weight, value_t *grad_output,
                                                         value_t *grad_weights, int N, int Cin, int H, int W,
                                                         int Cout) {
  int oc = threadIdx.x;
  int ic = threadIdx.y;
  int kh = blockIdx.x;
  int kw = blockIdx.y;
  for (int ih = 0; ih < H; ih++) {
    for (int iw = 0; iw < W; iw++) {
      for (int b = 0; b < N; b++) {
        int oh = ih - kh + 1;
        int ow = iw - kw + 1;
        if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
          // grad_weights[oc][ic][kh][kw] += grad_output[b][oc][oh][ow] * input[b][ic][ih][iw];
          grad_weights[oc * Cin * 3 * 3 + ic * 3 * 3 + kh * 3 + kw] +=
              grad_output[b * Cout * H * W + oc * H * W + oh * W + ow] *
              input[b * Cin * H * W + ic * H * W + ih * W + iw];
        }
      }
    }
  }
}

torch::Tensor conv3x3s1p1_forward_cuda_naive(torch::Tensor input, torch::Tensor weights) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(0);
  int height = input.size(2);
  int width = input.size(3);
  TORCH_CHECK(in_channels == weights.size(1), "input and weights channels must match");
  TORCH_CHECK(weights.size(2) == 3 && weights.size(3) == 3, "weights must be 3x3");

  input = input.cuda().contiguous();
  torch::Tensor output = torch::zeros({batch_size, out_channels, height, width},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
  conv3x3s1p1_forward_kernel_naive<<<grid, block>>>(input.data_ptr<float>(), weights.data_ptr<float>(),
                                                    output.data_ptr<float>(), batch_size, in_channels, height, width,
                                                    out_channels);
  return output;
}

std::vector<torch::Tensor> conv3x3s1p1_backward_cuda_naive(torch::Tensor input, torch::Tensor weights,
                                                           torch::Tensor grad_output) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(0);
  int height = input.size(2);
  int width = input.size(3);

  input = input.cuda().contiguous();
  weights = weights.cuda().contiguous();
  grad_output = grad_output.cuda().contiguous();
  torch::Tensor grad_input = torch::zeros_like(input);
  torch::Tensor grad_weights = torch::zeros_like(weights);
  {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
    conv3x3s1p1_backward_kernel_input_naive<<<grid, block>>>(
        input.data_ptr<float>(), weights.data_ptr<float>(), grad_output.data_ptr<float>(), grad_input.data_ptr<float>(),
        batch_size, in_channels, height, width, out_channels);
  }
  {
    dim3 block(in_channels, out_channels);
    dim3 grid(3, 3);
    conv3x3s1p1_backward_kernel_weight_naive<<<grid, block>>>(
        input.data_ptr<float>(), weights.data_ptr<float>(), grad_output.data_ptr<float>(),
        grad_weights.data_ptr<float>(), batch_size, in_channels, height, width, out_channels);
  }
  return {grad_input, grad_weights};
}

template <typename value_t = float>
__global__ void conv3x3s1p1_cluster_forward_kernel_naive(value_t *input, value_t *weight, int64_t *indice,
                                                         value_t *output, int N, int Cin, int H, int W, int Cout,
                                                         int K) {
  int b = blockIdx.z;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx = indice[b * H * W + oh * W + ow];
  if (oh < H && ow < W) {
    for (int oc = 0; oc < Cout; oc++) {
      value_t sum = 0;
      for (int ic = 0; ic < Cin; ic++) {
        for (int kh = 0; kh < 3; kh++) {
          for (int kw = 0; kw < 3; kw++) {
            int ih = oh + kh - 1;
            int iw = ow + kw - 1;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
              // output[b][oc][oh][ow] += input[b][ic][ih][iw] * weights[b][indice[b][oh][ow]][oc][ic][kh][kw];
              sum += input[b * Cin * H * W + ic * H * W + ih * W + iw] *
                     weight[b * K * Cout * Cin * 3 * 3 + idx * Cout * Cin * 3 * 3 + oc * Cin * 3 * 3 + ic * 3 * 3 +
                            kh * 3 + kw];
            }
          }
        }
      }
      output[b * Cout * H * W + oc * H * W + oh * W + ow] = sum;
    }
  }
}

template <typename value_t = float>
__global__ void conv3x3s1p1_cluster_backward_kernel_input_naive(value_t *input, value_t *weight, int64_t *indice,
                                                                value_t *grad_output, value_t *grad_input, int N,
                                                                int Cin, int H, int W, int Cout, int K) {
  int b = blockIdx.z;
  int ih = blockIdx.y * blockDim.y + threadIdx.y;
  int iw = blockIdx.x * blockDim.x + threadIdx.x;
  if (ih < H && iw < W) {
    for (int ic = 0; ic < Cin; ic++) {
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          int oh = ih - kh + 1;
          int ow = iw - kw + 1;
          if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
            int64_t idx = indice[b * H * W + oh * W + ow];
            for (int oc = 0; oc < Cout; oc++) {
              // grad_input[b][ic][ih][iw] += grad_output[b][oc][oh][ow] *
              // weights[b][indice[b][oh][ow]][oc][ic][kh][kw];
              grad_input[b * Cin * H * W + ic * H * W + ih * W + iw] +=
                  grad_output[b * Cout * H * W + oc * H * W + oh * W + ow] *
                  weight[b * K * Cout * Cin * 3 * 3 + idx * Cout * Cin * 3 * 3 + oc * Cin * 3 * 3 + ic * 3 * 3 +
                         kh * 3 + kw];
            }
          }
        }
      }
    }
  }
}

template <typename value_t = float>
__global__ void conv3x3s1p1_backward_kernel_weight_naive(value_t *input, value_t *weight, int64_t *indice,
                                                         value_t *grad_output, value_t *grad_weights, int N, int Cin,
                                                         int H, int W, int Cout, int K) {
  int oc = threadIdx.x;
  int ic = threadIdx.y;
  int kh = blockIdx.x;
  int kw = blockIdx.y;
  for (int ih = 0; ih < H; ih++) {
    for (int iw = 0; iw < W; iw++) {
      for (int b = 0; b < N; b++) {
        int oh = ih - kh + 1;
        int ow = iw - kw + 1;
        if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
          int64_t idx = indice[b * H * W + oh * W + ow];
          // grad_weights[b][indice[b][oh][ow]][oc][ic][kh][kw] += grad_output[b][oc][oh][ow] * input[b][ic][ih][iw];
          grad_weights[b * K * Cout * Cin * 3 * 3 + idx * Cout * Cin * 3 * 3 + oc * Cin * 3 * 3 + ic * 3 * 3 + kh * 3 +
                       kw] += grad_output[b * Cout * H * W + oc * H * W + oh * W + ow] *
                              input[b * Cin * H * W + ic * H * W + ih * W + iw];
        }
      }
    }
  }
}

torch::Tensor conv3x3s1p1_cluster_forward_cuda_naive(torch::Tensor input, torch::Tensor weights, torch::Tensor indice) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(2);
  int height = input.size(2);
  int width = input.size(3);
  int cluster_num = weights.size(1);
  TORCH_CHECK(in_channels == weights.size(2), "input and weights channels must match");
  TORCH_CHECK(weights.size(4) == 3 && weights.size(5) == 3, "weights must be 3x3");
  TORCH_CHECK(indice.size(0) == batch_size && indice.size(1) == height && indice.size(2) == width,
              "indice size must match input size");
  input = input.cuda().contiguous();
  weights = weights.cuda().contiguous();
  indice = indice.cuda().contiguous();

  torch::Tensor output = torch::zeros({batch_size, out_channels, height, width},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
  conv3x3s1p1_cluster_forward_kernel_naive<<<grid, block>>>(
      input.data_ptr<float>(), weights.data_ptr<float>(), indice.data_ptr<int64_t>(), output.data_ptr<float>(),
      batch_size, in_channels, height, width, out_channels, cluster_num);

  return output;
}

std::vector<torch::Tensor> conv3x3s1p1_cluster_backward_cuda_naive(torch::Tensor input, torch::Tensor weights,
                                                                   torch::Tensor indice, torch::Tensor grad_output) {
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int out_channels = weights.size(2);
  int height = input.size(2);
  int width = input.size(3);
  int cluster_num = weights.size(1);

  torch::Tensor grad_input = torch::zeros_like(input);
  torch::Tensor grad_weights = torch::zeros_like(weights);

  input = input.cuda().contiguous();
  weights = weights.cuda().contiguous();
  indice = indice.cuda().contiguous();
  grad_output = grad_output.cuda().contiguous();

  {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
    conv3x3s1p1_cluster_backward_kernel_input_naive<<<grid, block>>>(
        input.data_ptr<float>(), weights.data_ptr<float>(), indice.data_ptr<int64_t>(), grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(), batch_size, in_channels, height, width, out_channels, cluster_num);
  }
  {
    dim3 block(in_channels, out_channels);
    dim3 grid(3, 3);
    conv3x3s1p1_backward_kernel_weight_naive<<<grid, block>>>(
        input.data_ptr<float>(), weights.data_ptr<float>(), indice.data_ptr<int64_t>(), grad_output.data_ptr<float>(),
        grad_weights.data_ptr<float>(), batch_size, in_channels, height, width, out_channels, cluster_num);
  }

  return {grad_input, grad_weights};
}