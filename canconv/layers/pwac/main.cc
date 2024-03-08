#include "dispatch_indice.h"
#include "im2col.h"
#include "matmul_conv.h"
#include "matmul_attn.h"
#include "naive_conv.cuh"
#include "permute.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv3x3s1p1_forward_cpu", &conv3x3s1p1_forward_cpu, "conv3x3s1p1 forward on CPU, fp64");
  m.def("conv3x3s1p1_backward_cpu", &conv3x3s1p1_backward_cpu, "conv3x3s1p1 backward on CPU, fp64");
  m.def("conv3x3s1p1_cluster_forward_cpu", &conv3x3s1p1_cluster_forward_cpu,
        "conv3x3s1p1 cluster forward on CPU, fp64");
  m.def("conv3x3s1p1_cluster_backward_cpu", &conv3x3s1p1_cluster_backward_cpu,
        "conv3x3s1p1 cluster backward on CPU, fp64");
  m.def("conv3x3s1p1_forward_cuda_naive", &conv3x3s1p1_forward_cuda_naive, "conv3x3s1p1 forward naive on GPU, fp32");
  m.def("conv3x3s1p1_backward_cuda_naive", &conv3x3s1p1_backward_cuda_naive, "conv3x3s1p1 backward naive on GPU, fp32");
  m.def("conv3x3s1p1_cluster_forward_cuda_naive", &conv3x3s1p1_cluster_forward_cuda_naive,
        "conv3x3s1p1 cluster forward naive on GPU, fp32");
  m.def("conv3x3s1p1_cluster_backward_cuda_naive", &conv3x3s1p1_cluster_backward_cuda_naive,
        "conv3x3s1p1 cluster backward naive on GPU, fp32");

  m.def("set_cublas_handle", &set_cublas_handle, "set cublas handle");
  m.def("conv_by_im2col_cluster_nested", &conv_by_im2col_cluster_nested, "conv by im2col cluster naive");
  m.def("filter_indice", &filter_indice, "filter indice");
  m.def("dispatch_indice", &dispatch_indice, "dispatch indice");
  m.def("dispatch_attn_offset", &dispatch_attn_offset, "dispatch attn matrix");
  m.def("permute", &permute, "permute");
  m.def("inverse_permute", &inverse_permute, "inverse permute");
  m.def("batched_matmul_conv", &batched_matmul_conv, "batched matmul conv");
  m.def("batched_matmul_conv_bias", &batched_matmul_conv_bias, "batched matmul conv bias");
  m.def("batched_matmul_qk", &batched_matmul_qk, "batched matmul qk");
  m.def("batched_matmul_qkv", &batched_matmul_qkv, "batched matmul qkv");
  m.def("batched_softmax", &batched_softmax, "batched softmax");
}