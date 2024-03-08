# type: ignore

from typing import Any
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck, Function # type: ignore
from .custome_op import conv3x3s1p1_naive, conv3x3s1p1_cluster_naive, conv3x3s1p1_cluster_im2col
from .torch_op import conv3x3s1p1_cluster_masked, conv3x3s1p1_cluster_gather
from torch.profiler import profile, ProfilerActivity, record_function
from einops import rearrange
import os
from tqdm import tqdm
from canconv.util.seed import seed_everything

seed_everything(42)
print(os.getpid())

# 1. Test Conv3x3S1P1 CPU

input = torch.randn(1, 4, 4, 4, requires_grad=True, dtype=torch.float64)
weights = torch.randn(4, 4, 3, 3, requires_grad=True, dtype=torch.float64)

output_my = conv3x3s1p1_naive(input, weights)
output_real = F.conv2d(input, weights, padding=1)

assert torch.allclose(output_my, output_real, atol=1e-4)
print("Conv3x3S1P1 CPU forward pass test passed!")
# The following test is really slow, so I comment it out.
# assert gradcheck(Conv3x3S1P1.apply, (input, weights), eps=1e-6, atol=1e-4)
# print("Conv3x3S1P1 CPU backward pass test passed!")

# 2. Test Conv3x3S1P1Cluster CPU

input = torch.randn(1, 4, 4, 4, requires_grad=True, dtype=torch.float64)
weight = torch.randn(1, 4, 4, 4, 3, 3, requires_grad=True, dtype=torch.float64)
indice = torch.randint(0, 4, (1, 4, 4), dtype=torch.long)

output_custom = conv3x3s1p1_cluster_naive(input, weight, indice)
assert output_custom.shape == (1, 4, 4, 4)
print("Conv3x3S1P1Cluster CPU forward pass test passed!")
# print(gradcheck(Conv3x3S1P1Cluster.apply, (input, weight, indice), eps=1e-6, atol=1e-4))
# print("Conv3x3S1P1Cluster CPU backward pass test passed!")

# 3. Test masked op and gather op

output_masked = conv3x3s1p1_cluster_masked(input, weight, indice)
output_gather = conv3x3s1p1_cluster_gather(input, weight, indice)

assert torch.allclose(output_custom, output_masked, atol=1e-4)
assert torch.allclose(output_custom, output_gather, atol=1e-4)
print("Masked op and gather op test passed!")

# 4. Test Conv3x3S1P1 GPU

input = torch.randn(1, 4, 4, 4, requires_grad=True, dtype=torch.float32)
weights = torch.randn(4, 4, 3, 3, requires_grad=True, dtype=torch.float32)

input_cuda = input.cuda().detach().requires_grad_(True)
weights_cuda = weights.cuda().detach().requires_grad_(True)

output_cpu = conv3x3s1p1_naive(input, weights)
output_cuda = conv3x3s1p1_naive(input_cuda, weights_cuda)

assert torch.allclose(output_cpu.to(torch.float32),
                      output_cuda.cpu(), atol=1e-4)
print("Conv3x3S1P1 GPU forward pass test passed!")

torch.sum(output_cpu).backward()
torch.sum(output_cuda).backward()

assert torch.allclose(input.grad.to(torch.float32),
                      input_cuda.grad.cpu(), atol=1e-4)
assert torch.allclose(weights.grad.to(torch.float32),
                      weights_cuda.grad.cpu(), atol=1e-4)
print("Conv3x3S1P1 GPU backward pass test passed!")

# 5. Test Conv3x3S1P1Cluster GPU

batch = 2
cin = 1
cout = 1
h = 4
w = 4
k = 4

input = torch.randn(batch, cin, h, w, requires_grad=True, dtype=torch.float32)
weight = torch.randn(batch, k, cout, cin, 3, 3, requires_grad=True, dtype=torch.float32)
indice = torch.randint(0, k, (batch, h, w), dtype=torch.long)

input_cuda = input.cuda().detach().requires_grad_(True)
weight_cuda = weight.cuda().detach().requires_grad_(True)
indice_cuda = indice.cuda().detach()

output_cpu = conv3x3s1p1_cluster_naive(input, weight, indice)
output_cuda = conv3x3s1p1_cluster_naive(input_cuda, weight_cuda, indice_cuda)

assert torch.allclose(output_cpu.to(torch.float32),
                      output_cuda.cpu(), atol=1e-4)
print("Conv3x3S1P1Cluster GPU forward pass test passed!")

torch.sum(output_cpu).backward()
torch.sum(output_cuda).backward()

assert torch.allclose(input.grad.to(torch.float32),
                      input_cuda.grad.cpu(), atol=1e-4)
assert torch.allclose(weight.grad.to(torch.float32),
                      weight_cuda.grad.cpu(), atol=1e-4)
print("Conv3x3S1P1Cluster GPU backward pass test passed!")

# 6. Test conv3x3s1p1_cluster_im2col

input_cuda.grad = None
output_im2col = conv3x3s1p1_cluster_im2col(input_cuda, weight_cuda, indice_cuda)
assert torch.allclose(output_cuda, output_im2col, atol=1e-4)
print("Conv3x3S1P1Cluster GPU im2col forward test passed!")
torch.sum(output_im2col).backward()
assert torch.allclose(input.grad.to(torch.float32),
                      input_cuda.grad.cpu(), atol=1e-4)
print("Conv3x3S1P1Cluster GPU im2col backward test passed!")

# 7. Make sure Conv3x3S1P1Cluster GPU works on large (standard) input

batch = 32
cin = 32
cout = 32
h = 64
w = 64
k = 32
input_cuda = torch.randn(batch, cin, h, w, requires_grad=True,
                         dtype=torch.float32, device="cuda")
weight = torch.randn(batch, k, cout, cin, 3, 3, requires_grad=True,
                     dtype=torch.float32, device="cuda")
indice = torch.randint(0, k, (batch, h, w), dtype=torch.long, device="cuda")

output_cuda = conv3x3s1p1_cluster_naive(input_cuda, weight, indice)
torch.sum(output_cuda).backward()

assert output_cuda.shape == (batch, cout, h, w)
print("Conv3x3S1P1Cluster GPU large test passed!")

output_im2col = conv3x3s1p1_cluster_im2col(input_cuda, weight, indice)
assert torch.allclose(output_cuda, output_im2col, atol=1e-4)
print("Conv3x3S1P1Cluster GPU im2col large test passed!")

# 8. Test Bias

batch = 2
cin = 1
cout = 1
h = 4
w = 4
k = 4

input1 = torch.randn(batch, cin, h, w, requires_grad=True,
                         dtype=torch.float32, device="cuda")
weight1 = torch.randn(batch, k, cout, cin, 3, 3, requires_grad=True,
                     dtype=torch.float32, device="cuda")
indice1 = torch.randint(0, k, (batch, h, w), dtype=torch.long, device="cuda")
bias1 = torch.randn(batch, k, cout, requires_grad=True,
                        dtype=torch.float32, device="cuda")
input2 = input1.clone().detach().requires_grad_(True)
weight2 = weight1.clone().detach().requires_grad_(True)
indice2 = indice1.clone().detach()
bias2 = bias1.clone().detach().requires_grad_(True)

output1 = conv3x3s1p1_cluster_masked(input1, weight1, indice1, bias1)
output2 = conv3x3s1p1_cluster_im2col(input2, weight2, indice2, bias2)

assert torch.allclose(output1, output2, atol=1e-4)
print("Conv3x3S1P1Cluster GPU bias test passed!")

torch.sum(output1).backward()
torch.sum(output2).backward()

assert torch.allclose(input1.grad, input2.grad, atol=1e-4)
assert torch.allclose(weight1.grad, weight2.grad, atol=1e-4)
assert torch.allclose(bias1.grad, bias2.grad, atol=1e-4)
print("Conv3x3S1P1Cluster GPU bias backward test passed!")

from .custome_op import batched_matmul_qk, batched_matmul_qkv
q = torch.randn(8, 3, requires_grad=True, dtype=torch.float32, device="cuda")
k = torch.randn(8, 3, requires_grad=True, dtype=torch.float32, device="cuda")
v = torch.randn(8, 5, requires_grad=True, dtype=torch.float32, device="cuda")
permuted_offset = torch.tensor([0, 5], dtype=torch.long, device="cuda")
batch_height = torch.tensor([5], dtype=torch.long, device="cuda")
res = batched_matmul_qkv(torch.softmax(batched_matmul_qk(q, k, permuted_offset, batch_height), dim=-1), v, permuted_offset, batch_height)
res.sum().backward()

print(q.grad)

# 9. Benchmark different implementations

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(
#         wait=2,
#         warmup=2,
#         active=6,
#         repeat=1),
#     # record_shapes=True,
#     # with_stack=True
# ) as prof:
#     for step in tqdm(range(10)):
#         with record_function('conv3x3s1p1_cluster_naive'):
#             conv3x3s1p1_cluster_naive(input_cuda, weight, indice)
#             torch.cuda.synchronize()
#         with record_function('conv3x3s1p1_cluster_masked'):
#             conv3x3s1p1_cluster_masked(input_cuda, weight, indice)
#             torch.cuda.synchronize()
#         with record_function('conv3x3s1p1_cluster_gather'):
#             conv3x3s1p1_cluster_gather(input_cuda, weight, indice)
#             torch.cuda.synchronize()
#         with record_function('conv3x3s1p1_cluster_im2col'):
#             conv3x3s1p1_cluster_im2col(input_cuda, weight, indice)
#             torch.cuda.synchronize()
#         prof.step()
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
