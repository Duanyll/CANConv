import importlib
import logging
import torch
from torch.autograd import Function
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

logger.info("Loading custom op for conv_by_cluster...")
try:
    c_lib = importlib.import_module("..pwacnative", __name__)
    c_lib.set_cublas_handle(torch.cuda.current_blas_handle())
except Exception as e:
    logger.error("Failed to load custom op for conv_by_cluster!")
    logger.exception(e)
    print("""
Cannot load custom op for pwac. Please build the CMakelists.txt in the current directory, 
and make sure to copy the generated pwacnative.so file to the current directory. 
Also check if cuda and cublas are installed correctly.
          """)
    exit(1)


class Conv3x3S1P1(Function):
    @staticmethod
    def forward(ctx, input, weight):
        if input.device.type != weight.device.type:
            raise RuntimeError("Input and weight must be on the same device!")
        ctx.save_for_backward(input, weight)
        if input.device.type == "cpu":
            output = c_lib.conv3x3s1p1_forward_cpu(input, weight)
        elif input.device.type == "cuda":
            output = c_lib.conv3x3s1p1_forward_cuda_naive(input, weight)
        else:
            raise RuntimeError("Device type not supported!")
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if input.device.type == "cpu":
            grad_input, grad_weight = c_lib.conv3x3s1p1_backward_cpu(input, weight, grad_output)
        elif input.device.type == "cuda":
            grad_input, grad_weight = c_lib.conv3x3s1p1_backward_cuda_naive(input, weight, grad_output)
        else:
            raise RuntimeError("Device type not supported!")
        return grad_input, grad_weight
    
class Conv3x3S1P1Cluster(Function):
    @staticmethod
    def forward(ctx, input, weight, indice):
        if not (input.device.type == weight.device.type == indice.device.type):
            raise RuntimeError("Input, weight and indice must be on the same device!")
        ctx.save_for_backward(input, weight, indice)
        if input.device.type == "cpu":
            output = c_lib.conv3x3s1p1_cluster_forward_cpu(input, weight, indice)
        elif input.device.type == "cuda":
            output = c_lib.conv3x3s1p1_cluster_forward_cuda_naive(input, weight, indice)
        else:
            raise RuntimeError("Device type not supported!")
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, indice = ctx.saved_tensors
        if input.device.type == "cpu":
            grad_input, grad_weight = c_lib.conv3x3s1p1_cluster_backward_cpu(input, weight, indice, grad_output)
        elif input.device.type == "cuda":
            grad_input, grad_weight = c_lib.conv3x3s1p1_cluster_backward_cuda_naive(input, weight, indice, grad_output)
        else:
            raise RuntimeError("Device type not supported!")
        return grad_input, grad_weight, None
    
def conv3x3s1p1_naive(input, weight) -> torch.Tensor:
    return Conv3x3S1P1.apply(input, weight) # type: ignore

def conv3x3s1p1_cluster_naive(input, weight, indice) -> torch.Tensor:
    return Conv3x3S1P1Cluster.apply(input, weight, indice) # type: ignore

def conv_by_im2col_cluster_nested(input, weight, indice) -> torch.Tensor:
    return c_lib.conv_by_im2col_cluster_nested(input, weight, indice)

def filter_indice(indice: torch.Tensor, cluster_num: int, threshold: float) -> torch.Tensor:
    return c_lib.filter_indice(indice, cluster_num, threshold)

def dispatch_indice(indice: torch.Tensor, cluster_num: int) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return c_lib.dispatch_indice(indice, cluster_num)

def permute(input: torch.Tensor, indice_perm: torch.Tensor, padded_patch_num: int) -> torch.Tensor:
    return c_lib.permute(input, indice_perm, padded_patch_num)

def inverse_permute(input: torch.Tensor, indice_perm: torch.Tensor) -> torch.Tensor:
    return c_lib.inverse_permute(input, indice_perm)

def batched_matmul_conv(input_permuted, weight, permuted_offset, cluster_perm, batch_height, bias=None) -> torch.Tensor:
    if bias is None:
        return c_lib.batched_matmul_conv(input_permuted, weight, permuted_offset, cluster_perm, batch_height)
    else:
        return c_lib.batched_matmul_conv_bias(input_permuted, weight, permuted_offset, cluster_perm, batch_height, bias)
    
def conv3x3s1p1_cluster_im2col(input, weight, indice, bias=None) -> torch.Tensor:
    b = input.shape[0]
    ic = input.shape[1]
    h = input.shape[2]
    w = input.shape[3]
    k = weight.shape[1]
    oc = weight.shape[2]
    
    patches = F.unfold(input, kernel_size=3, padding=1)
    patches = rearrange(patches, "b f hw -> (b hw) f")
    weight = rearrange(weight, "b k oc ic kh kw -> (b k) (ic kh kw) oc")
    indice = indice + torch.arange(b, device=indice.device).view(-1, 1, 1) * k
    indice = rearrange(indice, "b h w -> (b h w)")
    if bias is not None:
        bias = rearrange(bias, "b k oc -> (b k) oc")
    
    indice_perm, padded_patch_num, cluster_size_sorted, permuted_offset, cluster_perm, batch_height = dispatch_indice(indice, b * k)
    input_permuted = permute(patches, indice_perm, padded_patch_num)
    output_permuted = batched_matmul_conv(input_permuted, weight, permuted_offset, cluster_perm, batch_height, bias)
    output = inverse_permute(output_permuted, indice_perm)
    
    output = rearrange(output, "(b h w) oc -> b oc h w", b=b, h=h, w=w)
    return output
    
def batched_matmul_qk(q, k, permuted_offset, batch_height, attn_offset, attn_size) -> torch.Tensor:
    return c_lib.batched_matmul_qk(q, k, permuted_offset, batch_height, attn_offset, attn_size)

def batched_matmul_qkv(attn_map, v, permuted_offset, batch_height, attn_offset, padded_patch_num) -> torch.Tensor:
    return c_lib.batched_matmul_qkv(attn_map, v, permuted_offset, batch_height, attn_offset, padded_patch_num)

def dispatch_attn_offset(batch_height, cluster_num) -> torch.Tensor:
    return c_lib.dispatch_attn_offset(batch_height, cluster_num)

def batched_softmax(x, batch_height, attn_offset) -> torch.Tensor:
    return c_lib.batched_softmax(x, batch_height, attn_offset)