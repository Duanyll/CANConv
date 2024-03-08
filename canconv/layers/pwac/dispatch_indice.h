#include "utils.h"

torch::Tensor filter_indice(torch::Tensor indice, int cluster_num, float threshold_ratio);

std::tuple<torch::Tensor, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatch_indice(torch::Tensor indice, int cluster_num);

std::tuple<int, torch::Tensor> dispatch_attn_offset(torch::Tensor batch_height, int cluster_num);