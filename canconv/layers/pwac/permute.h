#include "utils.h"

torch::Tensor permute(torch::Tensor input, torch::Tensor indice_perm, int padded_patch_num);
torch::Tensor inverse_permute(torch::Tensor input, torch::Tensor indice_perm);