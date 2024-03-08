#pragma once

#include "utils.h"

torch::Tensor conv_by_im2col_cluster_nested(torch::Tensor input, torch::Tensor weight, torch::Tensor indice);