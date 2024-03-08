import torch
import torch.nn as nn

from .cache import kmeans_batched

new_module_id = 0


class KMeans(nn.Module):
    def __init__(self, cluster_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        global new_module_id
        self.module_id = new_module_id
        new_module_id += 1

        self.cluster_num = cluster_num

    def forward(self, x, cache_indice=None, cluster_num=None):
        if not self.training:
            cache_indice = None
        if cluster_num is None:
            cluster_num = self.cluster_num
        return kmeans_batched(x, cluster_num, cache_indice, self.module_id)
