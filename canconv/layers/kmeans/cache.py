import torch
from einops import rearrange, repeat, reduce
from .cluster_center import get_cluster_centers_scatter
from .kmeans import kmeans

store = {}
"""
module_id -> (cache_size, sample_num)
Stored on the same device as samples.
"""
cache_mode = "disable"  # "disable", "init", "update", "ready",
"""
Cache is used to reduce kmeans computation by reusing previous results between a few epochs.
Disable: No cache is read or written to, kmeans is run from scratch every time.
Init: Run kmeans with random cluster center initialization and cache the result. Used for the first epoch.
Update: Run kmeans with previous cluster center initialization and cache the result. Used for the following epochs.
Ready: Use cached result directly. Used for the following epochs.
"""
cache_size = 0
"""
Defines size to allocate for cache. Usually set to the number of samples in the dataset.
"""


def reset_cache(size):
    global store
    store = {}

    global cache_size
    cache_size = size


def kmeans_batched(samples: torch.Tensor, cluster_num: int, cache_indice=None, module_id=None) -> torch.Tensor:
    """
    Run kmeans on batched samples. Will use cache if cache_indice and module_id is not None. Result is on the same device as samples.
    Args:
        samples: (batch_size, sample_num, feature_dim)
        cluster_num: int
        cache_indice: int tensor (batch_size) or None
    Returns:
        cluster_idx: (batch_size, sample_num)
    """
    dev = samples.device
    batch_size, sample_num, feature_dim = samples.shape
    if cache_mode != "disable" and cache_indice is not None and module_id is not None:
        if cache_mode == "init":
            if module_id not in store:
                store[module_id] = torch.zeros(
                    (cache_size, sample_num), dtype=torch.long, device=dev)
            for i in range(batch_size):
                store[module_id][cache_indice[i], :] = kmeans(
                    samples[i, :, :], cluster_num)
            return store[module_id][cache_indice]
        elif cache_mode == "update":
            last_result = store[module_id][cache_indice].to(dev)
            last_centers = get_cluster_centers_scatter(
                samples, last_result, cluster_num)
            cluster_indice = torch.zeros(
                batch_size, sample_num, dtype=torch.long, device=dev)
            for i in range(batch_size):
                cluster_indice[i, :] = kmeans(
                    samples[i, :, :], cluster_num, last_centers[i, :, :])
            store[module_id][cache_indice] = cluster_indice.detach()
            return cluster_indice.to(dev)
        else:
            return store[module_id][cache_indice]
    else:
        cluster_indice = torch.zeros(
            batch_size, sample_num, dtype=torch.long, device=dev)
        for i in range(batch_size):
            cluster_indice[i, :] = kmeans(samples[i, :, :], cluster_num)
        return cluster_indice


class KMeansCacheScheduler:
    def __init__(self, policy):
        assert isinstance(policy, int) or isinstance(
            policy, list), "policy must be int or list"
        self.policy = policy
        self.current_epoch = 0

    def step(self):
        """
        Example: policy = [(100, 10), (300, 20), 50]
        1. At epoch 1, cache_mode = "init"
        2. For epoch 2 to 100, Every 10 epochs, cache_mode = "update", else cache_mode = "ready"
        3. For epoch 101 to 300, Every 20 epochs, cache_mode = "update", else cache_mode = "ready"
        4. For epoch > 300, Every 50 epochs, cache_mode = "update", else cache_mode = "ready"

        Example: policy = 10
        1. At epoch 1, cache_mode = "init"
        2. Every 10 epochs, cache_mode = "update", else cache_mode = "ready"
        """
        global cache_mode
        self.current_epoch += 1
        if self.current_epoch == 1:
            cache_mode = "init"
        elif isinstance(self.policy, int):
            if self.current_epoch % self.policy == 0:
                cache_mode = "update"
            else:
                cache_mode = "ready"
        else:
            for i in range(len(self.policy)):
                if isinstance(self.policy[i], int):
                    if self.current_epoch % self.policy[i] == 0:
                        cache_mode = "update"
                        break
                    else:
                        cache_mode = "ready"
                        break
                else:
                    if self.current_epoch > self.policy[i][0]:
                        pass
                    elif self.current_epoch % self.policy[i][1] == 0:
                        cache_mode = "update"
                        break
                    else:
                        cache_mode = "ready"
                        break
