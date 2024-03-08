import logging
import torch
import numpy as np
from torch.profiler import record_function
from einops import rearrange, repeat, reduce

logger = logging.getLogger(__name__)
logger.info("Begin to load kmeans operator...")
try:
    from .libKMCUDA import kmeans_cuda  # type: ignore
except ImportError as e:
    logger.error("Fail to load kmeans operator from local path.")
    logger.exception(e)
    print("Please use libKMCUDA built from https://github.com/duanyll/kmcuda. The built libKMCUDA.so file should be placed in the same directory as this file. Do not use the official libKMCUDA from pip.")
    raise e
logger.info("Finish loading kmeans operator.")

seed = 42


def kmeans(samples: torch.Tensor, cluster_num: int, cached_center=None) -> torch.Tensor:
    """
    Run kmeans on samples. Result is on the same device as samples. If cached_center is not None, it will be used as the initial cluster center.
    Args:
        samples: (sample_num, feature_dim)
        cluster_num: int
        cached_center: (cluster_num, feature_dim)
    Returns:
        cluster_idx: (sample_num)
    """
    if cluster_num <= 1:
        return torch.zeros(samples.shape[0])
    if cluster_num > samples.shape[0]:
        logger.warning(
            f"cluster_num ({cluster_num}) > sample_num ({samples.shape[0]}).")
        cluster_num = samples.shape[0]
    with record_function("kmeans"):
        if cached_center is None:
            idx, _ = kmeans_cuda(samples, cluster_num, seed=seed)
        else:
            idx, _ = kmeans_cuda(samples, cluster_num,
                                 cached_center, seed=seed)
    return idx.long()
