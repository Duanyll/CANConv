import torch
from einops import rearrange, repeat, reduce


def get_cluster_centers_mask(samples: torch.Tensor, cluster_indice: torch.Tensor, cluster_num: int) -> torch.Tensor:
    """
    Args:
        samples: (batch_size, sample_num, feature_dim)
        cluster_indice: (batch_size, sample_num)
        cluster_num: int
    Returns:
        cluster_centers: (batch_size, cluster_num, feature_dim)
    """
    dev = samples.device
    batch_size = samples.shape[0]
    feature_dim = samples.shape[2]
    cluster_centers = torch.zeros(
        batch_size, cluster_num, feature_dim, device=dev, dtype=samples.dtype)
    for i in range(cluster_num):
        cluster_centers[:, i, :] = torch.mean(
            samples[cluster_indice == i, :], dim=0)
    return cluster_centers


def get_cluster_centers_scatter(samples: torch.Tensor, cluster_indice: torch.Tensor, cluster_num: int) -> torch.Tensor:
    """
    Args:
        samples: (batch_size, sample_num, feature_dim)
        cluster_indice: (batch_size, sample_num)
        cluster_num: int
    Returns:
        cluster_centers: (batch_size, cluster_num, feature_dim)
    """
    dev = samples.device
    batch_size = samples.shape[0]
    sample_num = samples.shape[1]
    feature_dim = samples.shape[2]
    # print(cluster_indice.min(), cluster_indice.max())
    cluster_centers = torch.zeros(batch_size, cluster_num, feature_dim, device=dev).scatter_add_(
        dim=1, index=repeat(cluster_indice, 'b p -> b p s', s=feature_dim), src=samples)
    cluster_size = torch.zeros(batch_size, cluster_num, device=dev).scatter_add_(
        dim=1, index=cluster_indice, src=torch.ones(batch_size, sample_num, device=dev)).unsqueeze_(dim=2)
    cluster_size[cluster_size < 1] = 1
    cluster_centers /= cluster_size
    return cluster_centers
