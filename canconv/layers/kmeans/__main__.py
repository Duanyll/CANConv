import torch
from .cluster_center import get_cluster_centers_mask, get_cluster_centers_scatter


def _compare_center_performance():
    # Compare performance between get_cluster_centers_mask and get_cluster_centers_scatter
    import time
    batch_size = 32
    sample_num = 4096
    feature_dim = 128
    cluster_num = 32
    samples = torch.rand(batch_size, sample_num, feature_dim).cuda()
    cluster_indice = torch.randint(
        0, cluster_num, (batch_size, sample_num)).cuda()
    start = time.time()
    for i in range(100):
        get_cluster_centers_mask(samples, cluster_indice, cluster_num)
    end = time.time()
    print("get_cluster_centers_mask: ", end - start)
    # get_cluster_centers_mask:  0.809556245803833
    start = time.time()
    for i in range(100):
        get_cluster_centers_scatter(samples, cluster_indice, cluster_num)
    end = time.time()
    print("get_cluster_centers_scatter: ", end - start)
    # get_cluster_centers_scatter:  0.010624885559082031


if __name__ == "__main__":
    print("Compare performance between get_cluster_centers_mask and get_cluster_centers_scatter, please check the output:")
    _compare_center_performance()
