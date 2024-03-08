import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from einops import rearrange
from tqdm import tqdm

from .h5pan import H5PanDataset
from canconv.layers.kmeans import kmeans


def generate_cluster_indice_cache(dataset: H5PanDataset, cluster_num: int, cache_path: str):
    """
    Args:
        dataset: H5PanDataset
        cluster_num: int
        cache_path: str
    """
    img_count = len(dataset)
    img_sample = dataset[0]
    height = img_sample['pan'].shape[1]
    width = img_sample['pan'].shape[2]
    res = torch.zeros((img_count, height * width), dtype=torch.int8)
    for i in tqdm(range(img_count)):
        img = dataset[i]
        patches = F.unfold(img['pan'].unsqueeze(0), kernel_size=3, padding=1)
        patches = rearrange(
            patches, '1 (cin area) (h w) -> (h w) (cin area)', area=9, h=height, w=width)
        res[i] = kmeans(patches, cluster_num).type_as(res)
    torch.save(res, cache_path)


class H5PanDatasetWithCluster(H5PanDataset):
    def __init__(self, h5_path: str, cluster_num: int = 32, cluster_cache_path = None):
        """
        Args:
            h5_path: str
            cluster_num: int
            cluster_cache_path: str
        """
        super().__init__(h5_path)
        if cluster_cache_path is None:
            cluster_cache_path = os.path.join(os.path.dirname(
                h5_path), f'{os.path.splitext(h5_path)[0]}_cluster{cluster_num}.pt')
        if not os.path.exists(cluster_cache_path):
            print(f'Generating cluster indice cache: {cluster_cache_path} ...')
            generate_cluster_indice_cache(self, cluster_num, cluster_cache_path)
        print(f'Loading cluster indice cache: {cluster_cache_path} ...')
        self.cluster_cache = torch.load(cluster_cache_path).long()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        img = super().__getitem__(idx)
        if hasattr(self, "cluster_cache"):
            img['cluster'] = self.cluster_cache[idx]
        return img

if __name__ == '__main__':
    dataset = H5PanDataset('/mnt/d/datasets/wv3/train_wv3.h5')
    generate_cluster_indice_cache(dataset, 32, 'cluster_indice_wv3.pt')