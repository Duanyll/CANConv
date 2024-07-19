import torch
import numpy as np
import h5py
import logging
from torch.utils.data import Dataset
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def load_h5_key_to_torch(file: h5py.File, key: str, scale: float) -> torch.Tensor:
    data = file[key][...] # type: ignore
    data = np.array(data, dtype=np.float32) / scale
    return torch.from_numpy(data)

class H5PanDataset(Dataset):
    """
    加载符合 README.md 中规范的 h5py 数据集
    """
    def __init__(self, file_path, scale=0., device="cpu", pool=False) -> None:
        super().__init__()
        self.file_path = file_path
        if scale == 0.:
            if "wv3" in file_path or "qb" in file_path or "wv2" in file_path:
                scale = 2047.
            elif "gf2" in file_path:
                scale = 1023.
            else:
                scale = 1.
                logger.warning(f"Cannot detect scale of dataset {file_path}, set to 1.0. Please check manually.")
        with h5py.File(file_path) as file:
            if "/gt" in file:
                self.has_gt = True
                self.gt = load_h5_key_to_torch(file, "/gt", scale).to(device)
            else:
                self.has_gt = False
            
            self.ms = load_h5_key_to_torch(file, "/ms", scale).to(device)
            self.lms = load_h5_key_to_torch(file, "/lms", scale).to(device)
            self.pan = load_h5_key_to_torch(file, "/pan", scale).to(device)
            
        if pool:
            self.ms = F.avg_pool2d(self.ms, 2)
            self.lms = F.avg_pool2d(self.lms, 2)
            self.pan = F.avg_pool2d(self.pan, 2)

        self.length = self.pan.shape[0]
        self.spectral_num = self.ms.shape[1]
        self.index = torch.arange(self.length, device=device)
        self.scale = scale


    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        item = {
            'ms': self.ms[index, :, :, :],
            'lms': self.lms[index, :, :, :],
            'pan': self.pan[index, :, :, :],
            'index': self.index[index]
        }
        if (self.has_gt):
            item['gt'] = self.gt[index, :, :, :]
        return item
    
    def validate(self):
        print("Validating dataset...")
        print(f"File: {self.file_path}")
        print(f"Dataset length: {self.length}")
        ms_count = torch.count_nonzero((self.ms < 0) | (self.ms > self.scale))
        ms_images = torch.count_nonzero(torch.count_nonzero((self.ms < 0) | (self.ms > self.scale), dim=(1, 2, 3)))
        lms_count = torch.count_nonzero((self.lms < 0) | (self.lms > self.scale))
        lms_images = torch.count_nonzero(torch.count_nonzero((self.lms < 0) | (self.lms > self.scale), dim=(1, 2, 3)))
        pan_count = torch.count_nonzero((self.pan < 0) | (self.pan > self.scale))
        pan_images = torch.count_nonzero(torch.count_nonzero((self.pan < 0) | (self.pan > self.scale), dim=(1, 2, 3)))
        print(f"MS out of range: {ms_count} Pixels in {ms_images} images")
        print(f"LMS out of range: {lms_count} Pixels in {lms_images} images")
        print(f"PAN out of range: {pan_count} Pixels in {pan_images} images")
        if (self.has_gt):
            gt_count = torch.count_nonzero((self.gt < 0) | (self.gt > self.scale))
            gt_images = torch.count_nonzero(torch.count_nonzero((self.gt < 0) | (self.gt > self.scale), dim=(1, 2, 3)))
            print(f"GT out of range: {gt_count} Pixels in {gt_images} images")