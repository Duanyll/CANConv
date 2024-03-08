import os
import numpy as np
import scipy.io as sio
import torch
from einops import rearrange

class BufferedReporter:
    def __init__(self, tag: str, writer, period=0) -> None:
        self.tag = tag
        self.writer = writer
        self.period = period
        self.count = 0
        self.data = []

    def flush(self, count=0):
        val = np.nanmean(self.data)
        if (self.period == 0):
            if (count != 0):
                self.writer.add_scalar(self.tag, val, count)
            else:
                self.writer.add_scalar(self.tag, val)

        else:
            self.writer.add_scalar(self.tag, val, self.count)
        self.data = []

    def add_scalar(self, x):
        self.count += 1
        self.data.append(x)
        if len(self.data) == self.period:
            self.flush()
            
def save_mat_data(sr, scale, output_dir):
    mat_dir = os.path.join(output_dir, "results")
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    data_size = sr.shape[0]
    for i in range(data_size):
        batch = rearrange(sr[i], 'c h w -> h w c') * scale
        batch = batch.cpu().numpy()
        out_file = os.path.join(mat_dir, f'output_mulExm_{i}.mat')
        sio.savemat(out_file, {"sr": batch})
        
def find_epoch(run_name):
    if os.path.exists(f"runs/{run_name}/weights"):
        if os.path.exists(f"runs/{run_name}/weights/final.pth"):
            return "final"
        # Find runs/{run_name}/weights/*.pth, where * is the largest number
        files = os.listdir(f"runs/{run_name}/weights")
        files = [int(file.split(".")[0]) for file in files]
        files.sort()
        return files[-1]
    raise FileNotFoundError(f"Cannot find weights in runs/{run_name}/weights")

def linstretch(ImageToView, tol_low, tol_high):
    N, M = ImageToView.shape
    NM = N * M
    b = ImageToView[:, :].reshape(NM, 1).to(torch.float32)
    sorted_b, _ = torch.sort(b, dim=0)
    t_low = sorted_b[int(NM * tol_low)]
    t_high = sorted_b[int(NM * tol_high)]
    b = torch.clamp((b - t_low) / (t_high - t_low), 0, 1)
    return b.reshape(N, M)

def to_rgb(x: torch.Tensor | np.ndarray, tol_low=0.01, tol_high=0.99):
    x = torch.Tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        has_batch = False
        x = x.unsqueeze(0)
    else:
        has_batch = True
    # Try to detect BCHW or BHWC
    if x.shape[1] > 8:
        x = rearrange(x, 'b h w c -> b c h w')
    c = x.shape[1]
    if c == 1:
        x = torch.cat([x, x, x], dim=1)
    elif c == 3:
        pass
    elif c == 4:
        x = x[:, [2, 1, 0], :, :]
    elif c == 8:
        x = x[:, [4, 2, 1], :, :]
    else:
        raise ValueError(f"Unsupported channel number: {c}")
    b, c, h, w = x.shape
    x = rearrange(x, 'b c h w -> c (b h w)')
    sorted_x, _ = torch.sort(x, dim=1)
    t_low = sorted_x[:, int(b * h * w * tol_low)].unsqueeze(1)
    t_high = sorted_x[:, int(b * h * w * tol_high)].unsqueeze(1)
    x = torch.clamp((x - t_low) / (t_high - t_low), 0, 1)
    x = rearrange(x, 'c (b h w) -> b h w c', b=b, c=c, h=h, w=w)
    if not has_batch:
        x = x.squeeze(0)
    return x.cpu().numpy()