import logging
import os
import importlib
import argparse
import json
import sys
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

from canconv.util.log import to_rgb

def show(result_dir, save_file=None):
    if os.path.basename(result_dir) != "results":
        result_dir = os.path.join(result_dir, "results")
    filenames = [f"output_mulExm_{i}.mat" for i in range(20)]
    sr = [loadmat(os.path.join(result_dir, filename))["sr"] for filename in filenames]
    sr = np.array(sr)
    sr = to_rgb(sr)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sr[i])
        ax.axis("off")
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    parser.add_argument("save_file", type=str, default=None)
    args = parser.parse_args()
    show(args.result_dir, args.save_file)