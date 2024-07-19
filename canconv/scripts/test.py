import json
import argparse
import os
import importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from canconv.util.seed import seed_everything
from canconv.dataset.h5pan import H5PanDataset
from canconv.util.log import save_mat_data

@torch.no_grad()
def run_test(model_name, weight_file, test_dataset="reduced", cfg=None, preset=None, override=None, output_dir=None):
    module = importlib.import_module(f"canconv.models.{model_name}")

    if cfg is None:
        cfg = module.cfg
    if preset is not None:
        with open("presets.json", 'r') as f:
            prresets = json.load(f)
        cfg = cfg | prresets[preset]
        cfg["exp_name"] += f'_{preset}'
    if override is not None:
        cfg = cfg | json.loads(override)

    if test_dataset == "reduced":
        dataset = H5PanDataset(cfg["test_reduced_data"], scale=cfg["dataset_scale"])
    elif test_dataset == "origscale":
        dataset = H5PanDataset(cfg["test_origscale_data"], scale=cfg["dataset_scale"])
    else:
        dataset = H5PanDataset(test_dataset, scale=cfg["dataset_scale"])

    trainer = module.Trainer(cfg)
    trainer.model.load_state_dict(torch.load(weight_file))
    trainer.model.eval()

    sr = trainer.run_test(dataset)
    if output_dir is None:
        output_dir = f"{os.path.dirname(weight_file)}/{os.path.basename(weight_file).split('.')[0]}"
    save_mat_data(sr, dataset.scale, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("weight_file", type=str)
    parser.add_argument("test_dataset", type=str)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = None
    if args.cfg is not None:
        with open(args.cfg, 'r') as f:
            cfg = json.load(f)
    run_test(args.model_name, args.weight_file, args.test_dataset, cfg, args.preset, args.override, args.output_dir)