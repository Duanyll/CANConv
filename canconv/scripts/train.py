import logging
import os
import importlib
import argparse
import json
import sys
import time
from canconv.util.log import save_mat_data


def main(model_name, cfg=None, save_mat=True, preset=None, override=None):
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

    trainer = module.Trainer(cfg)
    trainer.train()

    if save_mat:
        sr = trainer.run_test(trainer.test_dataset)
        save_mat_data(sr, trainer.test_dataset.scale, trainer.out_dir)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                f"logs/train_{int(time.time())}_{os.getpid()}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Train script invoked with args: {sys.argv}")

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("preset", nargs='?', type=str, default=None)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--save_mat", type=bool, default=True)
    parser.add_argument("--override", type=str, default=None)
    args = parser.parse_args()

    cfg = None
    if args.cfg is not None:
        with open(args.cfg, 'r') as f:
            cfg = json.load(f)
    try:
        main(args.model_name, cfg, args.save_mat, args.preset, args.override)
    except Exception as e:
        logging.exception(e)
        raise e
