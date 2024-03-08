from abc import ABCMeta, abstractmethod
import os
from glob import glob
import time
import shutil
import logging
from datetime import datetime
import json
import inspect
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
from tqdm import tqdm

from .seed import seed_everything
from .git import git, get_git_commit
from .log import BufferedReporter, to_rgb
from ..dataset.h5pan import H5PanDataset

class SimplePanTrainer(metaclass=ABCMeta):
    cfg: dict
    
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    
    train_dataset: H5PanDataset
    val_dataset: H5PanDataset
    test_dataset: H5PanDataset
    
    train_loader: DataLoader
    val_loader: DataLoader
    
    out_dir: str
    
    disable_alloc_cache: bool
    
    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError
    
    @abstractmethod
    def _create_model(self, cfg):
        raise NotImplementedError
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(f"canconv.{cfg['exp_name']}")
        self.logger.setLevel(logging.INFO)
        seed_everything(cfg["seed"])
        self.logger.info(f"Seed set to {cfg['seed']}")
        
        self.dev = torch.device(cfg['device'])
        self.logger.info(f"Using device: {self.dev}")
        self._create_model(cfg)
        self.forward({
            'gt': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'ms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 16, 16),
            'lms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'pan': torch.randn(cfg['batch_size'], 1, 64, 64)
        })
        self.disable_alloc_cache = cfg.get("disable_alloc_cache", False)
        self.logger.info(f"Model loaded.")
        
    def _load_dataset(self):
        self.train_dataset = H5PanDataset(self.cfg["train_data"])
        self.val_dataset = H5PanDataset(self.cfg["val_data"])
        self.test_dataset = H5PanDataset(self.cfg["test_reduced_data"])
        
    def _create_output_dir(self):
        self.out_dir = os.path.join('runs', self.cfg["exp_name"])
        os.makedirs(os.path.join(self.out_dir, 'weights'), exist_ok=True)
        logging.info(f"Output dir: {self.out_dir}")
            
    def _dump_config(self):
        with open(os.path.join(self.out_dir, "cfg.json"), "w") as file:
            self.cfg["git_commit"] = get_git_commit()
            self.cfg["run_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
            json.dump(self.cfg, file, indent=4)
            
        try:
            source_path = inspect.getsourcefile(self.__class__)
            assert source_path is not None
            source_path = os.path.dirname(source_path)
            shutil.copytree(source_path, os.path.join(self.out_dir, "source"), ignore=shutil.ignore_patterns('*.pyc', '__pycache__'), dirs_exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to copy source code: ")
            self.logger.exception(e)
            
    def _on_train_start(self):
        pass
    
    def _on_val_start(self):
        pass
    
    def _on_epoch_start(self, epoch):
        pass
    
    @torch.no_grad()
    def run_test(self, dataset: H5PanDataset):
        self.model.eval()
        sr = torch.zeros(
            dataset.lms.shape[0], dataset.lms.shape[1], dataset.pan.shape[2], dataset.pan.shape[3], device=self.dev)
        for i in range(len(dataset)):
            sr[i:i+1] = self.forward(dataset[i:i+1])
        return sr

    @torch.no_grad()
    def run_test_for_selected_image(self, dataset, image_ids):
        self.model.eval()
        sr = torch.zeros(
            len(image_ids), dataset.lms.shape[1], dataset.pan.shape[2], dataset.pan.shape[3], device=self.dev)
        for i, image_id in enumerate(image_ids):
            sr[i:i+1] = self.forward(dataset[image_id:image_id+1])
        return sr
        
    def train(self):
        self._load_dataset()
        train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.cfg['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
        val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=self.cfg['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
        self.logger.info(f"Dataset loaded.")
        
        self._create_output_dir()
        self._dump_config()
        self._on_train_start()
        
        writer = SummaryWriter(log_dir=self.out_dir)
        train_loss = BufferedReporter(f'train/{self.criterion.__class__.__name__}', writer)
        val_loss = BufferedReporter(f'val/{self.criterion.__class__.__name__}', writer)
        train_time = BufferedReporter('train/time', writer)
        val_time = BufferedReporter('val/time', writer)
        
        self.logger.info(f"Begin Training.")
        
        for epoch in tqdm(range(1, self.cfg['epochs'] + 1, 1)):
            self._on_epoch_start(epoch)
            
            self.model.train()
            for batch in tqdm(train_loader):
                start_time = time.time()
                
                self.model.zero_grad()
                sr = self.forward(batch)
                loss = self.criterion(sr, batch['gt'].to(self.dev))
                train_loss.add_scalar(loss.item())
                loss.backward()
                self.optimizer.step()
                
                if self.disable_alloc_cache:
                    torch.cuda.empty_cache()
                
                train_time.add_scalar(time.time() - start_time)
            train_loss.flush(epoch)
            train_time.flush(epoch)
            self.scheduler.step()
            self.logger.debug(f"Epoch {epoch} train done")
            
            if epoch % self.cfg['val_interval'] == 0:
                self._on_val_start()
                with torch.no_grad():
                    self.model.eval()
                    for batch in val_loader:
                        start_time = time.time()
                        sr = self.forward(batch)
                        loss = self.criterion(sr, batch['gt'].to(self.dev))
                        val_loss.add_scalar(loss.item())
                        val_time.add_scalar(time.time() - start_time)
                    val_loss.flush(epoch)
                    val_time.flush(epoch)
                self.logger.debug(f"Epoch {epoch} val done")
                
            if epoch % self.cfg['checkpoint'] == 0 or ("save_first_epoch" in self.cfg and epoch <= self.cfg["save_first_epoch"]):
                torch.save(self.model.state_dict(), os.path.join(
                    self.out_dir, f'weights/{epoch}.pth'))
                self.logger.info(f"Epoch {epoch} checkpoint saved")
        
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "weights/final.pth"))
        self.logger.info(f"Training finished.")