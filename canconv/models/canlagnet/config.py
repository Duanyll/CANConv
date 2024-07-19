import torch
import torch.nn as nn

from .model import CANLAGNET

from canconv.util.trainer import SimplePanTrainer
from canconv.layers.kmeans import KMeansCacheScheduler, reset_cache

cfg = {
    'seed': 10,
    'train_data': "/datasets/wv3/train_wv3.h5",
    'val_data': "/datasets/wv3/valid_wv3.h5",
    'test_reduced_data': "/datasets/wv3/test_wv3_multiExm1.h5",
    'test_origscale_data': "/datasets/wv3/test_wv3_OrigScale_multiExm1.h5",
    'spectral_num': 8,
    'batch_size': 32,
    'device': 'cuda:0',
    'learning_rate': 1e-3,
    'lr_step_size': 500,
    'epochs': 1000,
    'checkpoint': 25,
    'exp_name': "canlagnet",
    'cluster_num': 32,
    "filter_threshold": 0,
    "kmeans_cache_update": [[100, 10], [300, 20], 50],
    "val_interval": 10,
    "disable_alloc_cache": False,
    "origscale_image": [16]
}

class CANLAGNetTrainer(SimplePanTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
    def _create_model(self, cfg):
        self.criterion = nn.MSELoss(reduction='mean').to(self.dev)
        self.model = CANLAGNET(cfg['spectral_num'], cfg['cluster_num']).to(self.dev)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"], weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg["lr_step_size"])
        
        self.km_scheduler = KMeansCacheScheduler(cfg['kmeans_cache_update'])
        
    def _on_train_start(self):
        reset_cache(len(self.train_dataset))
        
    def _on_epoch_start(self, epoch):
        self.km_scheduler.step()

    def forward(self, data):
        if "index" in data:
            return self.model(data['pan'].to(self.dev), data['lms'].to(self.dev), data['index'].to(self.dev))
        else:
            return self.model(data['pan'].to(self.dev), data['lms'].to(self.dev))