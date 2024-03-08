import torch
import torch.nn as nn

from .model import CANFusionNet

from canconv.util.trainer import SimplePanTrainer
from canconv.layers.kmeans import KMeansCacheScheduler, reset_cache

class CANFusionNetTrainer(SimplePanTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _create_model(self, cfg):
        self.criterion = nn.MSELoss(reduction='mean').to(self.dev)
        self.model = CANFusionNet(cfg['spectral_num']).to(self.dev)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["learning_rate"], weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg["lr_step_size"])
        
        self.km_scheduler = KMeansCacheScheduler(cfg['kmeans_cache_update'])

    def _on_train_start(self):
        reset_cache(len(self.train_dataset))

    def _on_epoch_start(self, epoch):
        self.km_scheduler.step()

    def forward(self, data):
        if "index" in data:
            return self.model(data['lms'].to(self.dev), data['pan'].to(self.dev), data['index'].to(self.dev))
        else:
            return self.model(data['lms'].to(self.dev), data['pan'].to(self.dev))
