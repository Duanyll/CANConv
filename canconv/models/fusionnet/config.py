import torch
import torch.nn as nn

from .model import FusionNet

from canconv.util.trainer import SimplePanTrainer

class FusionNetTrainer(SimplePanTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def forward(self, batch):
        return self.model(batch['lms'].to(self.dev), batch['pan'].to(self.dev))
    
    def _create_model(self, cfg):
        self.criterion = nn.MSELoss(reduction='mean').to(self.dev)
        self.model = FusionNet(cfg['spectral_num']).to(self.dev)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["learning_rate"], weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg["lr_step_size"])
