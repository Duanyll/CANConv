from .model import CANDiCNN
from .config import CANDiCNNTrainer as Trainer

import os.path
import json
# load config from default.json
with open(os.path.join(os.path.dirname(__file__), 'default.json'), 'r') as file:
    cfg = json.load(file)