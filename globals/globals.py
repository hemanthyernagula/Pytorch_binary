import sys
sys.path.append("../")


from datetime import datetime

LOG_FILE_PATH = f"logs/out_{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_LEVEL = "INFO"
from logger.log import  logger


import os
print(os.listdir())
import torch as th
from models.load_data import (
    train_data,
    test_data,
    validation_data,
    dataloader,
    BATCH_SIZE
)

from torch import nn
from tqdm import tqdm,tqdm_notebook,trange

import torch.optim as optim

VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
VOCAB_SIZE = len(dataloader.vocab)
EPOCHS = 10

DEVICE = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
logger.info(f"Starting with {DEVICE} device")


class Optimizer:
    def __init__(self,model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)

    def step(self):
        self.optimizer.step()
        