import torch.nn as nn
import numpy as np

import copy


class EarlyStopping:
    def __init__(self, 
                 patience: int) -> None:
        self.best_loss  = None
        self.best_model = None
        self.best_threshold = None

        self.patience = patience
        self.counter  = 0

        self.stop = False


    def __call__(self, 
                 loss: float, 
                 sens: float, 
                 spec: float, 
                 net: nn.Module, 
                 threshold: np.ndarray) -> bool:
        if self.best_loss is None and self.best_sens is None and self.best_spec is None:
                self.best_loss  = loss
                self.best_sens  = sens
                self.best_spec  = spec
                self.best_model = copy.deepcopy(net)
                self.best_threshold = threshold
        elif loss <= self.best_loss:
            self.best_loss  = loss
            self.best_model = copy.deepcopy(net)
            self.best_threshold = threshold
            self.counter = 0
            print(f'\nBest Loss: {self.best_loss:.4f}\n'
                  f'Best threshold: {self.best_threshold}')
        else:
            self.counter += 1

        print(f"EarlyStopping: {self.counter} / {self.patience}\n")

        if self.counter >= self.patience:
            self.stop = True

        return self.stop





