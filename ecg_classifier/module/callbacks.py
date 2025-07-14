import os
import torch
import torch.nn as nn
import numpy as np

import copy

from ..utils.log import log_output


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.best_loss = None
        self.best_model = None
        self.best_threshold = None

        self.patience = patience
        self.counter = 0

        self.stop = False

    def __call__(
        self,
        loss: float,
        net: nn.Module,
        threshold: np.ndarray,
    ) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
            self.best_model = copy.deepcopy(net)
            self.best_threshold = threshold
        elif loss <= self.best_loss:
            self.best_loss = loss
            self.best_model = copy.deepcopy(net)
            self.best_threshold = threshold
            self.counter = 0
            log_output(
                f"\nBest Loss: {self.best_loss:.4f}\n"
                f"Best threshold: {[round(float(x), 4) for x in self.best_threshold]}"
            )
        else:
            self.counter += 1

        log_output(f"EarlyStopping: {self.counter} / {self.patience}\n")

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

    def save_best_model(self):
        os.makedirs("save_best_models", exist_ok=True)

        torch.save(self.best_model.state_dict(), "save_best_models/best_model.pt")
        torch.save(self.best_threshold, "save_best_models/best_threshold.pt")
