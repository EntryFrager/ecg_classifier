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
        self.best_threshold_ecg = None
        self.best_threshold_meta = None

        self.patience = patience
        self.counter = 0

        self.stop = False

    def __call__(
        self,
        loss: float,
        net: nn.Module,
        threshold_ecg: np.ndarray,
        threshold_meta: np.ndarray = None,
    ) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
            self.best_model = copy.deepcopy(net)
            self.best_threshold_ecg = threshold_ecg
            self.best_threshold_meta = threshold_meta
        elif loss <= self.best_loss:
            self.best_loss = loss
            self.best_model = copy.deepcopy(net)
            self.best_threshold_ecg = threshold_ecg
            self.best_threshold_meta = threshold_meta
            self.counter = 0
            log_output(
                f"\nBest Loss: {self.best_loss:.4f}\n"
                f"\nBest threshold for ecg: {[round(float(x), 4) for x in self.best_threshold_ecg]}\n"
                f"Best threshold for meta: {round(float(self.best_threshold_meta), 4)}"
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
        np.save("save_best_models/best_threshold_ecg.npy", self.best_threshold_ecg)
        np.save("save_best_models/best_threshold_meta.npy", self.best_threshold_meta)
