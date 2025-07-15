import torch

from .log import log_output


def setup_device():
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_output(f"Training will take on {device}")

    return device
