import torch
import numpy as np
import os
import warnings
import random

warnings.filterwarnings("ignore")
DEFAULT_RANDOM_SEED = 42


def SeedBasic(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def SeedTorch(seed: int = DEFAULT_RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def SeedEverything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    SeedBasic(seed)
    SeedTorch(seed)
