import pandas as pd
import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict, Any, Iterable, Callable


class Compose:
    def __init__(self, transforms: Iterable[Callable[[Any], Any]]) -> None:
        if transforms is None:
            raise ValueError("Compose expected a list of transforms, but got None.")
        self.transforms = transforms

    def __call__(self, obj: Any) -> Any:
        for t in self.transforms:
            obj = t(obj)

        return obj


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        for key, _ in sample.items():
            sample[key] = torch.tensor(sample[key], dtype=torch.float32)

        return sample


def get_stat(dataset: pd.DataFrame, target_labels: Dict[str, Any]) -> None:
    for key, _ in target_labels.items():
        label = dataset[key].value_counts()
        print(f"{key} unique labels: {label}")


def min_max_norm(sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    for key, value in sample.items():
        if key != "labels":
            min = value.min()
            max = value.max()
            sample[key] = (value - min) / (max - min)

    return sample


def get_mean_std(
    value: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(
        value.mean(axis=axis).astype(np.float32), dtype=torch.float32
    ), torch.tensor(value.std(axis=axis).astype(np.float32), dtype=torch.float32)
