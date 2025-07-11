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


class Normalize(torch.nn.Module):
    def __init__(
        self,
        mean_ecg: torch.Tensor,
        std_ecg: torch.Tensor,
        mean_meta: torch.Tensor,
        std_meta: torch.Tensor,
        mean_pqrst: torch.Tensor,
        std_pqrst: torch.Tensor,
    ) -> None:
        super().__init__()

        self.mean_ecg = mean_ecg
        self.std_ecg = std_ecg

        self.mean_meta = mean_meta
        self.std_meta = std_meta

        self.mean_pqrst = mean_pqrst
        self.std_pqrst = std_pqrst

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample["ecg_signals"] = (sample["ecg_signals"] - self.mean_ecg) / self.std_ecg

        if self.mean_meta is not None:
            sample["metadata"] = (sample["metadata"] - self.mean_meta) / self.std_meta

        if self.mean_pqrst is not None:
            sample["pqrst_features"] = (
                sample["pqrst_features"] - self.mean_pqrst
            ) / self.std_pqrst

        return sample


def get_stat(dataset: pd.DataFrame, target_labels: Dict[str, Any]) -> None:
    for key, _ in target_labels.items():
        label = dataset[key].value_counts()
        print(f"{key} unique labels: {label}")


def get_mean_std(
    value: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(
        value.mean(axis=axis).astype(np.float32), dtype=torch.float32
    ), torch.tensor(value.std(axis=axis).astype(np.float32), dtype=torch.float32)
