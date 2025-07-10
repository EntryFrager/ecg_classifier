import numpy as np
import pandas as pd
import wfdb
import ast
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple, List, Dict

from ecg_classifier.utils.data_stat import Compose, ToTensor, Normalize


class ECGDataset(Dataset):
    ecg_stat = {
        "mean": torch.tensor(
            [
                -0.0018,
                -0.0013,
                0.0005,
                0.0016,
                -0.0011,
                -0.0004,
                0.0002,
                -0.0009,
                -0.0015,
                -0.0017,
                -0.0008,
                -0.0021,
            ],
            dtype=torch.float32,
        ),
        "std": torch.tensor(
            [
                [
                    0.1640,
                    0.1647,
                    0.1713,
                    0.1403,
                    0.1461,
                    0.1466,
                    0.2337,
                    0.3377,
                    0.3336,
                    0.3058,
                    0.2731,
                    0.2755,
                ]
            ],
            dtype=torch.float32,
        ),
    }

    metadata_stat = {
        "mean": torch.tensor(74.2453, dtype=torch.float32),
        "std": torch.tensor(60.3269, dtype=torch.float32),
    }

    pqrst_stat = {"mean": 0, "std": 0}

    valid_fold = 9
    test_fold = 10

    def __init__(
        self,
        target_labels: Dict[str, list[str]],
        path: str = "dataset/physionet.org/files/ptb-xl/1.0.1/",
        sampling_rate: int = 100,
        use_pqrst: bool = False,
        use_metadata: bool = False,
    ) -> None:
        if target_labels is None:
            raise ValueError("Target labels should be initialized.")

        self.path = path
        self.sampling_rate = sampling_rate

        self.scp_statements = (
            pd.read_csv(path + "scp_statements.csv", index_col=0)
            .reset_index()
            .rename(columns={"index": "scp_code"})
        )
        self.ptbxl_dataset = pd.read_csv(
            path + "ptbxl_database.csv", index_col="ecg_id"
        )

        self.target_labels = target_labels

        self.labels = self._set_target_labels()
        self.ecg_signals = self._process_ecg_signals()
        self.pqrst_features = self._process_pqrst_features() if use_pqrst else None
        self.metadata = self._process_metadata() if use_metadata else None

        norm = Normalize(
            self.ecg_stat["mean"],
            self.ecg_stat["std"],
            self.metadata_stat["mean"] if use_metadata else None,
            self.metadata_stat["std"] if use_metadata else None,
            self.pqrst_stat["mean"] if use_pqrst else None,
            self.pqrst_stat["std"] if use_pqrst else None,
        )
        self.transform = Compose([ToTensor(), norm])

    def get_dataset(self) -> Tuple[Subset, Subset, Subset]:
        train_idx = np.where(
            (self.ptbxl_dataset.strat_fold != self.valid_fold)
            & (self.ptbxl_dataset["strat_fold"] != self.test_fold)
        )[0]
        val_idx = np.where((self.ptbxl_dataset.strat_fold == self.valid_fold))[0]
        test_idx = np.where((self.ptbxl_dataset.strat_fold == self.test_fold))[0]

        train = Subset(self, train_idx)
        val = Subset(self, val_idx)
        test = Subset(self, test_idx)

        return train, val, test

    def _set_target_labels(self) -> np.ndarray:
        self.ptbxl_dataset["scp_codes"] = self.ptbxl_dataset["scp_codes"].apply(
            lambda x: ast.literal_eval(x)
        )

        for label, target_labels in self.target_labels.items():
            target_values = np.zeros(len(self.ptbxl_dataset), dtype=int)
            for target_label in target_labels:
                target_values += (
                    self.ptbxl_dataset["scp_codes"]
                    .apply(lambda x: 1 if target_label in x else 0)
                    .to_numpy(dtype=int)
                )
            self.ptbxl_dataset[label] = np.where(target_values >= 1, 1, 0)

        return self.ptbxl_dataset[self.target_labels.keys()].values

    def _process_ecg_signals(self) -> List[str]:
        if self.sampling_rate == 100:
            files = [filename for filename in self.ptbxl_dataset["filename_lr"]]
        elif self.sampling_rate == 500:
            files = [filename for filename in self.ptbxl_dataset["filename_hr"]]

        return files

    def _process_metadata(self) -> np.ndarray:
        metadata = self.ptbxl_dataset[["age", "sex", "height", "weight"]].copy()

        with pd.option_context("future.no_silent_downcasting", True):
            metadata["age"] = metadata["age"].fillna(metadata["age"].median())
            metadata["height"] = metadata["height"].fillna(metadata["height"].median())
            metadata["weight"] = metadata["weight"].fillna(metadata["weight"].median())

        return metadata.values

    def _process_pqrst_features(self):
        # in develop
        return

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = {
            "ecg_signals": wfdb.rdsamp(self.path + self.ecg_signals[index])[0],
            "labels": self.labels[index],
        }

        if self.metadata is not None:
            sample["metadata"] = self.metadata[index]

        if self.pqrst_features is not None:
            sample["pqrst_features"] = self.pqrst_features[index]

        if self.transform is not None:
            sample = self.transform(sample)

        sample["ecg_signals"] = sample["ecg_signals"].transpose(0, 1)

        return sample

    def __len__(self) -> int:
        return len(self.labels)

    def get_pos_weight(self) -> torch.Tensor:
        pos_weight = []

        train_mask = (self.ptbxl_dataset.strat_fold != self.valid_fold) & (
            self.ptbxl_dataset["strat_fold"] != self.test_fold
        )
        train = self.ptbxl_dataset[train_mask]

        for label, _ in self.target_labels.items():
            unique_value = train[label].value_counts()
            pos_weight.append(unique_value[0] / unique_value[1])

        return torch.tensor(pos_weight, dtype=torch.float32)

    def close_dataset(self) -> None:
        del self.ptbxl_dataset
        del self.scp_statements
