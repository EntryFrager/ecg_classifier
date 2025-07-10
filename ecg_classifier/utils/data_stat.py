from typing import Dict, Any
import pandas as pd


def get_stat(dataset: pd.DataFrame, target_labels: Dict[str, Any]) -> None:
    for key, _ in target_labels.items():
        label = dataset[key].value_counts()
        print(f"{key} unique labels: {label}")
