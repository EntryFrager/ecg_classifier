import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from typing import Tuple, List, Callable, Optional


def get_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: np.ndarray
) -> Tuple[float, float]:
    tp, fp, tn, fn = [], [], [], []
    sensitivity, specificity, precision = [], [], []
    roc_auc = []

    y_pred = (y_probs >= threshold).astype(np.float32)

    print("Confusion matrix:")
    for i in range(0, y_true.shape[1]):
        tp_i, fp_i, tn_i, fn_i, sens, spec, prec = compute_confusion_metrics(
            y_true[:, i], y_pred[:, i]
        )

        print(
            pd.DataFrame([{"TP": tp_i, "FP": fp_i, "TN": tn_i, "FN": fn_i}]).to_string(
                index=False
            )
        )

        tp.append(tp_i)
        fp.append(fp_i)
        tn.append(tn_i)
        fn.append(fn_i)

        sensitivity.append(sens)
        specificity.append(spec)
        precision.append(prec)

        roc_auc.append(roc_auc_score(y_true[:, i], y_probs[:, i]))

    # micro averaging

    micro_sens, micro_spec, micro_prec, micro_f1 = compute_micro_average(tp, fp, tn, fn)

    print("\nMicro averaging:")
    print(
        pd.DataFrame.from_dict(
            {
                "sensitivity": micro_sens,
                "specificity": micro_spec,
                "precision": micro_prec,
                "f1 score": micro_f1,
            },
            orient="index",
        ).to_string(header=False)
    )

    # macro averaging

    macro_sens, macro_spec, macro_prec, macro_f1 = compute_macro_average(
        sens, spec, prec
    )

    print("\nMacro averaging:")
    print(
        pd.DataFrame.from_dict(
            {
                "sensitivity": macro_sens,
                "specificity": macro_spec,
                "precision": macro_prec,
                "f1 score": macro_f1,
            },
            orient="index",
        ).to_string(header=False)
    )

    # roc auc and classification report

    print(f"\nROC AUC: {np.mean(roc_auc):.4f}")

    print(
        f"\nClassification report from sklearn:\n{classification_report(y_true, y_pred)}"
    )

    return macro_sens, macro_spec


def compute_confusion_metrics(
    y_true_class: np.ndarray, y_pred_class: np.ndarray
) -> Tuple[int, int, int, int, float, float, float]:
    conf_matrix = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)

    return tp, fp, tn, fn, sens, spec, prec


def compute_micro_average(
    tp: List[int], fp: List[int], tn: List[int], fn: List[int]
) -> Tuple[float, float, float, float]:
    tp_mean, fp_mean, tn_mean, fn_mean = (
        np.mean(tp),
        np.mean(fp),
        np.mean(tn),
        np.mean(fn),
    )
    micro_sens = tp_mean / (tp_mean + fn_mean)
    micro_spec = tn_mean / (tn_mean + fp_mean)
    micro_prec = tp_mean / (tp_mean + fp_mean)

    micro_f1 = 2 * micro_sens * micro_prec / (micro_sens + micro_prec)

    return micro_sens, micro_spec, micro_prec, micro_f1


def compute_macro_average(
    sens: float, spec: float, prec: float
) -> Tuple[float, float, float, float]:
    macro_sens = np.mean(sens)
    macro_spec = np.mean(spec)
    macro_prec = np.mean(prec)

    macro_f1 = 2 * macro_sens * macro_prec / (macro_sens + macro_prec)

    return macro_sens, macro_spec, macro_prec, macro_f1


def default_compute_metric_best_thr(y_true_class, y_prob_class) -> float:
    prec, sens, thresholds = precision_recall_curve(y_true_class, y_prob_class)
    f1 = 2 * sens[:-1] * prec[:-1] / (sens[:-1] + prec[:-1])

    best_idx = np.nanargmax(f1)
    return thresholds[best_idx]


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    compute_metric_best_thr: Optional[Callable[[np.ndarray, np.ndarray], float]],
) -> np.ndarray:
    if compute_metric_best_thr is None:
        compute_metric_best_thr = default_compute_metric_best_thr

    best_threshold = []

    for i in range(0, y_true.shape[1]):
        best_threshold.append(compute_metric_best_thr(y_true[:, i], y_prob[:, i]))

    return best_threshold
