import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    accuracy_score,
)
from typing import Tuple, List

from ecg_classifier.utils import log_output


def get_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: np.ndarray
) -> Tuple[float, float]:
    tp, fp, tn, fn = [], [], [], []
    sensitivity, specificity, precision = [], [], []
    roc_auc = []

    y_pred = (y_probs >= threshold).astype(np.float32)

    log_output("Confusion matrix:")
    for i in range(0, y_true.shape[1]):
        tp_i, fp_i, tn_i, fn_i, sens, spec, prec = compute_confusion_metrics(
            y_true[:, i], y_pred[:, i]
        )

        log_output(
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

    log_output("\nMicro averaging:")
    log_output(
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
        sensitivity, specificity, precision
    )

    log_output("\nMacro averaging:")
    log_output(
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

    log_output(f"\nROC AUC: {np.mean(roc_auc):.4f}")

    log_output(
        f"\nClassification report from sklearn:\n{classification_report(y_true, y_pred)}"
    )


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
    tp_sum = sum(tp)
    fp_sum = sum(fp)
    tn_sum = sum(tn)
    fn_sum = sum(fn)

    micro_sens = tp_sum / (tp_sum + fn_sum)
    micro_spec = tn_sum / (tn_sum + fp_sum)
    micro_prec = tp_sum / (tp_sum + fp_sum)

    micro_f1 = 2 * micro_sens * micro_prec / (micro_sens + micro_prec)

    return micro_sens, micro_spec, micro_prec, micro_f1


def compute_macro_average(
    sens: List[float], spec: List[float], prec: List[float]
) -> Tuple[float, float, float, float]:
    macro_sens = np.mean(sens)
    macro_spec = np.mean(spec)
    macro_prec = np.mean(prec)

    macro_f1 = np.mean([2 * s * p / (s + p) for s, p in zip(sens, prec)])

    return macro_sens, macro_spec, macro_prec, macro_f1


def compute_metric_best_thr(
    y_true_class: np.ndarray, y_prob_class: np.ndarray, alpha: float = 0.5
) -> float:
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"

    fpr, sens, thresholds = roc_curve(y_true_class, y_prob_class)
    weighted_sum = alpha * sens + (1 - alpha) * (1 - fpr)

    best_idx = np.argmax(weighted_sum)
    return thresholds[best_idx]


def find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    best_threshold = []

    for i in range(0, y_true.shape[1]):
        best_threshold.append(
            compute_metric_best_thr(y_true[:, i], y_prob[:, i], alpha)
        )

    return best_threshold


def find_best_threshold_mtl(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(np.round(y_prob, 4))
    accuracies = []

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        accuracies.append(accuracy_score(y_true, preds))

    best_idx = np.argmax(accuracies)

    return thresholds[best_idx]


def get_metrics_mtl(
    y_true_ecg: np.ndarray,
    y_probs_ecg: np.ndarray,
    y_true_meta: np.ndarray,
    y_probs_meta: np.ndarray,
    threshold_ecg: np.ndarray,
    threshold_meta: np.ndarray,
) -> Tuple[float, float]:
    get_metrics(y_true_ecg, y_probs_ecg, threshold_ecg)

    preds_meta = (y_probs_meta >= threshold_meta).astype(int)
    accuracy = accuracy_score(y_true_meta, preds_meta)

    log_output(f"Accuracy for sex prediction: {accuracy}")
