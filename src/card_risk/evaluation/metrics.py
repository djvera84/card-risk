import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_ks_statistic(y_true, y_score) -> float:
    '''
    Kolmogorov-Smirnov statistic: the largest gap between the score distributions of
    positives and negatives (max of TPR - FPR). Standard in credit risk.
    :param y_true: binary labels
    :param y_score: predicted probabilities or scores
    :return: KS in [0, 1]
    '''
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def find_threshold_for_target_recall(y_true, y_score, target_recall: float) -> float:
    '''
    The highest threshold whose recall is at least target_recall. Choose it on a
    validation set, never on the test set.
    :param y_true: binary labels
    :param y_score: predicted probabilities
    :param target_recall: desired recall, e.g. 0.8
    :return: threshold
    '''
    _, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # recalls has one more entry than thresholds; recall decreases as the threshold rises.
    meets_target = recalls[:-1] >= target_recall
    if not meets_target.any():
        return float(thresholds.min())
    return float(thresholds[meets_target].max())


def compute_binary_classification_metrics(y_true, y_score, threshold: float | None = None) -> dict:
    '''
    Ranking, calibration and (optionally) thresholded metrics.
    :param y_true: binary labels
    :param y_score: predicted probabilities
    :param threshold: decision threshold for precision/recall/F1; skipped if None
    :return: dict of metric name to value
    '''
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    metrics = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc),
        "gini": float(2 * roc_auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": compute_ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "mean_predicted": float(y_score.mean()),
    }
    if threshold is not None:
        y_pred = y_score >= threshold
        metrics.update(
            {
                "threshold": float(threshold),
                "alert_rate": float(y_pred.mean()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred)),
            }
        )
    return metrics


def summarize_score_deciles(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    '''
    Rank-ordering / calibration table: observed vs predicted rate by score decile.
    :param y_true: binary labels
    :param y_score: predicted probabilities
    :param n_bins: number of quantile bins
    :return: DataFrame with one row per bin, riskiest last
    '''
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df["bin"] = pd.qcut(df["score"].rank(method="first"), n_bins, labels=False) + 1
    table = df.groupby("bin").agg(
        n=("y", "size"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        mean_predicted=("score", "mean"),
        observed_rate=("y", "mean"),
    )
    table["share_of_positives"] = df.groupby("bin")["y"].sum() / df["y"].sum()
    return table
