import numpy as np
from sklearn.metrics import recall_score

from card_risk.evaluation import (
    compute_binary_classification_metrics,
    compute_ks_statistic,
    find_threshold_for_target_recall,
    summarize_score_deciles,
)


def test_ks_is_one_for_perfect_separation_and_zero_for_constant_scores():
    y = np.array([0, 0, 1, 1])
    assert compute_ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert compute_ks_statistic(y, np.full(4, 0.5)) == 0.0


def test_threshold_reaches_target_recall(rng):
    y = rng.integers(0, 2, 2000)
    scores = np.clip(0.3 * y + rng.normal(0.4, 0.2, 2000), 0, 1)
    threshold = find_threshold_for_target_recall(y, scores, 0.8)
    assert recall_score(y, scores >= threshold) >= 0.8


def test_gini_is_two_auc_minus_one(rng):
    y = rng.integers(0, 2, 500)
    scores = rng.random(500)
    metrics = compute_binary_classification_metrics(y, scores, threshold=0.5)
    assert np.isclose(metrics["gini"], 2 * metrics["roc_auc"] - 1)
    assert {"precision", "recall", "f1", "alert_rate"} <= set(metrics)


def test_score_deciles_cover_all_rows(rng):
    y = rng.integers(0, 2, 1000)
    table = summarize_score_deciles(y, rng.random(1000))
    assert table["n"].sum() == 1000
    assert np.isclose(table["share_of_positives"].sum(), 1.0)
