'''
Stage 5: evaluate the saved fraud model on the out-of-time test set.
Writes artifacts/reports/fraud/metrics.json and figures.

    fraud-evaluate [--config configs/default.yaml] [--shap-sample 10000]
'''

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from card_risk.config import load_config
from card_risk.evaluation.metrics import compute_binary_classification_metrics, summarize_score_deciles
from card_risk.evaluation.plots import (
    plot_confusion_matrix,
    plot_precision_recall_vs_threshold,
    plot_roc_curve,
    plot_shap_summary,
    save_fig,
)
from card_risk.fraud.data import get_label, load_fraud_model_splits
from card_risk.fraud.model import FraudModel
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)


def compute_fraud_dollar_metrics(df_test, y_test, flagged) -> dict:
    '''
    Business view of the model: share of fraud dollars caught, and the fraud loss rate
    per dollar of purchases (used by the credit policy simulator).
    :param df_test: test feature DataFrame
    :param y_test: fraud labels
    :param flagged: boolean array of flagged transactions
    :return: dict of dollar metrics
    '''
    amount = df_test["transactionAmount"].to_numpy()
    is_fraud = np.asarray(y_test).astype(bool)
    is_purchase = (df_test["transactionType"] == "PURCHASE").to_numpy()
    fraud_dollars = amount[is_fraud].sum()
    return {
        "fraud_dollars": float(fraud_dollars),
        "fraud_dollars_caught": float(amount[is_fraud & flagged].sum()),
        "fraud_dollar_recall": float(amount[is_fraud & flagged].sum() / fraud_dollars),
        "fraud_dollars_per_purchase_dollar": float(
            amount[is_fraud & is_purchase].sum() / amount[is_purchase].sum()
        ),
        "transactions_flagged_per_day": float(
            flagged.sum() / max(df_test["transactionDateTime"].dt.normalize().nunique(), 1)
        ),
    }


def evaluate_fraud_model(config, shap_sample: int = 10_000) -> dict:
    '''
    Scores the out-of-time test set, saves metrics and figures.
    :param config: pipeline Config
    :param shap_sample: number of test rows to explain with SHAP
    :return: metrics dict
    '''
    model = FraudModel.load(config.paths.model_dir)
    df_test = load_fraud_model_splits(config)["test"]
    y_test = get_label(df_test)
    scores = model.predict_proba(df_test)
    flagged = scores >= model.threshold

    metrics = compute_binary_classification_metrics(y_test, scores, model.threshold)
    metrics.update(compute_fraud_dollar_metrics(df_test, y_test, flagged))
    metrics["test_period"] = [
        str(df_test["transactionDateTime"].min()),
        str(df_test["transactionDateTime"].max()),
    ]

    reports_dir = config.paths.reports_dir / "fraud"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    summarize_score_deciles(y_test, scores).to_csv(reports_dir / "score_deciles.csv")

    ##### Figures #####
    plot_roc_curve(
        y_test, scores, "Fraud model ROC (out-of-time test)", label=f"AUC = {metrics['roc_auc']:.3f}"
    )
    save_fig(reports_dir / "roc_curve.png")
    plot_precision_recall_vs_threshold(y_test, scores, "Fraud model precision / recall", model.threshold)
    save_fig(reports_dir / "precision_recall_vs_threshold.png")
    plot_confusion_matrix(y_test, flagged, f"Fraud model at threshold {model.threshold:.3f}")
    save_fig(reports_dir / "confusion_matrix.png")

    X_sample = model.transform(df_test.sample(min(shap_sample, len(df_test)), random_state=42))
    contributions = model.classifier.get_booster().predict(
        xgb.DMatrix(X_sample), pred_contribs=True, iteration_range=(0, model.classifier.best_iteration + 1)
    )
    plt.figure()
    plot_shap_summary(contributions, X_sample, "Fraud model SHAP summary")
    save_fig(reports_dir / "shap_summary.png")

    logger.info("Test metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--shap-sample", type=int, default=10_000)
    args = parser.parse_args(argv)
    evaluate_fraud_model(load_config(args.config), args.shap_sample)


if __name__ == "__main__":
    main()
