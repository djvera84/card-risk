'''
Train and evaluate the underwriting model, then score the out-of-time through-the-door
population for the credit simulator.

    underwriting-train [--config configs/default.yaml]

Split by vintage: the first train_vintages months train the model, the next
valid_vintages months are for early stopping, and the remaining months are the
out-of-time test set. Models are fit on booked applicants only (the only ones with
outcomes); the simulator then scores everyone, including applicants the historical
rule declined.
'''

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from card_risk.config import load_config
from card_risk.evaluation.metrics import compute_binary_classification_metrics, summarize_score_deciles
from card_risk.evaluation.plots import plot_calibration, plot_roc_curve, plot_shap_summary, save_fig
from card_risk.logging_utils import get_logger
from card_risk.underwriting.features import build_underwriting_features
from card_risk.underwriting.model import UnderwritingModel, convert_pd_to_underwriting_score

logger = get_logger(__name__)

LABEL = "defaulted_12m"
SCORED_COLUMNS = [
    "application_id",
    "application_date",
    "vintage",
    "fico_score",
    "annual_income",
    "requested_credit_limit",
    "is_booked",
    "true_pd_12m",
    "revolving_balance_demand",
    "monthly_spend_demand",
]


def split_applications_by_vintage(df: pd.DataFrame, train_vintages: int, valid_vintages: int) -> dict:
    '''
    Out-of-time split of the applications by vintage.
    :param df: simulated applications
    :param train_vintages: number of initial vintages to train on
    :param valid_vintages: number of following vintages for early stopping
    :return: dict of "train", "valid", "test" DataFrames (all applicants)
    '''
    return {
        "train": df[df["vintage"] <= train_vintages],
        "valid": df[(df["vintage"] > train_vintages) & (df["vintage"] <= train_vintages + valid_vintages)],
        "test": df[df["vintage"] > train_vintages + valid_vintages],
    }


def booked_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_booked"] == 1]


def train_logistic_scorecard_baseline(X_train, y_train) -> Pipeline:
    '''
    Logistic regression baseline, the traditional scorecard approach.
    :param X_train: model matrix
    :param y_train: labels
    :return: fitted Pipeline
    '''
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    # numpy 2 on macOS Accelerate emits spurious matmul overflow warnings here; the fit is fine.
    with np.errstate(all="ignore"):
        return pipeline.fit(X_train.fillna(0), y_train)


def train_and_evaluate_underwriting_model(config) -> tuple[UnderwritingModel, dict]:
    '''
    Trains the XGBoost model and the logistic baseline on booked applicants, compares them
    on the out-of-time test vintages and writes reports.
    :param config: pipeline Config
    :return: (UnderwritingModel, metrics dict)
    '''
    uw = config.underwriting
    df = pd.read_parquet(config.paths.applications_parquet)
    splits = split_applications_by_vintage(df, uw.train_vintages, uw.valid_vintages)
    train, valid, test = (booked_only(splits[name]) for name in ["train", "valid", "test"])
    X_train, X_valid, X_test = (build_underwriting_features(d) for d in [train, valid, test])
    y_train, y_valid, y_test = train[LABEL], valid[LABEL], test[LABEL]
    logger.info("Booked applicants: train %d, valid %d, test %d", len(train), len(valid), len(test))

    ##### XGBoost with Monotone Constraints #####
    classifier = UnderwritingModel.make_classifier(
        uw.params, uw.n_estimators, uw.early_stopping_rounds, list(X_train.columns)
    )
    classifier.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    model = UnderwritingModel(classifier)

    ##### Logistic Scorecard Baseline #####
    scorecard = train_logistic_scorecard_baseline(X_train, y_train)

    ##### Out-of-Time Evaluation #####
    xgb_pd = classifier.predict_proba(X_test)[:, 1]
    with np.errstate(all="ignore"):
        scorecard_pd = scorecard.predict_proba(X_test.fillna(0))[:, 1]
    metrics = {
        "xgboost": compute_binary_classification_metrics(y_test, xgb_pd),
        "logistic_scorecard": compute_binary_classification_metrics(y_test, scorecard_pd),
        "oracle_true_pd": compute_binary_classification_metrics(y_test, test["true_pd_12m"]),
        "best_iteration": classifier.best_iteration,
        "test_vintages": [int(test["vintage"].min()), int(test["vintage"].max())],
    }
    model.metadata = {"metrics": metrics, "feature_names": list(X_train.columns)}

    reports_dir = config.paths.reports_dir / "underwriting"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    deciles = summarize_score_deciles(y_test, xgb_pd)
    deciles.to_csv(reports_dir / "score_deciles.csv")

    plot_roc_curve(
        y_test,
        xgb_pd,
        "Underwriting model ROC (out-of-time vintages)",
        label=f"XGBoost AUC = {metrics['xgboost']['roc_auc']:.3f}",
    )
    plt.gca().plot(
        *_roc_points(y_test, scorecard_pd),
        label=f"Scorecard AUC = {metrics['logistic_scorecard']['roc_auc']:.3f}",
    )
    plt.legend(loc="lower right")
    save_fig(reports_dir / "roc_curve.png")
    plot_calibration(deciles, "Underwriting model calibration by decile (out-of-time)")
    save_fig(reports_dir / "calibration.png")

    contributions = classifier.get_booster().predict(xgb.DMatrix(X_test), pred_contribs=True)
    plt.figure()
    plot_shap_summary(contributions, X_test, "Underwriting model SHAP summary")
    save_fig(reports_dir / "shap_summary.png")

    logger.info("Out-of-time metrics:\n%s", json.dumps(metrics, indent=2))
    return model, metrics


def _roc_points(y_true, y_score):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def score_through_the_door_population(config, model: UnderwritingModel) -> pd.DataFrame:
    '''
    Scores every applicant in the out-of-time test vintages (booked or not) and
    saves them for the credit simulator.
    :param config: pipeline Config
    :param model: fitted UnderwritingModel
    :return: scored DataFrame
    '''
    uw = config.underwriting
    df = pd.read_parquet(config.paths.applications_parquet)
    population = split_applications_by_vintage(df, uw.train_vintages, uw.valid_vintages)["test"]
    scored = population.loc[:, SCORED_COLUMNS].copy()
    scored["predicted_pd_12m"] = model.predict_pd(population)
    scored["underwriting_score"] = convert_pd_to_underwriting_score(scored["predicted_pd_12m"])
    scored.to_parquet(config.paths.scored_applications_parquet, index=False)
    logger.info(
        "Scored %d through-the-door applicants; mean predicted PD %.3f vs true %.3f",
        len(scored),
        scored["predicted_pd_12m"].mean(),
        np.mean(scored["true_pd_12m"]),
    )
    return scored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model, _ = train_and_evaluate_underwriting_model(config)
    model.save(config.paths.underwriting_model_dir)
    score_through_the_door_population(config, model)


if __name__ == "__main__":
    main()
