'''
Stage 3a (optional): compare baseline models with time-series cross validation on the
training data. Writes artifacts/reports/fraud/baselines.csv.

    fraud-baselines [--config configs/default.yaml]
'''

import argparse

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from card_risk.config import load_config
from card_risk.features.build import NUM_FEATURES
from card_risk.fraud.data import get_label, load_fraud_model_splits
from card_risk.fraud.model import FraudModel, make_categorical_encoder
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)


def make_baseline_models(random_state: int = 42) -> dict:
    '''
    Logistic regression and random forest need imputed (and, for the linear model, scaled)
    inputs; XGBoost handles missing values natively.
    :param random_state: seed
    :return: dict of model name to estimator
    '''
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, C=0.1)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=50,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def compare_baseline_models(config) -> pd.DataFrame:
    '''
    Time-series cross-validated ROC AUC for each baseline on the most recent
    baselines.max_rows training rows.
    :param config: pipeline Config
    :return: DataFrame of fold scores, mean and std per model
    '''
    df_train = load_fraud_model_splits(config)["train_full"].tail(config.baselines.max_rows)
    encoder = make_categorical_encoder().fit(df_train)
    X = FraudModel.make_model_matrix(df_train, encoder, NUM_FEATURES)
    y = get_label(df_train)
    cv_split = TimeSeriesSplit(n_splits=config.baselines.n_splits)

    results = {}
    for name, model in make_baseline_models(config.training.random_state).items():
        logger.info("Cross validating %s", name)
        scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv_split)
        fold_scores = {f"fold_{i + 1}": score for i, score in enumerate(scores)}
        results[name] = {**fold_scores, "mean": scores.mean(), "std": scores.std()}
    return pd.DataFrame(results).T.sort_values("mean", ascending=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    results = compare_baseline_models(config)
    output_path = config.paths.reports_dir / "fraud" / "baselines.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)
    logger.info("Baseline ROC AUC:\n%s", results.round(4))


if __name__ == "__main__":
    main()
