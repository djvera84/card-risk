'''
Stage 3b (optional): randomized hyperparameter search with time-series cross validation.
Writes artifacts/best_params.json, which fraud-train picks up.

    fraud-tune [--config configs/default.yaml]
'''

import argparse
import json

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from card_risk.config import load_config
from card_risk.features.build import NUM_FEATURES
from card_risk.fraud.data import get_label, load_fraud_model_splits
from card_risk.fraud.model import FraudModel, make_categorical_encoder
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)


def search_fraud_model_hyperparameters(config) -> tuple[dict, pd.DataFrame]:
    '''
    Randomized search over config.tuning.param_space, scored by ROC AUC on
    TimeSeriesSplit folds of the training data (the test set is never touched).
    A fixed number of trees is used here; fraud-train adds early stopping.
    :param config: pipeline Config
    :return: (best params, cv results DataFrame)
    '''
    df_train = load_fraud_model_splits(config)["train_full"]
    encoder = make_categorical_encoder().fit(df_train)
    X = FraudModel.make_model_matrix(df_train, encoder, NUM_FEATURES)
    y = get_label(df_train)

    random_search = RandomizedSearchCV(
        xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=config.tuning.n_estimators,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.training.random_state,
        ),
        param_distributions=config.tuning.param_space,
        n_iter=config.tuning.n_iter,
        cv=TimeSeriesSplit(n_splits=config.tuning.n_splits),
        scoring="roc_auc",
        random_state=config.training.random_state,
        refit=False,
        verbose=1,
    )
    random_search.fit(X, y)
    cv_results = pd.DataFrame(random_search.cv_results_).sort_values("rank_test_score")
    return random_search.best_params_, cv_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    best_params, cv_results = search_fraud_model_hyperparameters(config)
    config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(config.paths.best_params_json, "w") as f:
        json.dump(best_params, f, indent=2)
    reports_dir = config.paths.reports_dir / "fraud"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(reports_dir / "tuning_results.csv", index=False)
    logger.info("Best params %s (cv ROC AUC %.4f)", best_params, cv_results["mean_test_score"].iloc[0])


if __name__ == "__main__":
    main()
