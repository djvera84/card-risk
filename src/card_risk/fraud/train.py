'''
Stage 4: train the final fraud model and save it to artifacts/model/.

    fraud-train [--config configs/default.yaml]

Uses artifacts/best_params.json from fraud-tune if it exists, otherwise the params in the config.
'''

import argparse
import json

import xgboost as xgb

from card_risk.config import Config, load_config
from card_risk.evaluation.metrics import (
    compute_binary_classification_metrics,
    find_threshold_for_target_recall,
)
from card_risk.features.build import NUM_FEATURES
from card_risk.fraud.data import get_label, load_fraud_model_splits
from card_risk.fraud.model import FraudModel, make_categorical_encoder
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)


def get_model_params(config: Config) -> dict:
    '''
    Tuned params from artifacts/best_params.json if present, else the config's params.
    :param config: pipeline Config
    :return: XGBoost params
    '''
    if config.paths.best_params_json.exists():
        with open(config.paths.best_params_json) as f:
            logger.info("Using tuned params from %s", config.paths.best_params_json)
            return json.load(f)
    return dict(config.training.params)


def train_fraud_model(config: Config) -> FraudModel:
    '''
    Fits the encoder on the training split, trains XGBoost with early stopping on the
    validation split and picks the threshold on validation for the target recall.
    :param config: pipeline Config
    :return: fitted FraudModel
    '''
    splits = load_fraud_model_splits(config)
    df_train, df_valid = splits["train"], splits["valid"]
    logger.info(
        "Train %d rows (%s to %s), valid %d rows",
        len(df_train),
        df_train["transactionDateTime"].min(),
        df_train["transactionDateTime"].max(),
        len(df_valid),
    )

    ##### Encode Categoricals (fit on train only) #####
    encoder = make_categorical_encoder().fit(df_train)
    X_train = FraudModel.make_model_matrix(df_train, encoder, NUM_FEATURES)
    X_valid = FraudModel.make_model_matrix(df_valid, encoder, NUM_FEATURES)
    y_train, y_valid = get_label(df_train), get_label(df_valid)
    logger.info("Model matrix has %d features", X_train.shape[1])

    ##### Train XGBoost with Early Stopping #####
    params = get_model_params(config)
    classifier = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        n_estimators=config.training.n_estimators,
        early_stopping_rounds=config.training.early_stopping_rounds,
        eval_metric="auc",
        tree_method="hist",
        random_state=config.training.random_state,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    logger.info("Best iteration %d", classifier.best_iteration)

    ##### Choose the Operating Threshold on Validation #####
    valid_scores = classifier.predict_proba(X_valid)[:, 1]
    threshold = find_threshold_for_target_recall(y_valid, valid_scores, config.training.target_recall)
    valid_metrics = compute_binary_classification_metrics(y_valid, valid_scores, threshold)
    logger.info("Validation metrics: %s", json.dumps(valid_metrics, indent=2))

    metadata = {
        "params": params,
        "best_iteration": classifier.best_iteration,
        "train_period": [df_train["transactionDateTime"].min(), df_train["transactionDateTime"].max()],
        "valid_period": [df_valid["transactionDateTime"].min(), df_valid["transactionDateTime"].max()],
        "target_recall": config.training.target_recall,
        "valid_metrics": valid_metrics,
    }
    return FraudModel(encoder, classifier, threshold, metadata=metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model = train_fraud_model(config)
    model.save(config.paths.model_dir)
    logger.info("Saved model to %s", config.paths.model_dir)


if __name__ == "__main__":
    main()
