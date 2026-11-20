import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from card_risk.underwriting.features import UW_MONOTONE_CONSTRAINTS, build_underwriting_features


def convert_pd_to_underwriting_score(pd_12m) -> np.ndarray:
    '''
    The underwriting score: the model's probability of NOT defaulting in the next
    12 months (higher is safer). Credit policy bands and cutoffs are set on this score.
    :param pd_12m: array of 12-month probabilities of default
    :return: array of P(no default) in [0, 1]
    '''
    return 1.0 - np.asarray(pd_12m, dtype=float)


class UnderwritingModel:
    '''
    Monotone-constrained XGBoost probability-of-default model on application data.
    '''

    def __init__(self, classifier: xgb.XGBClassifier | None = None, metadata: dict | None = None):
        self.classifier = classifier
        self.metadata = metadata or {}

    @staticmethod
    def make_classifier(
        params: dict, n_estimators: int, early_stopping_rounds: int, feature_names: list[str]
    ):
        constraints = tuple(UW_MONOTONE_CONSTRAINTS.get(name, 0) for name in feature_names)
        return xgb.XGBClassifier(
            **{
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 20,
                **params,
            },
            objective="binary:logistic",
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="auc",
            monotone_constraints=constraints,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        '''
        :param df: applications DataFrame
        :return: predicted 12-month probability of default
        '''
        return self.classifier.predict_proba(build_underwriting_features(df))[:, 1]

    def predict_underwriting_score(self, df: pd.DataFrame) -> np.ndarray:
        '''
        :param df: applications DataFrame
        :return: P(no default in 12 months)
        '''
        return convert_pd_to_underwriting_score(self.predict_pd(df))

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(model_dir / "xgboost_model.json")
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    @classmethod
    def load(cls, model_dir: Path) -> "UnderwritingModel":
        model_dir = Path(model_dir)
        classifier = xgb.XGBClassifier()
        classifier.load_model(model_dir / "xgboost_model.json")
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
        return cls(classifier, metadata)
