import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from card_risk.features.build import CAT_FEATURES, NUM_FEATURES
from card_risk.features.categorical import CategoricalFeatureEncoder


class FraudModel:
    '''
    Everything needed to score a transaction: the fitted categorical encoder, the
    XGBoost classifier, the feature order and the operating threshold.
    '''

    def __init__(
        self,
        encoder: CategoricalFeatureEncoder,
        classifier: xgb.XGBClassifier,
        threshold: float = 0.5,
        num_features: list[str] | None = None,
        metadata: dict | None = None,
    ):
        self.encoder = encoder
        self.classifier = classifier
        self.threshold = threshold
        self.num_features = num_features or NUM_FEATURES
        self.metadata = metadata or {}

    @staticmethod
    def make_model_matrix(df: pd.DataFrame, encoder: CategoricalFeatureEncoder, num_features: list[str]):
        '''
        Numeric features plus the one-hot encoded categoricals, in a fixed column order.
        :param df: feature DataFrame (output of build_transaction_features)
        :param encoder: fitted CategoricalFeatureEncoder
        :param num_features: numeric feature names
        :return: DataFrame ready for the classifier
        '''
        return pd.concat([df.loc[:, num_features], encoder.transform(df)], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.make_model_matrix(df, self.encoder, self.num_features)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        '''
        :param df: feature DataFrame
        :return: probability of fraud for each row
        '''
        return self.classifier.predict_proba(self.transform(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        '''
        :param df: feature DataFrame
        :return: 1 where the transaction should be flagged
        '''
        return (self.predict_proba(df) >= self.threshold).astype(int)

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(model_dir / "xgboost_model.json")
        joblib.dump(self.encoder, model_dir / "categorical_encoder.joblib")
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(
                {"threshold": self.threshold, "num_features": self.num_features, **self.metadata},
                f,
                indent=2,
                default=str,
            )

    @classmethod
    def load(cls, model_dir: Path) -> "FraudModel":
        model_dir = Path(model_dir)
        classifier = xgb.XGBClassifier()
        classifier.load_model(model_dir / "xgboost_model.json")
        encoder = joblib.load(model_dir / "categorical_encoder.joblib")
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
        threshold = metadata.pop("threshold")
        num_features = metadata.pop("num_features")
        return cls(encoder, classifier, threshold, num_features, metadata)


def make_categorical_encoder() -> CategoricalFeatureEncoder:
    return CategoricalFeatureEncoder(CAT_FEATURES)
