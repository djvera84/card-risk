import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

MISSING_CATEGORY = "OTHER"
LOCATION_NUMBER_REGEX = re.compile(r"\s*#.*$")
# XGBoost rejects feature names containing these characters.
UNSAFE_FEATURE_NAME_CHARS = re.compile(r"[\[\]<>,]")


def clean_merchant_name(merchant: str) -> str:
    '''
    Trims the location number from a merchant name, e.g. "AMC #191138" -> "AMC",
    so the many locations of a chain share one category.
    :param merchant: raw merchant name
    :return: merchant name without the location number
    '''
    return LOCATION_NUMBER_REGEX.sub("", merchant)


def add_clean_merchant_name(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds merchant_name_clean (see clean_merchant_name).
    :param df: transactions DataFrame with merchantName
    :return: df with merchant_name_clean
    '''
    df["merchant_name_clean"] = df["merchantName"].map(clean_merchant_name)
    return df


class CategoricalFeatureEncoder:
    '''
    One-hot encodes the categorical features. Fit on the training set only; categories
    seen fewer than min_frequency times, and categories not seen in training, go to an
    "infrequent" bucket so the encoder is safe to use on new data.
    '''

    def __init__(self, cat_features: list[str], min_frequency: int = 50):
        self.cat_features = cat_features
        self.encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
            dtype=np.uint8,
        )

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.cat_features].astype(object).fillna(MISSING_CATEGORY)

    def fit(self, df: pd.DataFrame) -> "CategoricalFeatureEncoder":
        '''
        :param df: training DataFrame containing cat_features
        :return: self
        '''
        self.encoder.fit(self._fill_missing(df))
        return self

    @property
    def feature_names(self) -> list[str]:
        names = self.encoder.get_feature_names_out(self.cat_features)
        return [UNSAFE_FEATURE_NAME_CHARS.sub("_", name) for name in names]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        :param df: DataFrame containing cat_features
        :return: one-hot DataFrame with the same index as df
        '''
        encoded = self.encoder.transform(self._fill_missing(df))
        return pd.DataFrame(encoded, columns=self.feature_names, index=df.index)
