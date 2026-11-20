'''
Feature engineering for the transaction fraud model.

Every customer-history feature uses only transactions strictly before (or, for the
rolling windows, up to and including) the current one, so nothing leaks from the future.
'''

from card_risk.features.build import (
    CAT_FEATURES,
    NUM_FEATURES,
    build_transaction_features,
)
from card_risk.features.categorical import CategoricalFeatureEncoder, clean_merchant_name

__all__ = [
    "CAT_FEATURES",
    "NUM_FEATURES",
    "CategoricalFeatureEncoder",
    "build_transaction_features",
    "clean_merchant_name",
]
