'''
Stage 2: build the transaction features into data/processed/features.parquet.

    fraud-build-features [--config configs/default.yaml]
'''

import argparse

import numpy as np
import pandas as pd

from card_risk.config import load_config
from card_risk.data_utils.load import load_transactions_data
from card_risk.features.account_features import add_account_features
from card_risk.features.categorical import add_clean_merchant_name
from card_risk.features.customer_history import (
    add_expanding_transaction_window_stats,
    add_merchant_familiarity_features,
    add_rate_features,
    add_ratio_rolling_overall_attributes,
    add_rolling_transaction_attributes,
    add_small_transaction_history,
    add_time_as_customer,
    add_trends_in_transactions,
    add_velocity_features,
)
from card_risk.features.time_features import add_transaction_time_features
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)

LABEL = "isFraud"
ID_COLUMNS = [
    "customerId",
    "transactionDateTime",
]

CAT_FEATURES = [
    "merchant_name_clean",
    "merchantCategoryCode",
    "acqCountry",
    "merchantCountryCode",
    "posEntryMode",
    "posConditionCode",
    "transactionType",
]


def _build_num_features() -> list[str]:
    windows = [2, 3, 5, 25]
    stats = ["mean", "min", "max"]
    return [
        # raw account and transaction fields
        "creditLimit",
        "availableMoney",
        "transactionAmount",
        "currentBalance",
        "cardPresent",
        "expirationDateKeyInMatch",
        # time
        "transaction_hr",
        "transaction_min",
        "transaction_sec",
        "transaction_day_of_week",
        "is_weekend",
        "is_night",
        # expanding history
        "previous_transactions_count",
        "previous_amounts_median",
        "previous_amounts_std",
        "previous_amounts_min",
        "previous_amounts_max",
        "previous_amounts_mean",
        # trends
        "previous_transaction_amt",
        "delta_most_recent_amt_prev_amt",
        "delta_most_recent_transaction_amt_mean_amt",
        "delta_most_recent_transaction_amt_median_amt",
        "delta_most_recent_transaction_amt_max_amt",
        "delta_most_recent_transaction_amt_min_amt",
        "days_since_previous_transaction",
        "days_between_transactions_median",
        "days_between_transactions_std",
        "days_between_transactions_min",
        "days_between_transactions_max",
        "days_between_transactions_mean",
        # rolling windows and ratios to history
        *[f"moving_{n}_transaction_{s}_{u}" for n in windows for s in stats for u in ["amt", "days"]],
        *[f"ratio_{n}_recent_to_overall_{s}_{u}" for n in windows for s in stats for u in ["amt", "days"]],
        # velocity
        "transactions_last_1h",
        "amount_last_1h",
        "transactions_last_24h",
        "amount_last_24h",
        # tenure and rates
        "time_as_customer",
        "days_since_account_open",
        "avg_transactions_per_day",
        "avg_amounts_per_day",
        "amount_velocity",
        # small transactions
        "is_amount_less_dollar",
        "num_previous_transactions_under_dollar",
        # merchants
        "is_new_merchant_for_customer",
        "previous_distinct_merchants",
        # account
        "credit_utilization",
        "transaction_use_of_credit_limit",
        "transaction_use_of_available_money",
        "CVV_match",
        "days_since_last_address_change",
        "days_until_expiration",
        "is_foreign_merchant",
    ]


# Note: cardCVV, enteredCVV, cardLast4Digits and accountNumber are identifiers, not
# quantities, so they are not features; CVV_match carries the signal.
NUM_FEATURES = _build_num_features()


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds every engineered feature to the cleaned transactions.
    :param df: cleaned transactions sorted by transactionDateTime (see data_utils)
    :return: DataFrame of ID_COLUMNS + NUM_FEATURES + CAT_FEATURES + LABEL, sorted by time
    '''
    df = df.sort_values("transactionDateTime", kind="mergesort").reset_index(drop=True)

    ##### Transaction Time Features #####
    df = add_transaction_time_features(df)

    ##### Merchant Names #####
    df = add_clean_merchant_name(df)

    ##### Expanding Window Transaction Statistics #####
    df = add_expanding_transaction_window_stats(df)
    df = add_trends_in_transactions(df)

    ##### Time As Customer #####
    df = add_time_as_customer(df)

    ##### Rolling Stats and Ratios to History #####
    df = add_rolling_transaction_attributes(df)
    df = add_ratio_rolling_overall_attributes(df)

    ##### Velocity, Merchant Familiarity, Small Transactions #####
    df = add_velocity_features(df)
    df = add_merchant_familiarity_features(df)
    df = add_small_transaction_history(df)
    df = add_rate_features(df)

    ##### Account Features #####
    df = add_account_features(df)

    # Clean up any zero divisions
    df[NUM_FEATURES] = df[NUM_FEATURES].replace([np.inf, -np.inf], np.nan).astype("float32")
    return df.loc[:, ID_COLUMNS + NUM_FEATURES + CAT_FEATURES + [LABEL]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = load_transactions_data(config)
    logger.info("Building features for %d transactions", len(df))
    df_features = build_transaction_features(df)

    output_path = config.paths.features_parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows x %d columns to %s", *df_features.shape, output_path)


if __name__ == "__main__":
    main()
