'''
Customer transaction-history features. All functions assume df is sorted by
transactionDateTime (as returned by data_utils) and keep that order.
'''

import numpy as np
import pandas as pd

CUSTOMER = "customerId"
ROLLING_WINDOWS = [2, 3, 5, 25]
EXPANDING_STATS = ["count", "median", "std", "min", "max", "mean"]


def get_days_between(later_times: pd.Series, earlier_times: pd.Series) -> pd.Series:
    '''
    Days between two datetime columns, rounded to the millisecond-ish.
    :param later_times: datetime Series
    :param earlier_times: datetime Series
    :return: float Series of days
    '''
    return ((later_times - earlier_times).dt.total_seconds() / (24.0 * 3600.0)).round(3)


def get_expanding_stat_of_previous_transactions(df: pd.DataFrame, column: str, stat: str) -> pd.Series:
    '''
    Expanding statistic over each customer's previous transactions (excluding the current one).
    :param df: transactions DataFrame sorted by time
    :param column: column to summarize
    :param stat: pandas expanding aggregation name, e.g. "mean"
    :return: Series aligned with df.index
    '''
    previous_values = df.groupby(CUSTOMER)[column].shift()
    stat_of_previous = previous_values.groupby(df[CUSTOMER]).expanding().agg(stat)
    return stat_of_previous.reset_index(level=0, drop=True)


def get_rolling_stat_of_recent_transactions(
    df: pd.DataFrame, column: str, window: int, stat: str
) -> pd.Series:
    '''
    Rolling statistic over each customer's last `window` transactions, including the current one
    (the current amount is known at authorization time, the label is not).
    :param df: transactions DataFrame sorted by time
    :param column: column to summarize
    :param window: number of transactions
    :param stat: pandas rolling aggregation name
    :return: Series aligned with df.index
    '''
    rolling_stat = df.groupby(CUSTOMER)[column].rolling(window, min_periods=1).agg(stat)
    return rolling_stat.reset_index(level=0, drop=True)


def add_expanding_transaction_window_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Summary statistics of all of a customer's previous transaction amounts.
    :param df: transactions DataFrame sorted by time
    :return: df with previous_amounts_* and previous_transactions_count
    '''
    for stat in EXPANDING_STATS:
        name = "previous_transactions_count" if stat == "count" else f"previous_amounts_{stat}"
        df[name] = get_expanding_stat_of_previous_transactions(df, "transactionAmount", stat)
    return df


def add_trends_in_transactions(df: pd.DataFrame) -> pd.DataFrame:
    '''
    How the current transaction compares with the customer's previous ones, and the
    customer's usual time between transactions.
    Must be run after add_expanding_transaction_window_stats.
    :param df: transactions DataFrame sorted by time
    :return: df with the trend features
    '''
    ##### Transaction Amount Trends #####
    df["previous_transaction_amt"] = df.groupby(CUSTOMER)["transactionAmount"].shift()
    df["delta_most_recent_amt_prev_amt"] = df["transactionAmount"] - df["previous_transaction_amt"]
    for stat in ["mean", "median", "max", "min"]:
        df[f"delta_most_recent_transaction_amt_{stat}_amt"] = (
            df["transactionAmount"] - df[f"previous_amounts_{stat}"]
        )

    ##### Transaction Time Trends #####
    previous_transaction_dt = df.groupby(CUSTOMER)["transactionDateTime"].shift()
    df["days_since_previous_transaction"] = get_days_between(
        df["transactionDateTime"], previous_transaction_dt
    )
    for stat in ["median", "std", "min", "max", "mean"]:
        df[f"days_between_transactions_{stat}"] = get_expanding_stat_of_previous_transactions(
            df, "days_since_previous_transaction", stat
        )
    return df


def add_time_as_customer(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Days from the customer's first transaction (in the data) to the current one, and days
    since the account was opened.
    :param df: transactions DataFrame
    :return: df with time_as_customer and days_since_account_open
    '''
    first_transaction_time = df.groupby(CUSTOMER)["transactionDateTime"].transform("min")
    df["time_as_customer"] = get_days_between(df["transactionDateTime"], first_transaction_time)
    df["days_since_account_open"] = get_days_between(df["transactionDateTime"], df["accountOpenDate"])
    return df


def add_rolling_transaction_attributes(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Rolling 2, 3, 5 and 25 transaction stats of amounts and of days between transactions.
    Must be run after add_trends_in_transactions.
    :param df: transactions DataFrame sorted by time
    :return: df with moving_* features
    '''
    for num in ROLLING_WINDOWS:
        for stat in ["mean", "min", "max"]:
            ##### Transaction Amounts #####
            df[f"moving_{num}_transaction_{stat}_amt"] = get_rolling_stat_of_recent_transactions(
                df, "transactionAmount", num, stat
            )
            ##### Time Between Transactions #####
            df[f"moving_{num}_transaction_{stat}_days"] = get_rolling_stat_of_recent_transactions(
                df, "days_since_previous_transaction", num, stat
            )
    return df


def add_ratio_rolling_overall_attributes(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Ratio of the recent rolling stats to the customer's overall history: is the customer
    behaving differently than usual?
    Must be run after add_rolling_transaction_attributes.
    :param df: transactions DataFrame
    :return: df with ratio_* features
    '''
    for num in ROLLING_WINDOWS:
        for stat in ["mean", "min", "max"]:
            ##### Transaction Amounts #####
            df[f"ratio_{num}_recent_to_overall_{stat}_amt"] = (
                df[f"moving_{num}_transaction_{stat}_amt"] / df[f"previous_amounts_{stat}"]
            )
            ##### Time Between Transactions #####
            df[f"ratio_{num}_recent_to_overall_{stat}_days"] = (
                df[f"moving_{num}_transaction_{stat}_days"] / df[f"days_between_transactions_{stat}"]
            )
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Count and dollar amount of the customer's transactions in the trailing 1 hour and
    24 hours (including the current one). Bursts of activity are a classic fraud signal.
    :param df: transactions DataFrame sorted by time
    :return: df with transactions_last_* and amount_last_* features
    '''
    indexed = df.set_index("transactionDateTime")
    for window in ["1h", "24h"]:
        rolling = indexed.groupby(CUSTOMER)["transactionAmount"].rolling(window)
        # groupby().rolling() returns rows grouped by customer; put them back in df's order.
        order = df.sort_values([CUSTOMER, "transactionDateTime"], kind="mergesort").index
        df.loc[order, f"transactions_last_{window}"] = rolling.count().to_numpy()
        df.loc[order, f"amount_last_{window}"] = rolling.sum().to_numpy()
    return df


def add_merchant_familiarity_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Has the customer transacted at this merchant before, and how many merchants have they used.
    Requires merchant_name_clean (see categorical.clean_merchant_name).
    :param df: transactions DataFrame sorted by time
    :return: df with is_new_merchant_for_customer and previous_distinct_merchants
    '''
    previous_visits = df.groupby([CUSTOMER, "merchant_name_clean"]).cumcount()
    df["is_new_merchant_for_customer"] = (previous_visits == 0).astype(int)
    df["previous_distinct_merchants"] = (
        df.groupby(CUSTOMER)["is_new_merchant_for_customer"].cumsum() - df["is_new_merchant_for_customer"]
    )
    return df


def add_small_transaction_history(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Fraud on an account often starts with small "test" charges. Flags transactions under a
    dollar and counts the customer's previous ones.
    :param df: transactions DataFrame sorted by time
    :return: df with is_amount_less_dollar and num_previous_transactions_under_dollar
    '''
    df["is_amount_less_dollar"] = (df["transactionAmount"] < 1).astype(int)
    df["num_previous_transactions_under_dollar"] = (
        df.groupby(CUSTOMER)["is_amount_less_dollar"].cumsum() - df["is_amount_less_dollar"]
    )
    return df


def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Per-day rates of activity. Must be run after add_time_as_customer.
    :param df: transactions DataFrame
    :return: df with avg_transactions_per_day, avg_amounts_per_day, amount_velocity
    '''
    days = df["time_as_customer"].replace(0, np.nan)
    df["avg_transactions_per_day"] = df["previous_transactions_count"] / days
    df["avg_amounts_per_day"] = df["previous_amounts_mean"] / days
    df["amount_velocity"] = df["delta_most_recent_amt_prev_amt"] / df["days_since_previous_transaction"]
    return df
