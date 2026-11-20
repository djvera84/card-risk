'''
Identification of duplicated transactions: reversals and multi-swipes.
'''

import pandas as pd

MULTI_SWIPE_WINDOW_SECONDS = 180


def identify_reversed_transactions(df: pd.DataFrame) -> pd.Series:
    '''
    Flags reversal transactions.
    :param df: transactions DataFrame with transactionType
    :return: boolean Series aligned with df
    '''
    return df["transactionType"].eq("REVERSAL")


def identify_multi_swipe_transactions(
    df: pd.DataFrame, window_seconds: int = MULTI_SWIPE_WINDOW_SECONDS
) -> pd.Series:
    '''
    Flags multi-swipes: a purchase by the same customer, at the same merchant, for the
    same amount, within window_seconds of the previous such purchase. The first
    transaction of a run is treated as normal and is not flagged.
    :param df: transactions DataFrame
    :param window_seconds: maximum seconds between swipes to count as a duplicate
    :return: boolean Series aligned with df
    '''
    purchases = df[df["transactionType"].ne("REVERSAL")]
    purchases = purchases.sort_values("transactionDateTime", kind="mergesort")
    seconds_since_same_swipe = (
        purchases.groupby(["customerId", "merchantName", "transactionAmount"])["transactionDateTime"]
        .diff()
        .dt.total_seconds()
    )
    is_multi_swipe = seconds_since_same_swipe.le(window_seconds)
    return is_multi_swipe.reindex(df.index, fill_value=False)


def summarize_duplicate_transactions(
    df: pd.DataFrame, window_seconds: int = MULTI_SWIPE_WINDOW_SECONDS
) -> pd.DataFrame:
    '''
    Count and dollar total of reversed and multi-swipe transactions.
    :param df: transactions DataFrame
    :param window_seconds: multi-swipe window, see identify_multi_swipe_transactions
    :return: DataFrame indexed by duplicate type with count, total_amount and fraud_rate
    '''
    rows = {}
    for name, mask in {
        "reversal": identify_reversed_transactions(df),
        "multi_swipe": identify_multi_swipe_transactions(df, window_seconds),
    }.items():
        rows[name] = {
            "count": int(mask.sum()),
            "total_amount": float(df.loc[mask, "transactionAmount"].sum()),
            "fraud_rate": float(df.loc[mask, "isFraud"].mean()),
        }
    return pd.DataFrame(rows).T
