import pandas as pd


def add_transaction_time_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds clock and calendar features of the transaction time.
    Month and quarter are deliberately left out: with one year of data they are
    confounded with time and would not generalize to next year's transactions.
    :param df: transactions DataFrame with transactionDateTime
    :return: df with the time features added
    '''
    transaction_time = df["transactionDateTime"]
    df["transaction_hr"] = transaction_time.dt.hour
    df["transaction_min"] = transaction_time.dt.minute
    df["transaction_sec"] = transaction_time.dt.second
    # Sunday as 0 through Saturday as 6
    df["transaction_day_of_week"] = (transaction_time.dt.dayofweek + 1) % 7
    df["is_weekend"] = df["transaction_day_of_week"].isin([0, 6]).astype(int)
    df["is_night"] = df["transaction_hr"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    return df
