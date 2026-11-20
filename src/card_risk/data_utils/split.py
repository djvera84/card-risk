import pandas as pd


def out_of_time_split(
    df: pd.DataFrame, holdout_fraction: float, time_column: str = "transactionDateTime"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Splits by time: the most recent holdout_fraction of rows becomes the holdout set.
    A random split would let the model learn from transactions that happen after the
    ones it is scored on (e.g. a fraud wave in a given week) and overstate performance.
    :param df: DataFrame with a datetime column
    :param holdout_fraction: fraction of the most recent rows to hold out, between 0 and 1
    :param time_column: column to sort on
    :return: (earlier, later) DataFrames, both sorted by time_column
    '''
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    df = df.sort_values(time_column, kind="mergesort")
    cutoff = int(len(df) * (1 - holdout_fraction))
    return df.iloc[:cutoff], df.iloc[cutoff:]
