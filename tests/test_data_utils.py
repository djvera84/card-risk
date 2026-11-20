import pandas as pd
import pytest

from card_risk.data_utils import (
    EMPTY_COLUMNS,
    identify_multi_swipe_transactions,
    identify_reversed_transactions,
    out_of_time_split,
    summarize_duplicate_transactions,
    summarize_fields,
)


def test_fix_DataFrame_types_sorts_by_time_and_sets_types(clean_transactions):
    df = clean_transactions
    assert df["transactionDateTime"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(df["currentExpDate"])
    assert df["customerId"].dtype.kind == "i"
    assert df["isFraud"].tolist().count(1) == 1
    assert not set(EMPTY_COLUMNS) & set(df.columns)


def test_summarize_fields_counts_nulls_and_uniques(clean_transactions):
    summary = summarize_fields(clean_transactions)
    assert summary.loc["customerId", "unique_count"] == 2
    assert summary.loc["transactionAmount", "max"] == 60.0


def test_reversals_and_multi_swipes_are_identified(clean_transactions):
    df = clean_transactions
    reversals = identify_reversed_transactions(df)
    multi_swipes = identify_multi_swipe_transactions(df)
    assert reversals.sum() == 1
    # Only the second of the two identical AMC swipes a minute apart counts.
    assert multi_swipes.sum() == 1
    assert df.loc[multi_swipes, "transactionDateTime"].iloc[0] == pd.Timestamp("2016-01-01 12:01:00")

    summary = summarize_duplicate_transactions(df)
    assert summary.loc["multi_swipe", "total_amount"] == 20.0


def test_out_of_time_split_holds_out_the_most_recent_rows(clean_transactions):
    earlier, later = out_of_time_split(clean_transactions.sample(frac=1, random_state=1), 0.3)
    assert len(earlier) + len(later) == len(clean_transactions)
    assert earlier["transactionDateTime"].max() <= later["transactionDateTime"].min()


def test_out_of_time_split_rejects_bad_fraction(clean_transactions):
    with pytest.raises(ValueError):
        out_of_time_split(clean_transactions, 1.5)
