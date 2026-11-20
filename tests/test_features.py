import numpy as np
import pandas as pd

from card_risk.features import (
    CAT_FEATURES,
    NUM_FEATURES,
    CategoricalFeatureEncoder,
    build_transaction_features,
    clean_merchant_name,
)


def customer_rows(features, customer):
    return features[features["customerId"] == customer].reset_index(drop=True)


def test_clean_merchant_name_strips_location_number():
    assert clean_merchant_name("AMC #191138") == "AMC"
    assert clean_merchant_name("EZ Putt Putt #982199") == "EZ Putt Putt"
    assert clean_merchant_name("Netflix") == "Netflix"


def test_feature_table_has_expected_columns_and_no_infinities(clean_transactions):
    features = build_transaction_features(clean_transactions)
    assert set(NUM_FEATURES + CAT_FEATURES) <= set(features.columns)
    assert len(NUM_FEATURES) == len(set(NUM_FEATURES)), "duplicate feature names"
    assert not np.isinf(features[NUM_FEATURES].to_numpy()).any()
    assert features["transactionDateTime"].is_monotonic_increasing


def test_history_features_use_only_previous_transactions(clean_transactions):
    customer_1 = customer_rows(build_transaction_features(clean_transactions), 1)
    amounts = customer_1["transactionAmount"].tolist()
    assert customer_1.loc[0, "previous_transactions_count"] == 0
    assert np.isnan(customer_1.loc[0, "previous_amounts_mean"])
    for i in range(1, len(amounts)):
        assert customer_1.loc[i, "previous_transactions_count"] == i
        assert np.isclose(customer_1.loc[i, "previous_amounts_mean"], np.mean(amounts[:i]))
        assert np.isclose(customer_1.loc[i, "previous_amounts_max"], np.max(amounts[:i]))


def test_changing_a_later_transaction_does_not_change_earlier_features(clean_transactions):
    original = build_transaction_features(clean_transactions)
    modified_input = clean_transactions.copy()
    last = modified_input["transactionDateTime"].idxmax()
    modified_input.loc[last, "transactionAmount"] = 99_999.0
    modified = build_transaction_features(modified_input)
    earlier = original["transactionDateTime"] < clean_transactions["transactionDateTime"].max()
    pd.testing.assert_frame_equal(original.loc[earlier, NUM_FEATURES], modified.loc[earlier, NUM_FEATURES])


def test_small_transactions_and_velocity(clean_transactions):
    customer_2 = customer_rows(build_transaction_features(clean_transactions), 2)
    assert customer_2["is_amount_less_dollar"].tolist() == [1, 1]
    assert customer_2["num_previous_transactions_under_dollar"].tolist() == [0, 1]

    customer_1 = customer_rows(build_transaction_features(clean_transactions), 1)
    # 10:00, 12:00 and 12:01 on Jan 1 fall within 24 hours of 12:01.
    assert customer_1.loc[2, "transactions_last_24h"] == 3
    assert customer_1.loc[2, "transactions_last_1h"] == 2
    assert customer_1["is_new_merchant_for_customer"].tolist()[:3] == [1, 1, 0]


def test_cvv_mismatch_is_a_feature(clean_transactions):
    features = build_transaction_features(clean_transactions)
    fraud_row = features[features["isFraud"] == 1].iloc[0]
    assert fraud_row["CVV_match"] == 0


def test_categorical_encoder_handles_unseen_categories(clean_transactions):
    features = build_transaction_features(clean_transactions)
    encoder = CategoricalFeatureEncoder(CAT_FEATURES, min_frequency=1).fit(features)
    new = features.head(1).copy()
    new["merchant_name_clean"] = "Never Seen Before"
    encoded = encoder.transform(new)
    assert list(encoded.columns) == encoder.feature_names
    assert encoded.filter(like="merchant_name_clean_").sum(axis=1).iloc[0] == 0
