import numpy as np
import pandas as pd

from card_risk.config import UnderwritingConfig
from card_risk.underwriting import (
    UW_NUM_FEATURES,
    build_underwriting_features,
    convert_pd_to_underwriting_score,
    simulate_applicant_population,
)
from card_risk.underwriting.simulate import GROUND_TRUTH_COLUMNS

SMALL = UnderwritingConfig(n_applicants=5000, n_vintages=6, seed=3)


def test_simulation_is_reproducible():
    pd.testing.assert_frame_equal(simulate_applicant_population(SMALL), simulate_applicant_population(SMALL))


def test_outcomes_observed_only_for_booked_applicants():
    df = simulate_applicant_population(SMALL)
    assert df.loc[df["is_booked"] == 0, "defaulted_12m"].isna().all()
    assert df.loc[df["is_booked"] == 1, "defaulted_12m"].notna().all()
    assert 0 < df["true_pd_12m"].min() and df["true_pd_12m"].max() < 1


def test_risk_rises_as_fico_falls():
    df = simulate_applicant_population(SMALL)
    by_fico = df.groupby(pd.qcut(df["fico_score"], 5), observed=True)["true_pd_12m"].mean()
    assert by_fico.is_monotonic_decreasing


def test_model_features_exclude_ground_truth():
    features = build_underwriting_features(simulate_applicant_population(SMALL))
    assert not set(GROUND_TRUTH_COLUMNS) & set(features.columns)
    assert set(UW_NUM_FEATURES) <= set(features.columns)
    assert {"housing_status_RENT", "application_channel_DIGITAL"} <= set(features.columns)


def test_underwriting_score_is_probability_of_no_default():
    pd_12m = np.array([0.01, 0.10, 0.50])
    assert np.allclose(convert_pd_to_underwriting_score(pd_12m), [0.99, 0.90, 0.50])
