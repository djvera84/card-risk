import numpy as np
import pandas as pd
import pytest

from card_risk.simulator import (
    ECONOMIC_SCENARIOS,
    CardEconomics,
    CreditPolicy,
    CreditPolicySimulator,
    sample_random_credit_policies,
    simulate_macro_path,
    simulate_policies_in_fixed_economy,
    simulate_policy_across_economic_scenarios,
)
from card_risk.simulator.loss_model import forecast_quarterly_default_probability

N_QUARTERS = 12
POLICY = CreditPolicy(0.9, [500, 1000, 2000, 4000, 8000], [0.29, 0.27, 0.25, 0.23, 0.20])


@pytest.fixture
def simulator(rng):
    n = 3000
    true_pd = np.clip(rng.beta(1.2, 12, n), 0.002, 0.9)
    predicted_pd = np.clip(true_pd * rng.lognormal(0, 0.3, n), 0.001, 0.99)
    population = pd.DataFrame(
        {
            "true_pd_12m": true_pd,
            "underwriting_score": 1 - predicted_pd,
            "revolving_balance_demand": rng.lognormal(np.log(3000), 0.6, n),
            "monthly_spend_demand": rng.lognormal(np.log(500), 0.5, n),
        }
    )
    return CreditPolicySimulator(population, CardEconomics())


@pytest.mark.parametrize("scenario", list(ECONOMIC_SCENARIOS))
def test_macro_paths_are_complete(scenario, rng):
    path = simulate_macro_path(scenario, N_QUARTERS, rng)
    assert len(path) == N_QUARTERS
    assert not path.drop(columns="scenario").isna().any().any()
    # Quarter 1 is always today's economy.
    assert path.loc[1, "unemployment_rate"] == pytest.approx(4.0)


def test_recession_raises_default_probabilities():
    baseline_pd = np.array([0.02, 0.10])
    benign = forecast_quarterly_default_probability(
        baseline_pd, simulate_macro_path("benign", N_QUARTERS, with_noise=False)
    )
    recession = forecast_quarterly_default_probability(
        baseline_pd, simulate_macro_path("recession", N_QUARTERS, with_noise=False)
    )
    assert (recession[:, 1:] > benign[:, 1:]).all()


def test_policy_bands_and_threshold(simulator):
    decision = simulator.apply_credit_policy(POLICY)
    assert set(np.unique(decision["band"])) == {1, 2, 3, 4, 5}
    assert decision["approved"].mean() == pytest.approx((simulator.scores >= 0.9).mean())
    stricter = simulator.evaluate(
        CreditPolicy(0.97, POLICY.credit_lines_by_band, POLICY.apr_by_band),
        simulate_macro_path("benign", N_QUARTERS, with_noise=False),
    )
    looser = simulator.evaluate(POLICY, simulate_macro_path("benign", N_QUARTERS, with_noise=False))
    assert stricter["approval_rate"] < looser["approval_rate"]
    assert stricter["net_charge_off_rate"] < looser["net_charge_off_rate"]


def test_policy_needs_one_line_and_apr_per_band():
    with pytest.raises(ValueError):
        CreditPolicy(0.9, [1000, 2000], [0.25])


def test_recession_npv_below_benign(simulator):
    benign = simulator.evaluate(POLICY, simulate_macro_path("benign", N_QUARTERS, with_noise=False))
    recession = simulator.evaluate(POLICY, simulate_macro_path("recession", N_QUARTERS, with_noise=False))
    assert recession["npv"] < benign["npv"]
    assert recession["net_charge_off_rate"] > benign["net_charge_off_rate"]


def test_monte_carlo_modes(simulator, rng):
    across = simulate_policy_across_economic_scenarios(
        simulator, POLICY, n_paths=20, n_quarters=N_QUARTERS, seed=1
    )
    assert len(across) == 20 and across["npv"].notna().all()

    policies = sample_random_credit_policies(15, 5, rng)
    for policy in policies:
        assert np.all(np.diff(policy.credit_lines_by_band) >= 0)
        assert np.all(np.diff(policy.apr_by_band) <= 0)
    fixed = simulate_policies_in_fixed_economy(simulator, policies, "recession", N_QUARTERS)
    assert len(fixed) == 15 and (fixed["scenario"] == "recession").all()
