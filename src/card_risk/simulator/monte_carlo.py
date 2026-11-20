'''
The two Monte Carlo modes of the credit policy simulator.
'''

import numpy as np
import pandas as pd

from card_risk.simulator.economy import ECONOMIC_SCENARIOS, draw_economic_scenario, simulate_macro_path
from card_risk.simulator.policy import CreditPolicy, CreditPolicySimulator

CREDIT_LINE_GRID = np.array(
    [500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 9000, 10000, 12500, 15000]
)


def simulate_policy_across_economic_scenarios(
    simulator: CreditPolicySimulator, policy: CreditPolicy, n_paths: int, n_quarters: int, seed: int = 0
) -> pd.DataFrame:
    '''
    Mode 1, fixed policy and random economies: draws a scenario and a noisy macro path
    n_paths times and evaluates the same policy on each.
    :param simulator: CreditPolicySimulator
    :param policy: CreditPolicy
    :param n_paths: number of economic paths
    :param n_quarters: projection quarters
    :param seed: random seed
    :return: DataFrame with one row per path: scenario, peak unemployment, metrics
    '''
    rng = np.random.default_rng(seed)
    rows = []
    for path_id in range(n_paths):
        scenario = draw_economic_scenario(rng)
        macro_path = simulate_macro_path(scenario, n_quarters, rng)
        metrics = simulator.evaluate(policy, macro_path)
        rows.append(
            {
                "path_id": path_id,
                "scenario": scenario,
                "peak_unemployment_rate": macro_path["unemployment_rate"].max(),
                "min_hpi_avg_2q_yoy_growth": macro_path["hpi_avg_2q_yoy_growth"].min(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def sample_random_credit_policies(
    n_policies: int,
    n_bands: int,
    rng,
    threshold_range: tuple = (0.70, 0.97),
    apr_range: tuple = (0.17, 0.32),
) -> list[CreditPolicy]:
    '''
    Random but sensible policies: lines never decrease and APRs never increase as the
    score band improves.
    :param n_policies: number of policies
    :param n_bands: score bands per policy
    :param rng: numpy Generator
    :param threshold_range: range of approve thresholds on P(no default)
    :param apr_range: range of APRs
    :return: list of CreditPolicy
    '''
    policies = []
    for _ in range(n_policies):
        lines = np.sort(rng.choice(CREDIT_LINE_GRID, size=n_bands))
        aprs = np.sort(np.round(rng.uniform(*apr_range, size=n_bands), 3))[::-1]
        threshold = float(np.round(rng.uniform(*threshold_range), 3))
        policies.append(CreditPolicy(threshold, lines.tolist(), aprs.tolist()))
    return policies


def simulate_policies_in_fixed_economy(
    simulator: CreditPolicySimulator,
    policies: list[CreditPolicy],
    scenario_name: str,
    n_quarters: int,
    seed: int = 0,
    with_noise: bool = False,
) -> pd.DataFrame:
    '''
    Mode 2, fixed economy and random policies: evaluates every policy on the same macro path.
    :param simulator: CreditPolicySimulator
    :param policies: list of CreditPolicy (see sample_random_credit_policies)
    :param scenario_name: key of ECONOMIC_SCENARIOS
    :param n_quarters: projection quarters
    :param seed: random seed for the path noise
    :param with_noise: False uses the scenario's expected path
    :return: DataFrame with one row per policy: policy parameters and metrics
    '''
    if scenario_name not in ECONOMIC_SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_name}; choose from {list(ECONOMIC_SCENARIOS)}")
    macro_path = simulate_macro_path(
        scenario_name, n_quarters, np.random.default_rng(seed), with_noise=with_noise
    )
    rows = []
    for policy_id, policy in enumerate(policies):
        metrics = simulator.evaluate(policy, macro_path)
        rows.append(
            {
                "policy_id": policy_id,
                "scenario": scenario_name,
                "approve_threshold": policy.approve_threshold,
                "credit_lines_by_band": policy.credit_lines_by_band,
                "apr_by_band": policy.apr_by_band,
                **metrics,
            }
        )
    return pd.DataFrame(rows)
