'''
Loss forecasting model: quarterly probability of default, loss given default and
balance behaviour as functions of the account's baseline risk, its age (seasoning) and
the economy.

The macro sensitivities are illustrative, chosen so a severe recession roughly
doubles to triples card losses at the peak, in line with past US downturns.
'''

import numpy as np
import pandas as pd

from card_risk.simulator.economy import STARTING_CONDITIONS

# Relative default intensity by quarter on book: card losses ramp up and peak in year two.
SEASONING_CURVE = np.array([0.55, 0.85, 1.15, 1.35, 1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90, 0.90])

# Log-odds change in quarterly PD per unit move of each macro driver from its baseline.
MACRO_LOSS_SENSITIVITIES = {
    "unemployment_rate_lag1q": 0.10,  # per point of unemployment above baseline
    "unemployment_change_4q": 0.10,  # per point rise over the last year
    "hpi_avg_2q_yoy_growth": -0.012,  # per point of home price growth
    "gdp_growth_avg_2q": -0.02,  # per point of GDP growth
    "cpi_inflation_lag1q": 0.02,  # per point of inflation (squeezes real income)
}

MACRO_BASELINES = {
    "unemployment_rate_lag1q": STARTING_CONDITIONS["unemployment_rate"],
    "unemployment_change_4q": 0.0,
    "hpi_avg_2q_yoy_growth": STARTING_CONDITIONS["hpi_growth"],
    "gdp_growth_avg_2q": STARTING_CONDITIONS["gdp_growth"],
    "cpi_inflation_lag1q": STARTING_CONDITIONS["cpi_inflation"],
}


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_seasoning_curve(n_quarters: int) -> np.ndarray:
    '''
    :param n_quarters: projection quarters
    :return: seasoning multipliers, extended flat past the end of SEASONING_CURVE
    '''
    if n_quarters <= len(SEASONING_CURVE):
        return SEASONING_CURVE[:n_quarters]
    return np.concatenate([SEASONING_CURVE, np.full(n_quarters - len(SEASONING_CURVE), SEASONING_CURVE[-1])])


def compute_macro_log_odds_shift(macro_path: pd.DataFrame) -> np.ndarray:
    '''
    :param macro_path: output of simulate_macro_path
    :return: log-odds shift of default for each quarter
    '''
    shift = np.zeros(len(macro_path))
    for driver, sensitivity in MACRO_LOSS_SENSITIVITIES.items():
        shift += sensitivity * (macro_path[driver].to_numpy() - MACRO_BASELINES[driver])
    return shift


def forecast_quarterly_default_probability(
    baseline_pd_12m: np.ndarray, macro_path: pd.DataFrame
) -> np.ndarray:
    '''
    Quarterly PD for each account and quarter.
    :param baseline_pd_12m: (n_accounts,) 12-month PD under baseline economic conditions
    :param macro_path: output of simulate_macro_path
    :return: (n_accounts, n_quarters) array
    '''
    n_quarters = len(macro_path)
    baseline_quarterly = 1 - (1 - np.asarray(baseline_pd_12m)) ** 0.25
    seasoned = np.clip(baseline_quarterly[:, None] * get_seasoning_curve(n_quarters)[None, :], 1e-6, 0.95)
    return sigmoid(logit(seasoned) + compute_macro_log_odds_shift(macro_path)[None, :])


def forecast_loss_given_default(macro_path: pd.DataFrame) -> np.ndarray:
    '''
    Card LGD is high and rises a little when unemployment is high (lower recoveries).
    :return: (n_quarters,) array
    '''
    excess_unemployment = (
        macro_path["unemployment_rate"].to_numpy() - STARTING_CONDITIONS["unemployment_rate"]
    )
    return np.clip(0.85 + 0.01 * excess_unemployment, 0.75, 0.95)


def forecast_revolving_balance_multiplier(macro_path: pd.DataFrame) -> np.ndarray:
    '''
    Customers lean on revolving credit more when unemployment rises.
    :return: (n_quarters,) array
    '''
    excess_unemployment = (
        macro_path["unemployment_rate"].to_numpy() - STARTING_CONDITIONS["unemployment_rate"]
    )
    return np.clip(1 + 0.04 * excess_unemployment, 0.8, 1.4)


def forecast_spend_multiplier(macro_path: pd.DataFrame) -> np.ndarray:
    '''
    Purchase volume moves with nominal GDP growth.
    :return: (n_quarters,) array
    '''
    nominal_growth = macro_path["gdp_growth"].to_numpy() + macro_path["cpi_inflation"].to_numpy()
    baseline = STARTING_CONDITIONS["gdp_growth"] + STARTING_CONDITIONS["cpi_inflation"]
    return np.clip(1 + 0.03 * (nominal_growth - baseline), 0.7, 1.3)


def forecast_cost_of_funds(macro_path: pd.DataFrame) -> np.ndarray:
    '''
    Annual funding rate: a spread over a policy rate that tracks inflation.
    :return: (n_quarters,) array of annual rates
    '''
    return np.clip(0.015 + 0.01 * macro_path["cpi_inflation"].to_numpy(), 0.01, 0.10)
