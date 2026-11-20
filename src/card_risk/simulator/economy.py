'''
Economic scenarios and macro paths.

Quarter 1 is the current economy. After it, one scenario is drawn and the macro
variables follow that scenario's path plus autocorrelated noise.
'''

from dataclasses import dataclass

import numpy as np
import pandas as pd

STARTING_CONDITIONS = {
    "unemployment_rate": 4.0,  # percent
    "gdp_growth": 2.2,  # real GDP, annualized percent
    "cpi_inflation": 2.6,  # year over year percent
    "hpi_growth": 4.0,  # Case-Shiller year over year percent
    "case_shiller_hpi": 320.0,  # index level
}

# Quarterly noise around the scenario path: (AR(1) coefficient, innovation sd)
MACRO_NOISE = {
    "unemployment_rate": (0.7, 0.15),
    "gdp_growth": (0.5, 0.60),
    "cpi_inflation": (0.7, 0.30),
    "hpi_growth": (0.7, 1.00),
}

PATH_VARIABLES = [
    "unemployment_rate",
    "gdp_growth",
    "cpi_inflation",
    "hpi_growth",
]

# Quarters of pre-history (at starting conditions) needed for the lag variables.
PRE_HISTORY_QUARTERS = 6


@dataclass(frozen=True)
class EconomicScenario:
    '''
    A scenario moves each macro variable from its starting value by `change`.
    shape "persistent" ramps to the change and stays; ("hump", q) peaks q quarters
    after the scenario starts and then recovers.
    '''

    name: str
    probability: float
    changes: dict
    shapes: dict


ECONOMIC_SCENARIOS = {
    "boom": EconomicScenario(
        "boom",
        0.15,
        changes={"unemployment_rate": -0.8, "gdp_growth": 1.6, "cpi_inflation": 0.8, "hpi_growth": 5.0},
        shapes={v: "persistent" for v in PATH_VARIABLES},
    ),
    "benign": EconomicScenario(
        "benign",
        0.50,
        changes={"unemployment_rate": 0.0, "gdp_growth": -0.2, "cpi_inflation": -0.1, "hpi_growth": 0.0},
        shapes={v: "persistent" for v in PATH_VARIABLES},
    ),
    "minor_recession": EconomicScenario(
        "minor_recession",
        0.25,
        changes={"unemployment_rate": 1.8, "gdp_growth": -2.7, "cpi_inflation": -0.8, "hpi_growth": -5.0},
        shapes={
            "unemployment_rate": ("hump", 4),
            "gdp_growth": ("hump", 2),
            "cpi_inflation": ("hump", 4),
            "hpi_growth": ("hump", 4),
        },
    ),
    "recession": EconomicScenario(
        "recession",
        0.10,
        changes={"unemployment_rate": 5.0, "gdp_growth": -6.0, "cpi_inflation": -1.8, "hpi_growth": -14.0},
        shapes={
            "unemployment_rate": ("hump", 6),
            "gdp_growth": ("hump", 3),
            "cpi_inflation": ("hump", 5),
            "hpi_growth": ("hump", 5),
        },
    ),
}


def get_scenario_shape(shape, quarters_since_start: np.ndarray) -> np.ndarray:
    '''
    Fraction of the scenario's change reached after t quarters.
    :param shape: "persistent" or ("hump", peak_quarter)
    :param quarters_since_start: array of t >= 0
    :return: array in [0, 1]
    '''
    t = np.asarray(quarters_since_start, dtype=float)
    if shape == "persistent":
        return 1 - np.exp(-t / 2)
    _, peak = shape
    x = t / peak
    return x * np.exp(1 - x)


def draw_economic_scenario(rng) -> str:
    '''
    :param rng: numpy Generator
    :return: scenario name, drawn with the scenario probabilities
    '''
    names = list(ECONOMIC_SCENARIOS)
    probabilities = np.array([ECONOMIC_SCENARIOS[n].probability for n in names])
    return str(rng.choice(names, p=probabilities / probabilities.sum()))


def add_lagged_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Lagged and smoothed macro variables used by the loss model, e.g. the year over year
    growth of the average Case-Shiller index over the last 2 quarters.
    :param df: quarterly macro path including the pre-history quarters
    :return: df with the lag features
    '''
    df["unemployment_rate_lag1q"] = df["unemployment_rate"].shift(1)
    df["unemployment_change_4q"] = df["unemployment_rate"] - df["unemployment_rate"].shift(4)
    df["case_shiller_hpi_avg_2q"] = df["case_shiller_hpi"].rolling(2).mean()
    df["hpi_avg_2q_yoy_growth"] = 100 * (
        df["case_shiller_hpi_avg_2q"] / df["case_shiller_hpi_avg_2q"].shift(4) - 1
    )
    df["gdp_growth_avg_2q"] = df["gdp_growth"].rolling(2).mean()
    df["cpi_inflation_lag1q"] = df["cpi_inflation"].shift(1)
    return df


def simulate_macro_path(
    scenario_name: str, n_quarters: int, rng=None, with_noise: bool = True
) -> pd.DataFrame:
    '''
    Quarterly macro path: quarter 1 at starting conditions, then the scenario.
    :param scenario_name: key of ECONOMIC_SCENARIOS
    :param n_quarters: projection quarters
    :param rng: numpy Generator (required if with_noise)
    :param with_noise: add AR(1) noise; False gives the scenario's expected path
    :return: DataFrame indexed by quarter 1..n_quarters with levels and lag features
    '''
    scenario = ECONOMIC_SCENARIOS[scenario_name]
    total = PRE_HISTORY_QUARTERS + n_quarters
    quarter = np.arange(1 - PRE_HISTORY_QUARTERS, n_quarters + 1)
    quarters_since_start = np.clip(quarter - 1, 0, None)

    path = {}
    for variable in PATH_VARIABLES:
        shape = get_scenario_shape(scenario.shapes[variable], quarters_since_start)
        values = STARTING_CONDITIONS[variable] + scenario.changes[variable] * shape
        if with_noise:
            rho, sd = MACRO_NOISE[variable]
            noise = np.zeros(total)
            for i in range(PRE_HISTORY_QUARTERS + 1, total):
                noise[i] = rho * noise[i - 1] + rng.normal(0, sd)
            values = values + noise
        path[variable] = values

    df = pd.DataFrame(path, index=pd.Index(quarter, name="quarter"))
    df["unemployment_rate"] = df["unemployment_rate"].clip(lower=2.0)
    quarterly_hpi_growth = (1 + df["hpi_growth"] / 100) ** 0.25
    df["case_shiller_hpi"] = STARTING_CONDITIONS["case_shiller_hpi"] * (
        quarterly_hpi_growth.cumprod() / quarterly_hpi_growth.iloc[PRE_HISTORY_QUARTERS]
    )
    df = add_lagged_macro_features(df)
    df["scenario"] = scenario_name
    return df.loc[1:]
