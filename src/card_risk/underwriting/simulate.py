'''
Simulated credit card applicant population with a known ground-truth default process.

Every applicant gets bureau and application attributes driven by a latent
creditworthiness factor. The true 12-month probability of default (under baseline
economic conditions) is a fixed function of those attributes plus an unobserved
component. A historical underwriting rule decides who was booked; only booked
applicants have an observed 12-month default outcome.

    underwriting-simulate [--config configs/default.yaml]
'''

import argparse

import numpy as np
import pandas as pd

from card_risk.config import UnderwritingConfig, load_config
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)

HOUSING_STATUSES = ["RENT", "MORTGAGE", "OWN"]
APPLICATION_CHANNELS = ["DIGITAL", "DIRECT_MAIL", "BRANCH"]

# Columns that exist only because the data is simulated. Models must never use them.
GROUND_TRUTH_COLUMNS = [
    "latent_creditworthiness",
    "true_pd_12m",
    "revolving_balance_demand",
    "monthly_spend_demand",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def add_application_dates(df: pd.DataFrame, config: UnderwritingConfig, rng) -> pd.DataFrame:
    '''
    Spreads applications evenly over config.n_vintages monthly vintages.
    :param df: applicant DataFrame
    :param config: UnderwritingConfig
    :param rng: numpy Generator
    :return: df with vintage (1-indexed month) and application_date
    '''
    df["vintage"] = rng.integers(1, config.n_vintages + 1, size=len(df))
    first = pd.Timestamp(config.first_vintage)
    df["application_date"] = [first + pd.DateOffset(months=int(v) - 1) for v in df["vintage"]]
    df["application_date"] = pd.to_datetime(df["application_date"]) + pd.to_timedelta(
        rng.integers(0, 28, size=len(df)), unit="D"
    )
    return df


def add_bureau_attributes(df: pd.DataFrame, rng) -> pd.DataFrame:
    '''
    Credit bureau attributes, all correlated through latent_creditworthiness.
    :param df: applicant DataFrame with latent_creditworthiness
    :param rng: numpy Generator
    :return: df with bureau attributes
    '''
    z = df["latent_creditworthiness"].to_numpy()
    n = len(df)
    df["fico_score"] = np.clip(np.round(690 + 55 * z + rng.normal(0, 25, n)), 300, 850).astype(int)
    df["revolving_utilization"] = np.clip(sigmoid(-0.6 - 0.9 * z + rng.normal(0, 0.8, n)), 0, 1.2).round(3)
    df["num_credit_inquiries_6m"] = rng.poisson(np.exp(0.2 - 0.35 * z))
    df["num_delinquencies_24m"] = rng.poisson(np.exp(-1.7 - 0.9 * z))
    df["months_since_oldest_tradeline"] = np.clip(rng.normal(150 + 45 * z, 70, n), 0, 600).round().astype(int)
    df["num_open_tradelines"] = rng.poisson(np.clip(6 + 1.5 * z, 1, None))
    return df


def add_application_attributes(df: pd.DataFrame, rng) -> pd.DataFrame:
    '''
    Self-reported application attributes: income, debt-to-income, employment, housing,
    channel and requested line.
    :param df: applicant DataFrame with latent_creditworthiness
    :param rng: numpy Generator
    :return: df with application attributes
    '''
    z = df["latent_creditworthiness"].to_numpy()
    n = len(df)
    df["annual_income"] = np.clip(
        np.exp(np.log(55_000) + 0.25 * z + rng.normal(0, 0.55, n)), 12_000, 750_000
    ).round(-2)
    df["debt_to_income"] = np.clip(0.30 - 0.06 * z + rng.normal(0, 0.10, n), 0, 0.9).round(3)
    df["employment_years"] = np.clip(rng.exponential(5, n) + z, 0, 45).round(1)

    mortgage_odds = np.exp(-0.6 + 0.6 * z + 0.8 * np.log(df["annual_income"] / 55_000))
    p_mortgage = mortgage_odds / (1 + mortgage_odds + 0.35)
    p_own = 0.35 / (1 + mortgage_odds + 0.35)
    u = rng.random(n)
    df["housing_status"] = np.where(
        u < p_mortgage, "MORTGAGE", np.where(u < p_mortgage + p_own, "OWN", "RENT")
    )
    df["application_channel"] = rng.choice(APPLICATION_CHANNELS, size=n, p=[0.6, 0.3, 0.1])

    requested = df["annual_income"] * 0.12 * rng.uniform(0.4, 1.6, n)
    df["requested_credit_limit"] = np.clip(np.round(requested / 500) * 500, 500, 30_000)
    df["total_revolving_balance"] = (
        df["revolving_utilization"] * df["annual_income"] * rng.uniform(0.1, 0.4, n)
    ).round(-1)
    return df


def add_true_default_probability(df: pd.DataFrame, rng) -> pd.DataFrame:
    '''
    Ground-truth 12-month probability of default under baseline economic conditions.
    Includes non-linear terms and an unobserved component the models cannot see.
    :param df: applicant DataFrame with bureau and application attributes
    :param rng: numpy Generator
    :return: df with true_pd_12m
    '''
    log_odds = (
        -3.3
        - 0.011 * (df["fico_score"] - 690)
        + 1.8 * (df["debt_to_income"] - 0.30)
        + 1.1 * (df["revolving_utilization"] - 0.35)
        + 0.15 * df["num_credit_inquiries_6m"]
        + 0.40 * df["num_delinquencies_24m"]
        - 0.30 * np.log(df["annual_income"] / 55_000)
        - 0.002 * (df["months_since_oldest_tradeline"] - 150)
        - 0.03 * df["employment_years"]
        + 0.20 * (df["housing_status"] == "RENT")
        + 0.25 * (df["application_channel"] == "DIGITAL")
        ##### Non-linear Effects #####
        + 0.70 * (df["revolving_utilization"] > 0.90)
        + 0.60 * ((df["debt_to_income"] > 0.45) & (df["fico_score"] < 640))
        + 0.40 * (df["num_credit_inquiries_6m"] >= 4)
        ##### Unobserved Heterogeneity #####
        + rng.normal(0, 0.45, len(df))
    )
    df["true_pd_12m"] = sigmoid(log_odds)
    return df


def add_account_behavior_demand(df: pd.DataFrame, rng) -> pd.DataFrame:
    '''
    How much the applicant would revolve and spend on the card if the line allowed it.
    Riskier customers revolve more of what they spend. Used only by the credit simulator.
    :param df: applicant DataFrame
    :param rng: numpy Generator
    :return: df with revolving_balance_demand and monthly_spend_demand
    '''
    n = len(df)
    revolve_share = np.clip(0.04 + 1.5 * df["true_pd_12m"] + 0.05 * df["revolving_utilization"], 0.02, 0.35)
    df["revolving_balance_demand"] = (df["annual_income"] * revolve_share * rng.lognormal(0, 0.5, n)).round(
        -1
    )
    df["monthly_spend_demand"] = (df["annual_income"] / 12 * 0.10 * rng.lognormal(0, 0.5, n)).round(-1)
    return df


def add_historical_decisions_and_outcomes(df: pd.DataFrame, rng) -> pd.DataFrame:
    '''
    The historical underwriting rule (with a few manual overrides) decides who was booked.
    Booked applicants get an observed 12-month default; declined applicants get NaN.
    Later vintages see a mild economic deterioration, so defaults drift up over time.
    :param df: applicant DataFrame with true_pd_12m
    :param rng: numpy Generator
    :return: df with is_booked and defaulted_12m
    '''
    n = len(df)
    passes_rule = (
        (df["fico_score"] >= 650) & (df["debt_to_income"] < 0.45) & (df["num_delinquencies_24m"] <= 1)
    )
    override = rng.random(n)
    df["is_booked"] = np.where(passes_rule, override > 0.03, override < 0.05).astype(int)

    vintage_drift = 0.25 * np.clip((df["vintage"] - 12) / 12, 0, 1)
    realized_pd = sigmoid(np.log(df["true_pd_12m"] / (1 - df["true_pd_12m"])) + vintage_drift)
    defaulted = (rng.random(n) < realized_pd).astype(float)
    df["defaulted_12m"] = np.where(df["is_booked"] == 1, defaulted, np.nan)
    return df


def simulate_applicant_population(config: UnderwritingConfig) -> pd.DataFrame:
    '''
    Simulates the full through-the-door applicant population.
    :param config: UnderwritingConfig
    :return: DataFrame with one row per application
    '''
    rng = np.random.default_rng(config.seed)
    df = pd.DataFrame({"application_id": np.arange(config.n_applicants)})
    df["latent_creditworthiness"] = rng.normal(0, 1, config.n_applicants)

    ##### Application Timing #####
    df = add_application_dates(df, config, rng)

    ##### Applicant Attributes #####
    df = add_bureau_attributes(df, rng)
    df = add_application_attributes(df, rng)

    ##### Ground Truth #####
    df = add_true_default_probability(df, rng)
    df = add_account_behavior_demand(df, rng)

    ##### Historical Policy and Observed Outcomes #####
    df = add_historical_decisions_and_outcomes(df, rng)
    return df.sort_values("application_date").reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = simulate_applicant_population(config.underwriting)
    path = config.paths.applications_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    booked = df["is_booked"] == 1
    logger.info(
        "Simulated %d applications, %.1f%% booked, booked default rate %.2f%%, "
        "through-the-door true PD %.2f%%",
        len(df),
        100 * booked.mean(),
        100 * df.loc[booked, "defaulted_12m"].mean(),
        100 * df["true_pd_12m"].mean(),
    )


if __name__ == "__main__":
    main()
