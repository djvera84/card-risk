import numpy as np
import pandas as pd

from card_risk.underwriting.simulate import APPLICATION_CHANNELS, HOUSING_STATUSES

UW_NUM_FEATURES = [
    "fico_score",
    "annual_income",
    "debt_to_income",
    "revolving_utilization",
    "total_revolving_balance",
    "num_credit_inquiries_6m",
    "num_delinquencies_24m",
    "months_since_oldest_tradeline",
    "num_open_tradelines",
    "employment_years",
    "requested_credit_limit",
    "monthly_income",
    "monthly_debt_payment",
    "monthly_residual_income",
    "requested_line_to_income",
    "revolving_balance_to_income",
    "credit_age_years",
    "has_recent_delinquency",
    "inquiries_per_tradeline",
    "is_high_utilization",
]

UW_CAT_FEATURES = {
    "housing_status": HOUSING_STATUSES,
    "application_channel": APPLICATION_CHANNELS,
}

# Direction of the relationship with default risk, enforced in the XGBoost model so the
# score behaves sensibly (and explainably) when it extrapolates to declined applicants.
UW_MONOTONE_CONSTRAINTS = {
    "fico_score": -1,
    "debt_to_income": 1,
    "revolving_utilization": 1,
    "num_credit_inquiries_6m": 1,
    "num_delinquencies_24m": 1,
    "months_since_oldest_tradeline": -1,
    "annual_income": -1,
}


def add_affordability_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Ability-to-pay features from income and debt.
    :param df: applications DataFrame
    :return: df with affordability features
    '''
    df["monthly_income"] = df["annual_income"] / 12
    df["monthly_debt_payment"] = df["debt_to_income"] * df["monthly_income"]
    df["monthly_residual_income"] = df["monthly_income"] - df["monthly_debt_payment"]
    df["requested_line_to_income"] = df["requested_credit_limit"] / df["annual_income"]
    df["revolving_balance_to_income"] = df["total_revolving_balance"] / df["annual_income"]
    return df


def add_credit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Credit history depth and recent credit-seeking behaviour.
    :param df: applications DataFrame
    :return: df with credit history features
    '''
    df["credit_age_years"] = df["months_since_oldest_tradeline"] / 12
    df["has_recent_delinquency"] = (df["num_delinquencies_24m"] > 0).astype(int)
    df["inquiries_per_tradeline"] = df["num_credit_inquiries_6m"] / df["num_open_tradelines"].replace(
        0, np.nan
    )
    df["is_high_utilization"] = (df["revolving_utilization"] > 0.9).astype(int)
    return df


def one_hot_encode_fixed_categories(df: pd.DataFrame) -> pd.DataFrame:
    '''
    One-hot encodes the categorical features against a fixed list of levels, so
    training and scoring always produce the same columns.
    :param df: applications DataFrame
    :return: one-hot DataFrame
    '''
    encoded = []
    for column, levels in UW_CAT_FEATURES.items():
        categories = pd.Categorical(df[column], categories=levels)
        encoded.append(pd.get_dummies(categories, prefix=column, dtype=np.uint8).set_index(df.index))
    return pd.concat(encoded, axis=1)


def build_underwriting_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Model matrix for the underwriting model.
    :param df: applications DataFrame (simulated or real)
    :return: DataFrame of UW_NUM_FEATURES plus one-hot categoricals
    '''
    df = df.copy()

    ##### Affordability #####
    df = add_affordability_features(df)

    ##### Credit History #####
    df = add_credit_history_features(df)

    return pd.concat([df.loc[:, UW_NUM_FEATURES].astype(float), one_hot_encode_fixed_categories(df)], axis=1)
