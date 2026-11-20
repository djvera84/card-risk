import pandas as pd

from card_risk.features.customer_history import get_days_between


def add_account_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Features about the account and card at the time of the transaction.
    :param df: transactions DataFrame
    :return: df with account features added
    '''
    ##### Credit Usage #####
    df["credit_utilization"] = df["currentBalance"] / df["creditLimit"]
    df["transaction_use_of_credit_limit"] = df["transactionAmount"] / df["creditLimit"]
    df["transaction_use_of_available_money"] = df["transactionAmount"] / df["availableMoney"].where(
        df["availableMoney"] > 0
    )

    ##### Card Verification #####
    df["CVV_match"] = (df["enteredCVV"] == df["cardCVV"]).astype(int)
    df["cardPresent"] = df["cardPresent"].astype(int)
    df["expirationDateKeyInMatch"] = df["expirationDateKeyInMatch"].astype(int)

    ##### Account Timeline #####
    df["days_since_last_address_change"] = get_days_between(
        df["transactionDateTime"], df["dateOfLastAddressChange"]
    )
    df["days_until_expiration"] = get_days_between(df["currentExpDate"], df["transactionDateTime"])

    ##### Geography #####
    df["is_foreign_merchant"] = (
        df["merchantCountryCode"].notna()
        & df["acqCountry"].notna()
        & (df["merchantCountryCode"] != df["acqCountry"])
    ).astype(int)
    return df
