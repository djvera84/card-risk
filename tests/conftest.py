import numpy as np
import pandas as pd
import pytest

from card_risk.data_utils.clean import drop_empty_columns, fix_DataFrame_types


def make_raw_transaction(
    customer,
    time,
    amount,
    merchant="Uber #123",
    transaction_type="PURCHASE",
    is_fraud=False,
    entered_cvv="123",
):
    return {
        "accountNumber": str(customer),
        "customerId": str(customer),
        "creditLimit": 5000.0,
        "availableMoney": 4000.0,
        "transactionDateTime": time,
        "transactionAmount": amount,
        "merchantName": merchant,
        "acqCountry": "US",
        "merchantCountryCode": "US",
        "posEntryMode": "05",
        "posConditionCode": "01",
        "merchantCategoryCode": "rideshare",
        "currentExpDate": "06/2023",
        "accountOpenDate": "2015-03-14",
        "dateOfLastAddressChange": "2015-03-14",
        "cardCVV": "123",
        "enteredCVV": entered_cvv,
        "cardLast4Digits": "1234",
        "transactionType": transaction_type,
        "echoBuffer": "",
        "currentBalance": 1000.0,
        "merchantCity": "",
        "merchantState": "",
        "merchantZip": "",
        "cardPresent": False,
        "posOnPremises": "",
        "recurringAuthInd": "",
        "expirationDateKeyInMatch": False,
        "isFraud": is_fraud,
    }


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    rows = [
        make_raw_transaction(1, "2016-01-01T10:00:00", 10.0),
        make_raw_transaction(2, "2016-01-01T10:30:00", 0.5, merchant="AMC #9"),
        make_raw_transaction(1, "2016-01-01T12:00:00", 20.0, merchant="AMC #191138"),
        make_raw_transaction(1, "2016-01-01T12:01:00", 20.0, merchant="AMC #191138"),  # multi-swipe
        make_raw_transaction(1, "2016-01-02T09:00:00", 30.0, transaction_type="REVERSAL"),
        make_raw_transaction(
            2, "2016-01-03T10:30:00", 0.75, merchant="AMC #9", entered_cvv="999", is_fraud=True
        ),
        make_raw_transaction(1, "2016-01-05T10:00:00", 60.0, merchant="Uber #9"),
    ]
    return pd.DataFrame(rows).sample(frac=1, random_state=0)  # shuffled on purpose


@pytest.fixture
def clean_transactions(raw_transactions) -> pd.DataFrame:
    return drop_empty_columns(fix_DataFrame_types(raw_transactions))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
