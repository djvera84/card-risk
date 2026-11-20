import pandas as pd

# Fields that are empty strings for every record in the raw data.
EMPTY_COLUMNS = [
    "echoBuffer",
    "merchantCity",
    "merchantState",
    "merchantZip",
    "posOnPremises",
    "recurringAuthInd",
]

DATETIME_COLUMNS = [
    "transactionDateTime",
    "currentExpDate",
    "accountOpenDate",
    "dateOfLastAddressChange",
]


def fix_DataFrame_types(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Fixes the column datatypes, turns empty strings into proper nulls and sorts by
    transaction time.
    :param df: raw transactions DataFrame
    :return: cleaned DataFrame sorted by transactionDateTime with a fresh index
    '''
    df = df.copy()
    text_columns = df.select_dtypes(include="object").columns
    df[text_columns] = df[text_columns].mask(df[text_columns].eq(""))

    df["accountNumber"] = pd.to_numeric(df["accountNumber"])
    df["customerId"] = pd.to_numeric(df["customerId"])
    df["transactionDateTime"] = pd.to_datetime(df["transactionDateTime"])
    # Expiration dates are given as MM/YYYY.
    df["currentExpDate"] = pd.to_datetime(df["currentExpDate"], format="%m/%Y")
    df["accountOpenDate"] = pd.to_datetime(df["accountOpenDate"])
    df["dateOfLastAddressChange"] = pd.to_datetime(df["dateOfLastAddressChange"])
    df["cardCVV"] = df["cardCVV"].astype(int)
    df["enteredCVV"] = df["enteredCVV"].astype(int)
    df["cardLast4Digits"] = df["cardLast4Digits"].astype(int)
    df["cardPresent"] = df["cardPresent"].astype(bool)
    df["expirationDateKeyInMatch"] = df["expirationDateKeyInMatch"].astype(bool)
    df["isFraud"] = df["isFraud"].astype(int)

    df = df.sort_values(by="transactionDateTime", kind="mergesort").reset_index(drop=True)
    return df


def drop_empty_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    '''
    Drops columns that carry no information (all null).
    :param df: transactions DataFrame
    :param columns: columns to drop; defaults to EMPTY_COLUMNS, keeping only those present
    :return: DataFrame without the empty columns
    '''
    columns = EMPTY_COLUMNS if columns is None else columns
    present = [col for col in columns if col in df.columns]
    return df.drop(columns=present)


def summarize_fields(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Per-field summary: dtype, null count, unique count, min and max.
    :param df: transactions DataFrame
    :return: summary DataFrame indexed by field name
    '''
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "null_count": df.isna().sum(),
            "unique_count": df.nunique(),
        }
    )
    comparable = df.select_dtypes(include=["number", "datetime", "bool"])
    summary["min"] = comparable.min()
    summary["max"] = comparable.max()
    return summary
