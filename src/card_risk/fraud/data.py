import pandas as pd

from card_risk.config import Config
from card_risk.data_utils.split import out_of_time_split
from card_risk.features.build import LABEL


def load_fraud_model_splits(config: Config) -> dict[str, pd.DataFrame]:
    '''
    Loads the feature table and splits it out of time into train, valid and test.
    The test set is the most recent split.test_fraction of transactions; valid is the
    most recent split.valid_fraction of what remains (used for early stopping and
    choosing the threshold).
    :param config: pipeline Config
    :return: dict with "train", "valid", "test" DataFrames
    '''
    path = config.paths.features_parquet
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run fraud-build-features first")
    df = pd.read_parquet(path)
    df_train_full, df_test = out_of_time_split(df, config.split.test_fraction)
    df_train, df_valid = out_of_time_split(df_train_full, config.split.valid_fraction)
    return {"train": df_train, "valid": df_valid, "test": df_test, "train_full": df_train_full}


def get_label(df: pd.DataFrame) -> pd.Series:
    return df[LABEL]
