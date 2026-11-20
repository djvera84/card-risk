from pathlib import Path

import pandas as pd

from card_risk.config import Config
from card_risk.data_utils.clean import drop_empty_columns, fix_DataFrame_types
from card_risk.data_utils.download import fetch_transactions_data
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)


def parse_transactions_file_to_DataFrame(raw_path: Path) -> pd.DataFrame:
    '''
    Parses the line-delimited JSON transactions file.
    :param raw_path: path to transactions.txt
    :return: DataFrame with one row per transaction, columns as in the raw file
    '''
    return pd.read_json(raw_path, lines=True, dtype=False, convert_dates=False)


def load_transactions_data(config: Config, refresh: bool = False) -> pd.DataFrame:
    '''
    Loads the cleaned transactions. Uses the cached parquet in the interim folder
    when present; otherwise downloads (if missing), parses, cleans and caches it.
    :param config: pipeline Config
    :param refresh: rebuild the cache even if it exists
    :return: cleaned DataFrame sorted by transactionDateTime
    '''
    cache_path = config.paths.raw_transactions_parquet
    if cache_path.exists() and not refresh:
        logger.info("Loading cached transactions from %s", cache_path)
        return pd.read_parquet(cache_path)

    raw_path = fetch_transactions_data(
        config.data.download_url, config.paths.raw_dir, config.data.raw_filename
    )
    logger.info("Parsing %s", raw_path)
    df = parse_transactions_file_to_DataFrame(raw_path)
    df = fix_DataFrame_types(df)
    df = drop_empty_columns(df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached %d transactions x %d fields to %s", len(df), df.shape[1], cache_path)
    return df
