'''
Stage 1: download, parse and clean the raw transactions into data/interim/transactions.parquet.

    fraud-prepare-data [--config configs/default.yaml] [--refresh]
'''

import argparse

from card_risk.config import load_config
from card_risk.data_utils.duplicates import summarize_duplicate_transactions
from card_risk.data_utils.load import load_transactions_data
from card_risk.logging_utils import get_logger

logger = get_logger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="rebuild the cached parquet")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = load_transactions_data(config, refresh=args.refresh)
    logger.info("%d transactions, fraud rate %.4f", len(df), df["isFraud"].mean())
    logger.info("Duplicate transactions:\n%s", summarize_duplicate_transactions(df))


if __name__ == "__main__":
    main()
