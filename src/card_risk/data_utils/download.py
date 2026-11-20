import os
import urllib.request
import zipfile
from pathlib import Path

from card_risk.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_transactions_data(data_url: str, raw_dir: Path, raw_filename: str = "transactions.txt") -> Path:
    '''
    Downloads the zipped transactions file and extracts it into raw_dir, but only
    if the extracted file is not already there.
    :param data_url: url of the zip file containing the line-delimited JSON transactions
    :param raw_dir: folder to download and extract the data into
    :param raw_filename: name of the extracted file
    :return: path to the extracted transactions file
    '''
    raw_dir = Path(raw_dir)
    raw_path = raw_dir / raw_filename
    if raw_path.exists():
        logger.info("Raw data already present at %s, skipping download", raw_path)
        return raw_path

    os.makedirs(raw_dir, exist_ok=True)
    zip_path = raw_dir / "transactions.zip"
    logger.info("Downloading %s", data_url)
    urllib.request.urlretrieve(data_url, zip_path)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(raw_dir)
    zip_path.unlink()

    if not raw_path.exists():
        raise FileNotFoundError(f"Expected {raw_filename} inside {data_url}")
    logger.info("Extracted %s", raw_path)
    return raw_path
