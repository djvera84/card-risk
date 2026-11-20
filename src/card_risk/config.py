'''
Typed configuration loaded from a YAML file (configs/default.yaml by default).
'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


@dataclass
class PathsConfig:
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")

    @property
    def raw_transactions_parquet(self) -> Path:
        return self.interim_dir / "transactions.parquet"

    @property
    def features_parquet(self) -> Path:
        return self.processed_dir / "features.parquet"

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "models" / "fraud"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def best_params_json(self) -> Path:
        return self.artifacts_dir / "best_params.json"

    @property
    def simulated_dir(self) -> Path:
        return Path("data/simulated")

    @property
    def applications_parquet(self) -> Path:
        return self.simulated_dir / "applications.parquet"

    @property
    def scored_applications_parquet(self) -> Path:
        return self.simulated_dir / "scored_applications.parquet"

    @property
    def underwriting_model_dir(self) -> Path:
        return self.artifacts_dir / "models" / "underwriting"


@dataclass
class DataConfig:
    download_url: str = "https://raw.github.com/CapitalOneRecruiting/DS/master/transactions.zip"
    raw_filename: str = "transactions.txt"


@dataclass
class SplitConfig:
    test_fraction: float = 0.05
    valid_fraction: float = 0.05


@dataclass
class TrainingConfig:
    random_state: int = 42
    n_estimators: int = 1000
    early_stopping_rounds: int = 25
    params: dict = field(default_factory=dict)
    target_recall: float = 0.80


@dataclass
class TuningConfig:
    n_iter: int = 20
    n_splits: int = 3
    n_estimators: int = 200
    param_space: dict = field(default_factory=dict)


@dataclass
class BaselinesConfig:
    n_splits: int = 5
    max_rows: int = 300_000


@dataclass
class UnderwritingConfig:
    n_applicants: int = 200_000
    n_vintages: int = 24
    first_vintage: str = "2017-01-01"
    seed: int = 7
    # Vintages (1-indexed months) used for training and validation; the rest is the out-of-time test.
    train_vintages: int = 16
    valid_vintages: int = 2
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    params: dict = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    # Monte Carlo
    n_economic_paths: int = 1000
    n_random_policies: int = 2000
    seed: int = 11
    # Horizon and discounting
    n_quarters: int = 12
    annual_discount_rate: float = 0.12
    # Card economics
    interchange_rate: float = 0.02
    rewards_rate: float = 0.01
    annual_fee: float = 0.0
    servicing_cost_per_month: float = 10.0
    acquisition_cost: float = 300.0
    # Transactors and promotional balances do not pay interest.
    share_of_balance_accruing_interest: float = 0.65
    annual_attrition_rate: float = 0.12
    exposure_at_default_utilization: float = 0.90
    # Used if the fraud model's metrics are not available.
    fraud_dollars_per_purchase_dollar: float = 0.024
    fraud_dollar_recall: float = 0.0
    # Default credit policy: one entry per underwriting-score band, band 1 = riskiest.
    approve_threshold: float = 0.90
    credit_lines_by_band: list = field(
        default_factory=lambda: [500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 9000, 12000]
    )
    apr_by_band: list = field(
        default_factory=lambda: [0.299, 0.289, 0.279, 0.269, 0.259, 0.249, 0.239, 0.229, 0.219, 0.199]
    )


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    underwriting: UnderwritingConfig = field(default_factory=UnderwritingConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)


def load_config(config_path: str | Path | None = None) -> Config:
    '''
    Loads the pipeline configuration from YAML, falling back to dataclass defaults
    for anything the file does not set.
    :param config_path: path to a YAML config; defaults to configs/default.yaml if it exists
    :return: Config
    '''
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    paths = {key: Path(value) for key, value in raw.get("paths", {}).items()}
    return Config(
        paths=PathsConfig(**paths),
        data=DataConfig(**raw.get("data", {})),
        split=SplitConfig(**raw.get("split", {})),
        training=TrainingConfig(**raw.get("training", {})),
        tuning=TuningConfig(**raw.get("tuning", {})),
        baselines=BaselinesConfig(**raw.get("baselines", {})),
        underwriting=UnderwritingConfig(**raw.get("underwriting", {})),
        simulator=SimulatorConfig(**raw.get("simulator", {})),
    )
