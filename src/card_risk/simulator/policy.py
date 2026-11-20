from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from card_risk.simulator.cashflows import CardEconomics, project_portfolio_cashflows
from card_risk.simulator.loss_model import forecast_quarterly_default_probability
from card_risk.simulator.metrics import summarize_credit_policy_metrics

SCORE = "underwriting_score"  # the underwriting model's P(no default in 12 months)


@dataclass
class CreditPolicy:
    '''
    A credit policy on the underwriting score, P(no default in 12 months).
    Bands are quantiles of the score on the through-the-door population, band 1 is the
    riskiest. The number of bands is len(credit_lines_by_band) (5 = quintiles, 10 = deciles, ...).
    '''

    approve_threshold: float
    credit_lines_by_band: list = field(default_factory=list)
    apr_by_band: list = field(default_factory=list)

    def __post_init__(self):
        if len(self.credit_lines_by_band) != len(self.apr_by_band):
            raise ValueError("credit_lines_by_band and apr_by_band need one entry per band")

    @property
    def n_bands(self) -> int:
        return len(self.credit_lines_by_band)

    @classmethod
    def from_config(cls, simulator_config) -> "CreditPolicy":
        return cls(
            simulator_config.approve_threshold,
            list(simulator_config.credit_lines_by_band),
            list(simulator_config.apr_by_band),
        )


def assign_score_bands(scores: np.ndarray, n_bands: int) -> np.ndarray:
    '''
    Quantile band of each applicant's underwriting score, 1 = riskiest.
    :param scores: P(no default) per applicant
    :param n_bands: number of quantile bands
    :return: integer bands in 1..n_bands
    '''
    ranks = pd.Series(scores).rank(method="first").to_numpy()
    return np.ceil(ranks / len(scores) * n_bands).astype(int).clip(1, n_bands)


class CreditPolicySimulator:
    '''
    Applies credit policies to a fixed through-the-door population and projects them
    under a macro path. Default probabilities for a macro path are cached, so many
    policies can be evaluated against the same economy cheaply.
    '''

    def __init__(self, population: pd.DataFrame, economics: CardEconomics):
        self.population = population.reset_index(drop=True)
        self.economics = economics
        self.scores = self.population[SCORE].to_numpy()
        self._bands = {}
        self._cached_macro_path = None
        self._cached_quarterly_pd = None

    def get_score_bands(self, n_bands: int) -> np.ndarray:
        if n_bands not in self._bands:
            self._bands[n_bands] = assign_score_bands(self.scores, n_bands)
        return self._bands[n_bands]

    def get_band_score_ranges(self, n_bands: int) -> pd.DataFrame:
        '''
        :return: min/max underwriting score and true default rate of each band
        '''
        df = pd.DataFrame(
            {
                "band": self.get_score_bands(n_bands),
                "score": self.scores,
                "true_pd_12m": self.population["true_pd_12m"],
            }
        )
        return df.groupby("band").agg(
            min_score=("score", "min"),
            max_score=("score", "max"),
            applicants=("score", "size"),
            true_pd_12m=("true_pd_12m", "mean"),
        )

    def _quarterly_pd_for(self, macro_path: pd.DataFrame) -> np.ndarray:
        if self._cached_macro_path is not macro_path:
            self._cached_quarterly_pd = forecast_quarterly_default_probability(
                self.population["true_pd_12m"].to_numpy(), macro_path
            )
            self._cached_macro_path = macro_path
        return self._cached_quarterly_pd

    def apply_credit_policy(self, policy: CreditPolicy) -> dict:
        '''
        :param policy: CreditPolicy
        :return: dict with boolean "approved" mask and per-applicant "credit_line", "apr"
        '''
        bands = self.get_score_bands(policy.n_bands)
        approved = self.scores >= policy.approve_threshold
        return {
            "approved": approved,
            "credit_line": np.asarray(policy.credit_lines_by_band, dtype=float)[bands - 1],
            "apr": np.asarray(policy.apr_by_band, dtype=float)[bands - 1],
            "band": bands,
        }

    def evaluate(self, policy: CreditPolicy, macro_path: pd.DataFrame, return_projection: bool = False):
        '''
        Books the policy and projects it under the macro path.
        :param policy: CreditPolicy
        :param macro_path: output of simulate_macro_path
        :param return_projection: also return the quarterly projection
        :return: metrics dict (and the projection if requested)
        '''
        decision = self.apply_credit_policy(policy)
        approved = decision["approved"]
        projection = project_portfolio_cashflows(
            self.population.loc[approved],
            decision["credit_line"][approved],
            decision["apr"][approved],
            macro_path,
            self.economics,
            quarterly_pd=self._quarterly_pd_for(macro_path)[approved],
        )
        metrics = summarize_credit_policy_metrics(
            projection,
            len(self.population),
            decision["credit_line"][approved],
            self.economics.acquisition_cost,
        )
        return (metrics, projection) if return_projection else metrics
