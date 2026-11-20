'''
NPV / lifetime value model: expected quarterly cash flows of every booked account.
'''

from dataclasses import dataclass

import numpy as np
import pandas as pd

from card_risk.simulator.loss_model import (
    forecast_cost_of_funds,
    forecast_loss_given_default,
    forecast_quarterly_default_probability,
    forecast_revolving_balance_multiplier,
    forecast_spend_multiplier,
)


@dataclass
class CardEconomics:
    n_quarters: int = 12
    annual_discount_rate: float = 0.12
    interchange_rate: float = 0.02
    rewards_rate: float = 0.01
    annual_fee: float = 0.0
    servicing_cost_per_month: float = 10.0
    acquisition_cost: float = 300.0
    share_of_balance_accruing_interest: float = 0.65
    annual_attrition_rate: float = 0.12
    exposure_at_default_utilization: float = 0.90
    fraud_dollars_per_purchase_dollar: float = 0.024
    fraud_dollar_recall: float = 0.0

    @property
    def net_fraud_loss_rate(self) -> float:
        '''Fraud losses per purchase dollar after the fraud model blocks what it catches.'''
        return self.fraud_dollars_per_purchase_dollar * (1 - self.fraud_dollar_recall)

    @classmethod
    def from_config(cls, simulator_config, fraud_metrics: dict | None = None) -> "CardEconomics":
        fields = {name: getattr(simulator_config, name) for name in cls.__dataclass_fields__}
        if fraud_metrics:
            fields["fraud_dollars_per_purchase_dollar"] = fraud_metrics["fraud_dollars_per_purchase_dollar"]
            fields["fraud_dollar_recall"] = fraud_metrics["fraud_dollar_recall"]
        return cls(**fields)


QUARTERLY_COMPONENTS = [
    "interest_income",
    "interchange_income",
    "fee_income",
    "rewards_cost",
    "fraud_losses",
    "funding_cost",
    "servicing_cost",
    "charge_offs",
]


def project_portfolio_cashflows(
    accounts: pd.DataFrame,
    credit_lines: np.ndarray,
    aprs: np.ndarray,
    macro_path: pd.DataFrame,
    economics: CardEconomics,
    quarterly_pd: np.ndarray | None = None,
) -> dict:
    '''
    Expected cash flows of the booked accounts, quarter by quarter.
    Balances are the customer's revolving demand (scaled by the economy) capped by the
    line; at default the customer has drawn the line down to exposure_at_default_utilization.
    :param accounts: booked accounts with true_pd_12m, revolving_balance_demand, monthly_spend_demand
    :param credit_lines: (n,) initial credit line per account
    :param aprs: (n,) APR per account
    :param macro_path: output of simulate_macro_path
    :param economics: CardEconomics
    :param quarterly_pd: optional precomputed (n, n_quarters) PDs for these accounts
    :return: dict with per-account "npv" (n,), "quarterly" DataFrame of portfolio totals
    '''
    n_quarters = len(macro_path)
    lines = np.asarray(credit_lines, dtype=float)[:, None]
    aprs = np.asarray(aprs, dtype=float)[:, None]
    if quarterly_pd is None:
        quarterly_pd = forecast_quarterly_default_probability(accounts["true_pd_12m"].to_numpy(), macro_path)

    ##### Survival: accounts still open and current at the start of each quarter #####
    quarterly_attrition = 1 - (1 - economics.annual_attrition_rate) ** 0.25
    stay = (1 - quarterly_pd) * (1 - quarterly_attrition)
    open_at_start = np.concatenate([np.ones((len(lines), 1)), np.cumprod(stay, axis=1)[:, :-1]], axis=1)

    ##### Balances and Spend #####
    revolving_demand = accounts["revolving_balance_demand"].to_numpy()[:, None]
    balance = np.minimum(
        revolving_demand * forecast_revolving_balance_multiplier(macro_path)[None, :], 0.95 * lines
    )
    spend_demand = 3 * accounts["monthly_spend_demand"].to_numpy()[:, None]
    spend = np.minimum(spend_demand * forecast_spend_multiplier(macro_path)[None, :], lines)
    exposure_at_default = np.maximum(balance, economics.exposure_at_default_utilization * lines)

    ##### Quarterly Components (per account, before survival weighting) #####
    performing = 1 - quarterly_pd
    components = {
        "interest_income": performing * balance * economics.share_of_balance_accruing_interest * aprs / 4,
        "interchange_income": performing * spend * economics.interchange_rate,
        "fee_income": performing * np.full_like(balance, economics.annual_fee / 4),
        "rewards_cost": performing * spend * economics.rewards_rate,
        "fraud_losses": spend * economics.net_fraud_loss_rate,
        "funding_cost": balance * forecast_cost_of_funds(macro_path)[None, :] / 4,
        "servicing_cost": np.full_like(balance, 3 * economics.servicing_cost_per_month),
        "charge_offs": quarterly_pd * exposure_at_default * forecast_loss_given_default(macro_path)[None, :],
    }
    components = {name: value * open_at_start for name, value in components.items()}
    revenue = components["interest_income"] + components["interchange_income"] + components["fee_income"]
    costs = (
        components["rewards_cost"]
        + components["fraud_losses"]
        + components["funding_cost"]
        + components["servicing_cost"]
        + components["charge_offs"]
    )
    net_cash_flow = revenue - costs

    ##### Discounting #####
    discount = (1 + economics.annual_discount_rate) ** (-np.arange(1, n_quarters + 1) / 4)
    npv = (net_cash_flow * discount[None, :]).sum(axis=1) - economics.acquisition_cost

    quarterly = pd.DataFrame(
        {name: value.sum(axis=0) for name, value in components.items()}, index=macro_path.index
    )
    quarterly["open_accounts"] = open_at_start.sum(axis=0)
    quarterly["defaults"] = (open_at_start * quarterly_pd).sum(axis=0)
    quarterly["receivables"] = (open_at_start * balance).sum(axis=0)
    quarterly["revenue"] = revenue.sum(axis=0)
    quarterly["net_cash_flow"] = net_cash_flow.sum(axis=0)
    quarterly["discounted_net_cash_flow"] = quarterly["net_cash_flow"] * discount
    return {"npv": npv, "quarterly": quarterly}
