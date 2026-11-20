'''
Credit policy simulator.

A credit policy (initial credit line and APR by underwriting-score band, plus an
approve/decline threshold on the score) is booked on a through-the-door population and
projected quarter by quarter. After the first quarter the economy moves into a random
scenario (boom, benign, minor recession, recession) that drives unemployment, GDP, CPI
and the Case-Shiller home price index; lagged versions of those feed the loss
forecasting model, and the cash flow model turns everything into NPV and a suite of
portfolio metrics.

Monte Carlo is used two ways:
    1. Fixed policy, random economies: where does the policy land across economic outcomes?
    2. Fixed economy, random policies: given a scenario, what NPVs are possible and where
       does the chosen policy rank?
'''

from card_risk.simulator.cashflows import CardEconomics, project_portfolio_cashflows
from card_risk.simulator.economy import ECONOMIC_SCENARIOS, simulate_macro_path
from card_risk.simulator.metrics import summarize_credit_policy_metrics
from card_risk.simulator.monte_carlo import (
    sample_random_credit_policies,
    simulate_policies_in_fixed_economy,
    simulate_policy_across_economic_scenarios,
)
from card_risk.simulator.policy import CreditPolicy, CreditPolicySimulator

__all__ = [
    "ECONOMIC_SCENARIOS",
    "CardEconomics",
    "CreditPolicy",
    "CreditPolicySimulator",
    "project_portfolio_cashflows",
    "sample_random_credit_policies",
    "simulate_macro_path",
    "simulate_policies_in_fixed_economy",
    "simulate_policy_across_economic_scenarios",
    "summarize_credit_policy_metrics",
]
