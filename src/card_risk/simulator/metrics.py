import numpy as np


def summarize_credit_policy_metrics(
    projection: dict, n_applicants: int, credit_lines, acquisition_cost: float
) -> dict:
    '''
    The metric suite for one policy under one macro path.
    :param projection: output of project_portfolio_cashflows
    :param n_applicants: through-the-door applicants the policy was applied to
    :param credit_lines: initial lines of the booked accounts
    :param acquisition_cost: cost per booked account
    :return: dict of metric name to value
    '''
    quarterly = projection["quarterly"]
    n_booked = len(projection["npv"])
    years = len(quarterly) / 4
    avg_receivables = quarterly["receivables"].mean()
    total_charge_offs = quarterly["charge_offs"].sum()
    margin_before_losses = (
        quarterly["revenue"] - quarterly["rewards_cost"] - quarterly["fraud_losses"]
    ).sum()

    cumulative_discounted = quarterly["discounted_net_cash_flow"].cumsum() - acquisition_cost * n_booked
    breakeven = np.flatnonzero(cumulative_discounted.to_numpy() >= 0)

    def per_year_of_receivables(total):
        return float(total / avg_receivables / years) if avg_receivables > 0 else np.nan

    return {
        "npv": float(projection["npv"].sum()),
        "npv_per_account": float(projection["npv"].mean()) if n_booked else 0.0,
        "approval_rate": n_booked / n_applicants,
        "accounts_booked": n_booked,
        "avg_initial_line": float(np.mean(credit_lines)) if n_booked else 0.0,
        "committed_exposure": float(np.sum(credit_lines)),
        "avg_receivables": float(avg_receivables),
        "total_revenue": float(quarterly["revenue"].sum()),
        "total_charge_offs": float(total_charge_offs),
        "cumulative_unit_default_rate": float(quarterly["defaults"].sum() / n_booked) if n_booked else 0.0,
        "net_charge_off_rate": per_year_of_receivables(total_charge_offs),
        "risk_adjusted_margin": per_year_of_receivables(margin_before_losses - total_charge_offs),
        "return_on_assets": per_year_of_receivables(quarterly["net_cash_flow"].sum()),
        "breakeven_quarter": int(quarterly.index[breakeven[0]]) if len(breakeven) else np.nan,
    }
