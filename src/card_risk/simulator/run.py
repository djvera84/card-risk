'''
Run the credit policy simulator for the policy in the config.
Writes artifacts/reports/simulator/.

    credit-simulator [--config configs/default.yaml] [--n-paths 1000] [--n-policies 2000]

Mode 1: the configured policy across n_paths random economies.
Mode 2: n_policies random policies under each fixed economic scenario, with the
configured policy's rank among them.
'''

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from card_risk.config import load_config
from card_risk.evaluation.plots import save_fig
from card_risk.logging_utils import get_logger
from card_risk.simulator.cashflows import CardEconomics
from card_risk.simulator.economy import ECONOMIC_SCENARIOS, simulate_macro_path
from card_risk.simulator.monte_carlo import (
    sample_random_credit_policies,
    simulate_policies_in_fixed_economy,
    simulate_policy_across_economic_scenarios,
)
from card_risk.simulator.policy import CreditPolicy, CreditPolicySimulator

logger = get_logger(__name__)

SCENARIO_ORDER = list(ECONOMIC_SCENARIOS)
SCENARIO_COLORS = {
    "boom": "tab:green",
    "benign": "tab:blue",
    "minor_recession": "tab:orange",
    "recession": "tab:red",
}


def load_credit_policy_simulator(config) -> CreditPolicySimulator:
    '''
    The scored through-the-door population plus card economics, with fraud losses net
    of the fraud model when its test metrics are available.
    :param config: pipeline Config
    :return: CreditPolicySimulator
    '''
    population = pd.read_parquet(config.paths.scored_applications_parquet)
    fraud_metrics_path = config.paths.reports_dir / "fraud" / "metrics.json"
    fraud_metrics = json.loads(fraud_metrics_path.read_text()) if fraud_metrics_path.exists() else None
    economics = CardEconomics.from_config(config.simulator, fraud_metrics)
    return CreditPolicySimulator(population, economics)


def summarize_npv_by_scenario(df_paths: pd.DataFrame) -> pd.DataFrame:
    '''
    :param df_paths: output of simulate_policy_across_economic_scenarios
    :return: NPV and loss percentiles per scenario and overall
    '''

    def summarize(group):
        return pd.Series(
            {
                "paths": len(group),
                "npv_mean": group["npv"].mean(),
                "npv_p5": group["npv"].quantile(0.05),
                "npv_p50": group["npv"].median(),
                "npv_p95": group["npv"].quantile(0.95),
                "prob_npv_negative": (group["npv"] < 0).mean(),
                "net_charge_off_rate_mean": group["net_charge_off_rate"].mean(),
                "return_on_assets_mean": group["return_on_assets"].mean(),
            }
        )

    by_scenario = df_paths.groupby("scenario").apply(summarize, include_groups=False).reindex(SCENARIO_ORDER)
    by_scenario.loc["all"] = summarize(df_paths)
    return by_scenario


def plot_macro_scenario_paths(n_quarters: int, seed: int, paths_per_scenario: int = 40):
    rng = np.random.default_rng(seed)
    variables = [
        ("unemployment_rate", "Unemployment rate (%)"),
        ("gdp_growth", "Real GDP growth (%)"),
        ("cpi_inflation", "CPI inflation (%)"),
        ("case_shiller_hpi", "Case-Shiller HPI"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for scenario in SCENARIO_ORDER:
        for i in range(paths_per_scenario):
            path = simulate_macro_path(scenario, n_quarters, rng)
            for ax, (variable, _) in zip(axes.flat, variables, strict=True):
                ax.plot(
                    path.index,
                    path[variable],
                    color=SCENARIO_COLORS[scenario],
                    alpha=0.15,
                    label=scenario if i == 0 else None,
                )
    for ax, (_, title) in zip(axes.flat, variables, strict=True):
        ax.set_title(title)
        ax.set_xlabel("Quarter")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Simulated economic paths: scenario drawn after quarter 1")


def plot_policy_npv_across_economies(df_paths: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    bins = np.linspace(df_paths["npv"].min(), df_paths["npv"].max(), 50) / 1e6
    for scenario in SCENARIO_ORDER:
        subset = df_paths.loc[df_paths["scenario"] == scenario, "npv"] / 1e6
        plt.hist(
            subset, bins=bins, alpha=0.6, color=SCENARIO_COLORS[scenario], label=f"{scenario} ({len(subset)})"
        )
    plt.axvline(df_paths["npv"].mean() / 1e6, color="k", linestyle="--", label="mean")
    plt.axvline(df_paths["npv"].quantile(0.05) / 1e6, color="k", linestyle=":", label="5th percentile")
    plt.xlabel("Portfolio NPV ($M)")
    plt.ylabel("Economic paths")
    plt.title("Mode 1: fixed credit policy, random economies")
    plt.legend()


def plot_policies_in_fixed_economies(df_policies: pd.DataFrame, chosen: pd.DataFrame):
    fig, axes = plt.subplots(2, len(SCENARIO_ORDER), figsize=(18, 8))
    for i, scenario in enumerate(SCENARIO_ORDER):
        subset = df_policies[df_policies["scenario"] == scenario]
        mine = chosen.loc[scenario]
        axes[0, i].hist(subset["npv"] / 1e6, bins=50, color=SCENARIO_COLORS[scenario], alpha=0.7)
        axes[0, i].axvline(mine["npv"] / 1e6, color="k", linewidth=2)
        axes[0, i].set_title(f"{scenario}: your policy at P{mine['npv_percentile']:.0f}")
        axes[0, i].set_xlabel("Portfolio NPV ($M)")
        axes[1, i].scatter(
            100 * subset["net_charge_off_rate"],
            subset["npv"] / 1e6,
            s=6,
            alpha=0.3,
            color=SCENARIO_COLORS[scenario],
        )
        axes[1, i].scatter(
            100 * mine["net_charge_off_rate"],
            mine["npv"] / 1e6,
            s=120,
            marker="*",
            color="k",
            label="your policy",
        )
        axes[1, i].set_xlabel("Annualized net charge-off rate (%)")
        axes[1, i].set_ylabel("NPV ($M)")
        axes[1, i].legend(loc="lower right")
    fig.suptitle("Mode 2: fixed economy, random credit policies")


def plot_quarterly_projection_by_scenario(simulator, policy, n_quarters):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for scenario in SCENARIO_ORDER:
        _, projection = simulator.evaluate(
            policy, simulate_macro_path(scenario, n_quarters, with_noise=False), return_projection=True
        )
        quarterly = projection["quarterly"]
        nco_rate = 400 * quarterly["charge_offs"] / quarterly["receivables"]
        axes[0].plot(quarterly.index, nco_rate, color=SCENARIO_COLORS[scenario], marker="o", label=scenario)
        axes[1].plot(
            quarterly.index,
            quarterly["net_cash_flow"] / 1e6,
            color=SCENARIO_COLORS[scenario],
            marker="o",
            label=scenario,
        )
    axes[0].set_title("Annualized net charge-off rate (%)")
    axes[1].set_title("Net cash flow ($M)")
    for ax in axes:
        ax.set_xlabel("Quarter")
        ax.legend()
    fig.suptitle("Your policy under each scenario's expected path")


def run_credit_simulator(config, n_paths: int, n_policies: int) -> dict:
    '''
    Runs both Monte Carlo modes for the configured policy and writes the reports.
    :param config: pipeline Config
    :param n_paths: economic paths for mode 1
    :param n_policies: random policies per scenario for mode 2
    :return: summary dict
    '''
    sim_config = config.simulator
    simulator = load_credit_policy_simulator(config)
    policy = CreditPolicy.from_config(sim_config)
    n_quarters = sim_config.n_quarters
    reports_dir = config.paths.reports_dir / "simulator"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ##### Score Bands #####
    bands = simulator.get_band_score_ranges(policy.n_bands)
    bands["credit_line"] = policy.credit_lines_by_band
    bands["apr"] = policy.apr_by_band
    bands["approved"] = bands["max_score"] >= policy.approve_threshold
    bands.to_csv(reports_dir / "policy_bands.csv")
    logger.info("Policy by underwriting-score band:\n%s", bands.round(3))

    ##### Mode 1: Fixed Policy, Random Economies #####
    logger.info("Mode 1: %d economic paths", n_paths)
    df_paths = simulate_policy_across_economic_scenarios(
        simulator, policy, n_paths, n_quarters, sim_config.seed
    )
    df_paths.to_csv(reports_dir / "mode1_policy_across_economies.csv", index=False)
    by_scenario = summarize_npv_by_scenario(df_paths)
    by_scenario.to_csv(reports_dir / "mode1_npv_by_scenario.csv")
    logger.info("NPV by scenario:\n%s", by_scenario.round(3))

    ##### Mode 2: Fixed Economy, Random Policies #####
    rng = np.random.default_rng(sim_config.seed + 1)
    policies = sample_random_credit_policies(n_policies, policy.n_bands, rng)
    frames, chosen_rows = [], {}
    for scenario in SCENARIO_ORDER:
        logger.info("Mode 2: %d policies under %s", n_policies, scenario)
        df_policies = simulate_policies_in_fixed_economy(simulator, policies, scenario, n_quarters)
        frames.append(df_policies)
        mine = simulator.evaluate(policy, simulate_macro_path(scenario, n_quarters, with_noise=False))
        mine["npv_percentile"] = 100 * (df_policies["npv"] < mine["npv"]).mean()
        chosen_rows[scenario] = mine
    df_policies = pd.concat(frames, ignore_index=True)
    df_policies.to_csv(reports_dir / "mode2_policies_in_fixed_economies.csv", index=False)
    chosen = pd.DataFrame(chosen_rows).T
    best = (
        df_policies.loc[df_policies.groupby("scenario")["npv"].idxmax()]
        .set_index("scenario")
        .reindex(SCENARIO_ORDER)
    )

    ##### Figures #####
    plot_macro_scenario_paths(n_quarters, sim_config.seed)
    save_fig(reports_dir / "macro_scenario_paths.png")
    plot_policy_npv_across_economies(df_paths)
    save_fig(reports_dir / "mode1_npv_distribution.png")
    plot_policies_in_fixed_economies(df_policies, chosen)
    save_fig(reports_dir / "mode2_policy_npv_by_scenario.png")
    plot_quarterly_projection_by_scenario(simulator, policy, n_quarters)
    save_fig(reports_dir / "policy_quarterly_projection.png")

    summary = {
        "policy": {
            "approve_threshold": policy.approve_threshold,
            "credit_lines_by_band": policy.credit_lines_by_band,
            "apr_by_band": policy.apr_by_band,
        },
        "economics": simulator.economics.__dict__,
        "mode1_npv_by_scenario": by_scenario.round(4).to_dict(orient="index"),
        "mode2_your_policy": chosen[["npv", "npv_percentile", "approval_rate", "net_charge_off_rate"]]
        .astype(float)
        .round(4)
        .to_dict(orient="index"),
        "mode2_best_random_policy": best[
            [
                "npv",
                "approve_threshold",
                "credit_lines_by_band",
                "apr_by_band",
                "approval_rate",
                "net_charge_off_rate",
            ]
        ].to_dict(orient="index"),
    }
    with open(reports_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("Your policy vs random policies:\n%s", chosen[["npv", "npv_percentile"]])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--n-policies", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_credit_simulator(
        config,
        args.n_paths or config.simulator.n_economic_paths,
        args.n_policies or config.simulator.n_random_policies,
    )


if __name__ == "__main__":
    main()
