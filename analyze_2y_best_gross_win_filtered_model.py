"""Backtest the overall best 2Y model from the gross-win-filtered grids."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference


INSTRUMENT = "2Y future"
GRID_DIR = reference.RESULTS / "best_four_gross_win_ratio_above_50_grid_2016_2025"
OUTPUT = reference.RESULTS / "2y_best_gross_win_filtered_model_2016_2025_and_2026"


def select_best() -> pd.Series:
    candidates = []
    for model in ("cdf", "ewma", "kama"):
        grid = pd.read_csv(GRID_DIR / f"{model}_best_parameters.csv")
        rows = grid.loc[grid["instrument"].eq(INSTRUMENT) & grid["selected_best"].eq(True)].copy()
        rows["model"] = model
        candidates.append(rows)
    best = pd.concat(candidates, ignore_index=True).sort_values(
        "sharpe_ratio", ascending=False, kind="stable"
    )
    if len(best) != 4 or not best["eligible"].all() or not best["win_ratio_pct"].gt(50).all():
        raise ValueError("Expected four eligible 2Y model winners.")
    row = best.iloc[0]
    if (
        row["model"] != "cdf" or not row["volatility_scaling"]
        or row["sharpe_status"] != "ok"
        or row["rebalance_frequency"] not in ("daily", "weekly")
        or row["volatility_lookback_days"] != reference.CDF_VOLATILITY_LOOKBACK
        or row["signal_cutoff"] != 0 or row["rebalance_threshold_fraction"] != 0
        or row["end_date"] != "2025-12-31"
        or not np.isclose(row["target_annualized_volatility_pct"], 1.5)
        or not np.isclose(row["trading_cost_ratio"], reference.cdf.TRADING_COST_RATIOS[INSTRUMENT])
    ):
        raise ValueError("The best model or its settings do not match this CDF report.")
    return row


def run(prices: pd.Series, row: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return reference.cdf.run_backtest(
        prices.loc[reference.START:end].rename(INSTRUMENT),
        lookback_days=int(row["signal_lookback_days"]),
        rebalance_frequency=row["rebalance_frequency"],
        position_change_threshold_fraction=0.0,
        signal_cutoff=0.0,
        signal_mode="rounded",
        volatility_lookback_days=int(row["volatility_lookback_days"]),
        output_start_date=start,
        position_rounding_increment=float(row["position_rounding_increment"]),
        use_volatility_scaling=True,
        target_annualized_volatility=float(row["target_annualized_volatility_pct"]) / 100,
        trading_cost_ratio=float(row["trading_cost_ratio"]),
    )


def save_report(directory: Path) -> None:
    row = select_best()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(directory / "selected_parameters.csv", index=False, float_format="%.10f")
    summary = []
    annual = []
    for period, start, end in (
        ("2016_2025", reference.START, reference.TRAIN_END),
        ("2026", reference.TEST_START, reference.TEST_END),
    ):
        frame = run(prices, row, start, end)
        frame["position_changed"] = frame["position_change"].ne(0)
        if period == "2016_2025":
            stats = reference.statistics(frame)
            for metric in ("sharpe_ratio", "win_ratio_pct"):
                if not np.isclose(stats[metric], row[metric], rtol=0, atol=1e-8):
                    raise ValueError(f"Training {metric} does not match the grid winner.")
        frame.to_csv(directory / f"{period}_daily.csv", index_label="date", float_format="%.10f")
        summary.append({
            "period": period, "instrument": INSTRUMENT, "strategy": "Normal CDF scaled",
            **reference.statistics(frame), **reference.trading_statistics(frame),
        })
        if period == "2016_2025":
            for year, yearly in frame.groupby(frame.index.year):
                annual.append({
                    "year": year, **reference.statistics(yearly),
                    **reference.trading_statistics(yearly),
                })
        for column, title, multiplier in (
            ("cumulative_pnl_pct", "Cumulative return (%)", 1),
            ("drawdown", "Drawdown (%)", 100),
        ):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(frame.index, frame[column] * multiplier)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title=f"{INSTRUMENT} best model: {period}", xlabel="Date", ylabel=title)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(directory / f"{period}_{column}.png", dpi=150)
            plt.close(fig)
    pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False, float_format="%.10f")
    pd.DataFrame(annual).to_csv(directory / "2016_2025_annual_stats.csv", index=False, float_format="%.10f")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nOutputs: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    save_report(args.output_dir)


if __name__ == "__main__":
    main()
