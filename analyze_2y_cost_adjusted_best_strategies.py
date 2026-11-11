"""Backtest the 2Y cost-adjusted grid winners in-sample and out-of-sample."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference


INSTRUMENT = "2Y future"
TARGET = 0.015
ROUNDING = 1.0
CDF_BEST = reference.RESULTS / "cdf_custom_vol_daily_weekly_grid_2016_2025" / "best_parameters.csv"
EWMA_BEST = reference.RESULTS / "ewma_kama_cost_adjusted_grid_2016_2025" / "ewma_best_parameters.csv"
KAMA_BEST = reference.RESULTS / "ewma_kama_cost_adjusted_grid_2016_2025" / "kama_best_parameters.csv"
OUTPUT = reference.RESULTS / "2y_best_four_cost_adjusted_grid_winners_2016_2025_and_2026"


def selected_row(
    path: Path, scaled: bool | None = None, instrument: str = INSTRUMENT,
) -> pd.Series:
    table = pd.read_csv(path)
    rows = table.loc[table["instrument"].eq(instrument) & table["selected_best"].eq(True)]
    if scaled is not None:
        rows = rows.loc[rows["volatility_scaling"].eq(scaled)]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected {instrument} winner in {path} (scaled={scaled}).")
    row = rows.iloc[0]
    if row["sharpe_rank"] != 1 or row["end_date"] != "2025-12-31":
        raise ValueError(f"Unexpected ranking or training period in {path}.")
    if not np.isclose(row["trading_cost_ratio"], reference.cdf.TRADING_COST_RATIOS[instrument], rtol=0, atol=1e-12):
        raise ValueError(f"Trading cost in {path} does not match the current {instrument} rate.")
    return row


def selected_parameters() -> tuple[
    dict[str, dict[str, int]],
    tuple[reference.cdf.RebalanceFrequency, reference.cdf.RebalanceFrequency],
    dict[str, float],
]:
    scaled = selected_row(CDF_BEST, True)
    unscaled = selected_row(CDF_BEST, False)
    e = selected_row(EWMA_BEST)
    k = selected_row(KAMA_BEST)
    for row in (scaled, unscaled):
        if (
            row["position_rounding_increment"] != ROUNDING
            or row["volatility_lookback_days"] != reference.CDF_VOLATILITY_LOOKBACK
            or row["signal_cutoff"] != 0
            or row["rebalance_threshold_fraction"] != 0
        ):
            raise ValueError("Selected CDF grid settings do not match the report.")
    if not np.isclose(scaled["target_annualized_volatility_pct"], TARGET * 100):
        raise ValueError("Selected scaled CDF target does not match the report.")
    scaled_frequency = scaled["rebalance_frequency"]
    unscaled_frequency = unscaled["rebalance_frequency"]
    if scaled_frequency not in ("daily", "weekly") or unscaled_frequency not in ("daily", "weekly"):
        raise ValueError("Selected CDF rebalance frequency is unsupported.")
    parameters = {
        "Normal CDF scaled": {"lookback": int(scaled["signal_lookback_days"])},
        "Normal CDF unscaled": {"lookback": int(unscaled["signal_lookback_days"])},
        "EWMA": {"n": int(e["n"]), "m": int(e["m"])},
        "KAMA": {
            "fast": int(k["alpha_fast_n"]), "slow": int(k["alpha_slow_n"]),
            "er": int(k["er_lookback_days"]), "m": int(k["kama_extreme_window_m"]),
        },
    }
    sharpes = {
        name: float(row["sharpe_ratio"])
        for name, row in zip(reference.STRATEGIES, (scaled, unscaled, e, k))
    }
    return parameters, (scaled_frequency, unscaled_frequency), sharpes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    parameters, frequencies, sharpes = selected_parameters()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    frames = reference.run_period(
        prices.loc[:reference.TRAIN_END], parameters, reference.START,
        ROUNDING, ROUNDING, TARGET, frequencies,
    )
    for name, frame in frames.items():
        actual = reference.statistics(frame)["sharpe_ratio"]
        if not np.isclose(actual, sharpes[name], rtol=0, atol=1e-8):
            raise ValueError(f"{name} training Sharpe {actual} does not match grid winner {sharpes[name]}.")
    reference.save_report(
        INSTRUMENT, prices, parameters, args.output_dir, ROUNDING, ROUNDING, TARGET, frequencies,
    )


if __name__ == "__main__":
    main()
