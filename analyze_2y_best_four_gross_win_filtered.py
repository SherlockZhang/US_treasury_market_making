"""Backtest the four 2Y winners selected by gross win ratio and net Sharpe."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_2y_cost_adjusted_best_strategies as previous
import analyze_2y_expanded_cdf_grid_winners as expanded
import analyze_best_four_futures_strategies as reference
import search_best_four_win_ratio_grid as search
import search_cdf_custom_vol_daily_weekly as cdf_grid


OUTPUT = reference.RESULTS / "2y_best_four_gross_win_filtered_2016_2025_and_2026"


def save_report(
    directory: Path, instrument: str = previous.INSTRUMENT,
    scaled_candidate: pd.Series | None = None,
) -> None:
    cdf_best = search.OUTPUT_DIR / "cdf_best_parameters.csv"
    ewma_best = search.OUTPUT_DIR / "ewma_best_parameters.csv"
    kama_best = search.OUTPUT_DIR / "kama_best_parameters.csv"
    parameters, frequencies, rounding, sharpes = expanded.selected_parameters(
        instrument, cdf_best=cdf_best, ewma_best=ewma_best, kama_best=kama_best,
    )
    winners = [
        previous.selected_row(cdf_best, True, instrument),
        previous.selected_row(cdf_best, False, instrument),
        previous.selected_row(ewma_best, instrument=instrument),
        previous.selected_row(kama_best, instrument=instrument),
    ]
    target, _ = cdf_grid.CONFIGURATIONS[instrument]
    if scaled_candidate is not None:
        row = scaled_candidate
        if (
            row["instrument"] != instrument or not row["volatility_scaling"]
            or row["sharpe_status"] != "ok"
            or row["rebalance_frequency"] not in ("daily", "weekly")
            or row["position_rounding_increment"] not in reference.cdf.POSITION_ROUNDING_INCREMENTS
            or row["volatility_lookback_days"] != reference.CDF_VOLATILITY_LOOKBACK
            or row["signal_cutoff"] != 0 or row["rebalance_threshold_fraction"] != 0
            or row["end_date"] != "2025-12-31"
            or not np.isclose(row["target_annualized_volatility_pct"], target * 100)
            or not np.isclose(row["trading_cost_ratio"], reference.cdf.TRADING_COST_RATIOS[instrument], rtol=0, atol=1e-12)
        ):
            raise ValueError(f"{instrument} scaled CDF candidate settings do not match the report.")
        winners[0] = row
        parameters["Normal CDF scaled"]["lookback"] = int(row["signal_lookback_days"])
        frequencies = (row["rebalance_frequency"], frequencies[1])
        rounding = (float(row["position_rounding_increment"]), rounding[1])
        sharpes["Normal CDF scaled"] = float(row["sharpe_ratio"])
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[instrument]
    frames = reference.run_period(
        prices.loc[:reference.TRAIN_END], parameters, reference.START,
        rounding[0], rounding[1], target, frequencies,
    )
    for name, row in zip(reference.STRATEGIES, winners):
        if not row["eligible"] or row["selection_status"] != "eligible" or row["win_ratio_pct"] <= 50:
            raise ValueError(f"{instrument} {name} is not an eligible gross-win-ratio grid winner.")
        stats = reference.statistics(frames[name])
        for metric, expected in (("sharpe_ratio", sharpes[name]), ("win_ratio_pct", row["win_ratio_pct"])):
            if not np.isclose(stats[metric], expected, rtol=0, atol=1e-8):
                raise ValueError(f"{instrument} {name} training {metric} {stats[metric]} does not match grid winner {expected}.")
    reference.save_report(
        instrument, prices, parameters, directory,
        rounding[0], rounding[1], target, frequencies,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    save_report(args.output_dir)


if __name__ == "__main__":
    main()
