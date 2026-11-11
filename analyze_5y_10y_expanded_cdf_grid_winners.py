"""Backtest the 5Y and 10Y cost-adjusted four-strategy grid winners."""

import argparse
from pathlib import Path

import numpy as np

import analyze_2y_expanded_cdf_grid_winners as winners
import analyze_best_four_futures_strategies as reference
import search_cdf_custom_vol_daily_weekly as cdf_grid


INSTRUMENTS = ("5Y future", "10Y future")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=reference.RESULTS)
    args = parser.parse_args()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    for instrument in INSTRUMENTS:
        parameters, frequencies, rounding, sharpes = winners.selected_parameters(instrument)
        target, _ = cdf_grid.CONFIGURATIONS[instrument]
        frames = reference.run_period(
            prices[instrument].loc[:reference.TRAIN_END], parameters, reference.START,
            rounding[0], rounding[1], target, frequencies,
        )
        for name, frame in frames.items():
            actual = reference.statistics(frame)["sharpe_ratio"]
            if not np.isclose(actual, sharpes[name], rtol=0, atol=1e-8):
                raise ValueError(
                    f"{instrument} {name} training Sharpe {actual} "
                    f"does not match grid winner {sharpes[name]}."
                )
        stem = instrument.split()[0].lower()
        directory = args.output_root / f"{stem}_best_four_expanded_rounding_grid_2016_2025_and_2026"
        reference.save_report(
            instrument, prices[instrument], parameters, directory,
            rounding[0], rounding[1], target, frequencies,
        )


if __name__ == "__main__":
    main()
