"""Rank Treasury EWMA and KAMA parameters by cost-adjusted training Sharpe."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import backtest_ewma_extreme_strategy as ewma
import backtest_kama_extreme_strategy as kama


INSTRUMENTS = ("2Y future", "5Y future", "10Y future")
PARAMETERS = {
    "EWMA": {"n": tuple(range(5, 56, 5)), "m": tuple(range(5, 56, 5))},
    "KAMA": {
        "alpha_fast_n": (2, 3, 5),
        "alpha_slow_n": (10, 15, 20, 25),
        "er_lookback_days": (5, 10, 15, 20, 25),
        "kama_extreme_window_m": (3,),
    },
}
OUTPUT_DIR = reference.RESULTS / "ewma_kama_cost_adjusted_grid_2016_2025"


def parameter_space() -> pd.DataFrame:
    """Return one row per allowed model parameter value."""
    return pd.DataFrame(
        [
            {"model": model, "parameter": parameter, "value": value}
            for model, parameters in PARAMETERS.items()
            for parameter, values in parameters.items()
            for value in values
        ]
    )


def search(prices_by_instrument: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """Run the original parameter grids over the 2016-2025 training window."""
    grids = {}
    for model, parameters in PARAMETERS.items():
        rows = []
        for instrument, prices in prices_by_instrument.items():
            if instrument not in reference.cdf.TRADING_COST_RATIOS:
                raise ValueError(f"No trading cost ratio configured for {instrument!r}.")
            training = prices.loc[reference.START:reference.TRAIN_END].rename(instrument)
            for values in product(*parameters.values()):
                settings = dict(zip(parameters, values))
                if model == "EWMA":
                    frame = ewma.run_backtest(training, settings["n"], settings["m"])
                    constants = {"alpha_2_over_n_plus_1": 2 / (settings["n"] + 1)}
                else:
                    frame = kama.run_backtest(
                        training, settings["alpha_fast_n"], settings["alpha_slow_n"],
                        settings["er_lookback_days"], settings["kama_extreme_window_m"],
                    )
                    constants = {
                        "fast_smoothing_constant": 2 / (settings["alpha_fast_n"] + 1),
                        "slow_smoothing_constant": 2 / (settings["alpha_slow_n"] + 1),
                    }
                stats = reference.statistics(frame)
                if not np.isfinite(stats["sharpe_ratio"]):
                    raise ValueError(
                        f"Undefined Sharpe ratio for {instrument}, {model}, {settings}."
                    )
                rows.append({
                    "instrument": instrument,
                    "training_start_date": frame.index[0].date(),
                    "training_end_date": frame.index[-1].date(),
                    "training_observations": len(frame),
                    **settings,
                    **constants,
                    "trading_cost_ratio": reference.cdf.TRADING_COST_RATIOS[instrument],
                    "position_turnover": float(frame["position_change"].sum()),
                    "trading_cost_accumulated": float(frame["trading_cost"].sum()),
                    **stats,
                })
        grid = pd.DataFrame(rows)
        grid = grid.sort_values(
            ["instrument", "sharpe_ratio"], ascending=[True, False], kind="stable"
        ).reset_index(drop=True)
        grid.insert(0, "sharpe_rank", grid.groupby("instrument").cumcount() + 1)
        grid["selected_best"] = grid["sharpe_rank"].eq(1)
        grids[model] = grid
    return grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    grids = search({instrument: prices[instrument] for instrument in INSTRUMENTS})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_space().to_csv(args.output_dir / "parameter_space.csv", index=False)
    for model, grid in grids.items():
        stem = model.lower()
        grid.to_csv(args.output_dir / f"{stem}_parameter_grid.csv", index=False, float_format="%.10f")
        best = grid.loc[grid["selected_best"]].copy()
        best.insert(
            4, "parameter_combinations",
            best["instrument"].map(grid.groupby("instrument").size()),
        )
        best.to_csv(args.output_dir / f"{stem}_best_parameters.csv", index=False, float_format="%.10f")
        print(f"\n{model} best parameters:")
        print(best.to_string(index=False))
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
