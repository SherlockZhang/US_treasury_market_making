"""Rerun the CDF, EWMA, and KAMA rank-sum grids with current trading costs."""

from pathlib import Path

import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as capped
import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as cdf_rank
import search_5y_10y_cdf_rank_sum as longer
import search_ewma_kama_rank_sum as trends
import search_ewma_kama_trading_cost_grid as trend_space


OUTPUT = reference.RESULTS / "all_models_rank_sum_updated_trading_costs_grid_2016_2025"


def save_report(directory: Path = OUTPUT) -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    cdf_grids = {
        capped.INSTRUMENT: cdf_rank.rank_grid(capped.search(prices[capped.INSTRUMENT])),
        **longer.search({instrument: prices[instrument] for instrument in longer.INSTRUMENTS}),
    }
    trend_grids = trends.search({instrument: prices[instrument] for instrument in trend_space.INSTRUMENTS})
    directory.mkdir(parents=True, exist_ok=True)
    winners = []
    for instrument, grid in cdf_grids.items():
        best = grid.loc[grid["selected_best"]].copy()
        grid.to_csv(directory / f"{instrument.split()[0].lower()}_cdf_parameter_grid.csv",
                    index=False, float_format="%.17g")
        best.to_csv(directory / f"{instrument.split()[0].lower()}_cdf_best_parameters.csv",
                    index=False, float_format="%.17g")
        best.insert(0, "model", "Normal CDF")
        winners.append(best)
    trend_space.parameter_space().to_csv(directory / "trend_parameter_space.csv", index=False)
    for model, grid in trend_grids.items():
        best = grid.loc[grid["selected_best"]].copy()
        best.insert(4, "parameter_combinations", best["instrument"].map(grid.groupby("instrument").size()))
        grid.to_csv(directory / f"{model.lower()}_parameter_grid.csv", index=False, float_format="%.17g")
        best.to_csv(directory / f"{model.lower()}_best_parameters.csv", index=False, float_format="%.17g")
        best.insert(0, "model", model)
        winners.append(best)
    combined = pd.concat(winners, ignore_index=True)
    combined.to_csv(directory / "all_best_parameters.csv", index=False, float_format="%.17g")
    print(f"Outputs: {directory}")
    print(combined[[
        "model", "instrument", "volatility_scaling", "rebalance_frequency",
        "signal_lookback_days", "n", "m", "alpha_fast_n", "alpha_slow_n",
        "er_lookback_days", "kama_extreme_window_m", "rank_sum",
        "sharpe_ratio", "max_drawdown_pct", "calmar_ratio", "trading_cost_ratio",
    ]].to_string(index=False))


if __name__ == "__main__":
    save_report()
