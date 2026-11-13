"""Compare the 2Y rank-sum CDF winners with the best EWMA and KAMA."""

import numpy as np
import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as report
import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as ranking


OUTPUT = (
    reference.RESULTS
    / "2y_six_models_cdf_rank_sum_rounding_0p5_cap_3_target_vol_1p5pct_2016_2025_and_2026"
)


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[report.INSTRUMENT]
    saved = pd.read_csv(ranking.OUTPUT / "cdf_best_parameters.csv")
    grid = ranking.rank_grid(report.search(prices))
    winners = grid.loc[grid["selected_best"]].copy()
    keys = [*ranking.GROUPS, "signal_lookback_days", "rank_sum"]
    expected = saved[keys].sort_values(keys).reset_index(drop=True)
    actual = winners[keys].sort_values(keys).reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError("Current rank-sum winners do not match the saved grid winners.")
    if len(winners) != 4:
        raise ValueError("Expected one rank-sum winner for each CDF mode.")
    if not np.isfinite(winners[list(ranking.METRICS)]).all().all():
        raise ValueError("Rank-sum winners contain non-finite metrics.")
    report.save_report(prices, winners, OUTPUT)
    grid.to_csv(OUTPUT / "cdf_parameter_grid.csv", index=False, float_format="%.10f")


if __name__ == "__main__":
    main()
