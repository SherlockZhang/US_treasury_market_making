"""Rank 2Y CDF lookbacks by the sum of Sharpe, drawdown, and Calmar ranks."""

import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as capped
import analyze_best_four_futures_strategies as reference
from rank_search_metrics import METRICS, rank_metrics


OUTPUT = (
    reference.RESULTS
    / "2y_cdf_rank_sum_rounding_0p5_cap_3_target_vol_1p5pct_grid_2016_2025"
)
GROUPS = ["volatility_scaling", "rebalance_frequency"]
def rank_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """Rank each metric from best to worst within a scaling/frequency group."""
    ranked = rank_metrics(grid, GROUPS)
    return ranked.sort_values(
        GROUPS + ["rank_sum", "signal_lookback_days"],
        ascending=[False, True, True, True], kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[capped.INSTRUMENT]
    grid = rank_grid(capped.search(prices))
    best = grid.loc[grid["selected_best"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT / "cdf_parameter_grid.csv", index=False, float_format="%.10f")
    best.to_csv(OUTPUT / "cdf_best_parameters.csv", index=False, float_format="%.10f")
    print(best.to_string(index=False))
    counts = best.groupby(GROUPS).size()
    if len(counts) != 4 or counts.ne(1).any():
        raise ValueError(
            "Expected one winner per CDF mode; tied winners are retained in the output."
        )
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
