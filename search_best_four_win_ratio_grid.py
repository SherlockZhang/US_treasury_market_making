"""Rerun grids, ranking net Sharpe only where gross win ratio exceeds 50%."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_cdf_custom_vol_daily_weekly as cdf_search
import search_cdf_rounding_grid as rounding_grid
import search_ewma_kama_trading_cost_grid as trend_search


OUTPUT_DIR = reference.RESULTS / "best_four_gross_win_ratio_above_50_grid_2016_2025"


def rank_eligible(grid: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Keep every candidate but rank only finite-Sharpe candidates winning over 50%."""
    ranked = grid.rename(columns={"sharpe_rank": "unfiltered_sharpe_rank"}).copy()
    finite_sharpe = np.isfinite(ranked["sharpe_ratio"])
    finite_win_ratio = np.isfinite(ranked["win_ratio_pct"])
    ranked["selection_status"] = np.select(
        [
            ~finite_sharpe,
            ~finite_win_ratio,
            ranked["win_ratio_pct"].le(50),
        ],
        ["undefined_sharpe", "undefined_win_ratio", "win_ratio_not_above_50pct"],
        default="eligible",
    )
    ranked["eligible"] = ranked["selection_status"].eq("eligible")
    ranked = ranked.sort_values(
        group_columns + ["eligible", "sharpe_ratio"],
        ascending=[True] * len(group_columns) + [False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranked["sharpe_rank"] = (
        ranked.groupby(group_columns, sort=False)["eligible"].cumsum()
        .where(ranked["eligible"])
        .astype("Int64")
    )
    ranked["selected_best"] = ranked["sharpe_rank"].eq(1).fillna(False)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    all_prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    prices = {instrument: all_prices[instrument] for instrument in trend_search.INSTRUMENTS}
    grids = {
        "cdf": rank_eligible(
            cdf_search.search(prices, rounding_grid.ROUNDING_INCREMENTS),
            ["instrument", "volatility_scaling"],
        ),
    }
    for model, grid in trend_search.search(prices).items():
        grids[model.lower()] = rank_eligible(grid, ["instrument"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, grid in grids.items():
        grid.to_csv(args.output_dir / f"{name}_parameter_grid.csv", index=False, float_format="%.10f")
        best = grid.loc[grid["selected_best"]]
        best.to_csv(args.output_dir / f"{name}_best_parameters.csv", index=False, float_format="%.10f")
        groups = ["instrument", "volatility_scaling"] if name == "cdf" else ["instrument"]
        counts = grid.groupby(groups, dropna=False)["eligible"].sum()
        missing = counts[counts.eq(0)]
        if not missing.empty:
            print(f"WARNING: No eligible {name} candidates for: {missing.index.tolist()}")
        print(f"\n{name.upper()} winners:")
        print(best.to_string(index=False))
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
