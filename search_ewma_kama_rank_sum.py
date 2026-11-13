"""Search EWMA and KAMA using the sum of Sharpe, drawdown, and Calmar ranks."""

import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_ewma_kama_trading_cost_grid as previous
from rank_search_metrics import rank_metrics


OUTPUT_DIR = reference.RESULTS / "ewma_kama_rank_sum_cost_adjusted_grid_2016_2025"


def search(prices_by_instrument: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    grids = previous.search(prices_by_instrument)
    return {
        model: rank_metrics(grid, ["instrument"]).sort_values(
            ["instrument", "rank_sum", *previous.PARAMETERS[model]],
            kind="stable",
        ).reset_index(drop=True)
        for model, grid in grids.items()
    }


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    grids = search({instrument: prices[instrument] for instrument in previous.INSTRUMENTS})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous.parameter_space().to_csv(OUTPUT_DIR / "parameter_space.csv", index=False)
    for model, grid in grids.items():
        stem = model.lower()
        best = grid.loc[grid["selected_best"]].copy()
        best.insert(
            4, "parameter_combinations",
            best["instrument"].map(grid.groupby("instrument").size()),
        )
        grid.to_csv(OUTPUT_DIR / f"{stem}_parameter_grid.csv", index=False, float_format="%.10f")
        best.to_csv(OUTPUT_DIR / f"{stem}_best_parameters.csv", index=False, float_format="%.10f")
        print(f"\n{model} winners:\n{best.to_string(index=False)}")
        counts = best.groupby("instrument").size()
        if len(counts) != len(previous.INSTRUMENTS) or counts.ne(1).any():
            print(f"{model}: tied winners, if any, are all included in the best-parameters file.")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
