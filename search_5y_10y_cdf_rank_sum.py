"""Rank 5Y and 10Y CDF lookbacks by Sharpe, drawdown, and Calmar."""

import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as ranking
import search_cdf_custom_vol_daily_weekly as common


INSTRUMENTS = ("5Y future", "10Y future")
ROUNDING = 0.5
OUTPUTS = {
    "5Y future": reference.RESULTS / "5y_cdf_rank_sum_rounding_0p5_target_vol_3pct_grid_2016_2025",
    "10Y future": reference.RESULTS / "10y_cdf_rank_sum_rounding_0p5_target_vol_5pct_grid_2016_2025",
}


def search(prices_by_instrument: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    if set(prices_by_instrument) != set(INSTRUMENTS):
        raise ValueError(f"Expected prices for exactly {INSTRUMENTS}.")
    grids = {}
    for instrument in INSTRUMENTS:
        grid = common.search({instrument: prices_by_instrument[instrument]}, (ROUNDING,))
        grids[instrument] = ranking.rank_grid(grid)
    return grids


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    grids = search({instrument: prices[instrument] for instrument in INSTRUMENTS})
    for instrument, grid in grids.items():
        best = grid.loc[grid["selected_best"]]
        output = OUTPUTS[instrument]
        output.mkdir(parents=True, exist_ok=True)
        grid.to_csv(output / "cdf_parameter_grid.csv", index=False, float_format="%.10f")
        best.to_csv(output / "cdf_best_parameters.csv", index=False, float_format="%.10f")
        print(f"\n{instrument}\n{best.to_string(index=False)}\nOutputs: {output}")
        counts = best.groupby(ranking.GROUPS).size()
        if len(counts) != 4 or counts.ne(1).any():
            raise ValueError(
                f"{instrument}: expected one winner per CDF mode; "
                "tied winners are retained in the output."
            )


if __name__ == "__main__":
    main()
