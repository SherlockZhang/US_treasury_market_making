"""Compare daily and weekly 10Y scaled and unscaled rounded CDF models."""

import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_5y_cdf_rounding_0p2_daily_weekly as common


INSTRUMENT = "10Y future"
ROUNDING = 0.1
OUTPUT_FILE = reference.RESULTS / "10y_cdf_scaled_unscaled_rounding_0p1_daily_weekly_grid_2016_2025.csv"
BEST_FILE = reference.RESULTS / "10y_cdf_scaled_unscaled_rounding_0p1_daily_weekly_best_2016_2025.csv"


def search(prices: pd.Series) -> pd.DataFrame:
    return common.search(prices, INSTRUMENT, ROUNDING)


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    grid = search(prices)
    reference.RESULTS.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_FILE, index=False, float_format="%.8f")
    grid.loc[grid["sharpe_rank"].eq(1)].to_csv(BEST_FILE, index=False, float_format="%.8f")
    print(grid.to_string(index=False))
    print(f"\nFull grid: {OUTPUT_FILE}\nBest per model: {BEST_FILE}")


if __name__ == "__main__":
    main()
