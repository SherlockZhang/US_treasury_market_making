"""Compare 5Y scaled and unscaled rounded-CDF lookbacks through 2025."""

import pandas as pd

import search_5y_scaled_cdf_rounding as common


INSTRUMENT = "5Y future"
ROUNDING = 0.2
OUTPUT_FILE = common.OUTPUT_DIR / "5y_cdf_scaled_unscaled_rounding_0p2_grid_2016_2025.csv"
BEST_FILE = common.OUTPUT_DIR / "5y_cdf_scaled_unscaled_rounding_0p2_best_2016_2025.csv"


def search(prices: pd.Series) -> pd.DataFrame:
    return pd.concat(
        [
            common.search(prices, INSTRUMENT, (ROUNDING,), use_volatility_scaling=scaled)
            for scaled in (True, False)
        ],
        ignore_index=True,
    )


def main() -> None:
    prices = common.reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    grid = search(prices)
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_FILE, index=False, float_format="%.8f")
    grid.loc[grid["sharpe_rank"].eq(1)].to_csv(
        BEST_FILE, index=False, float_format="%.8f"
    )
    print(grid.to_string(index=False))
    print(f"\nFull grid: {OUTPUT_FILE}\nBest per model: {BEST_FILE}")


if __name__ == "__main__":
    main()
