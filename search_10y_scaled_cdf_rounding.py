"""Search weekly 10Y scaled-CDF lookbacks and position rounding through 2025."""

import search_5y_scaled_cdf_rounding as common


INSTRUMENT = "10Y future"
ROUNDING_INCREMENTS = (0.1, 0.2, 0.3, 0.4)
OUTPUT_FILE = common.OUTPUT_DIR / "10y_scaled_cdf_rounding_grid_2016_2025.csv"
BEST_FILE = common.OUTPUT_DIR / "10y_scaled_cdf_rounding_best_2016_2025.csv"


def main() -> None:
    prices = common.reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    grid = common.search(prices, INSTRUMENT, ROUNDING_INCREMENTS)
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_FILE, index=False, float_format="%.8f")
    grid.head(1).to_csv(BEST_FILE, index=False, float_format="%.8f")
    print(grid.to_string(index=False))
    print(f"\nFull grid: {OUTPUT_FILE}\nBest candidate: {BEST_FILE}")


if __name__ == "__main__":
    main()
