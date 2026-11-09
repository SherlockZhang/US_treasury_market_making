"""Compare daily and weekly 5Y scaled and unscaled rounded CDF models."""

import numpy as np
import pandas as pd

import search_5y_cdf_rounding_0p2 as previous
import search_5y_scaled_cdf_rounding as common


OUTPUT_FILE = common.OUTPUT_DIR / "5y_cdf_scaled_unscaled_rounding_0p2_daily_weekly_grid_2016_2025.csv"
BEST_FILE = common.OUTPUT_DIR / "5y_cdf_scaled_unscaled_rounding_0p2_daily_weekly_best_2016_2025.csv"
REBALANCE_FREQUENCIES: tuple[common.reference.cdf.RebalanceFrequency, ...] = ("daily", "weekly")


def search(
    prices: pd.Series,
    instrument: str = previous.INSTRUMENT,
    rounding: float = previous.ROUNDING,
) -> pd.DataFrame:
    parts = []
    for scaled in (True, False):
        model = pd.concat(
            [
                common.search(
                    prices,
                    instrument,
                    (rounding,),
                    use_volatility_scaling=scaled,
                    rebalance_frequency=frequency,
                )
                for frequency in REBALANCE_FREQUENCIES
            ],
            ignore_index=True,
        )
        model = model.sort_values("sharpe_ratio", ascending=False, kind="stable").reset_index(drop=True)
        model["sharpe_rank"] = np.arange(1, len(model) + 1)
        parts.append(model)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    prices = common.reference.cdf.load_single_future_prices(include_dbmf=False)[previous.INSTRUMENT]
    grid = search(prices)
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_FILE, index=False, float_format="%.8f")
    grid.loc[grid["sharpe_rank"].eq(1)].to_csv(BEST_FILE, index=False, float_format="%.8f")
    print(grid.to_string(index=False))
    print(f"\nFull grid: {OUTPUT_FILE}\nBest per model: {BEST_FILE}")


if __name__ == "__main__":
    main()
