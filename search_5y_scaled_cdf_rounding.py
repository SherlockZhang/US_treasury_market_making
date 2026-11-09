"""Search weekly 5Y scaled-CDF lookbacks and position rounding through 2025.

Uses the same training window and lookback grid as the rounded scaled-CDF
comparison. Rank candidates by in-sample Sharpe, without report overrides.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference


ROUNDING_INCREMENTS = (0.2, 0.3, 0.4, 0.5)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "5y_scaled_cdf_rounding_grid_2016_2025.csv"
BEST_FILE = OUTPUT_DIR / "5y_scaled_cdf_rounding_best_2016_2025.csv"


def search(
    prices: pd.Series,
    instrument: str = "5Y future",
    rounding_increments: tuple[float, ...] = ROUNDING_INCREMENTS,
    use_volatility_scaling: bool = True,
    rebalance_frequency: reference.cdf.RebalanceFrequency = "weekly",
) -> pd.DataFrame:
    training_prices = prices.loc[reference.START:reference.TRAIN_END]
    rows = []
    for lookback in reference.CDF_LOOKBACKS:
        for increment in rounding_increments:
            frame = reference.cdf.run_backtest(
                training_prices,
                lookback_days=lookback,
                rebalance_frequency=rebalance_frequency,
                position_change_threshold_fraction=reference.CDF_REBALANCE_THRESHOLD,
                signal_cutoff=reference.CDF_SIGNAL_CUTOFF,
                signal_mode="rounded",
                volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
                output_start_date=reference.START,
                position_rounding_increment=increment,
                use_volatility_scaling=use_volatility_scaling,
            )
            rows.append({
                "instrument": instrument,
                "signal_lookback_days": lookback,
                "position_rounding_increment": increment,
                "rebalance_frequency": rebalance_frequency,
                "volatility_scaling": use_volatility_scaling,
                "target_annualized_volatility_pct": (
                    reference.cdf.TARGET_ANNUALIZED_VOLATILITY * 100
                    if use_volatility_scaling else np.nan
                ),
                "volatility_lookback_days": reference.CDF_VOLATILITY_LOOKBACK,
                "signal_cutoff": reference.CDF_SIGNAL_CUTOFF,
                "rebalance_threshold_fraction": reference.CDF_REBALANCE_THRESHOLD,
                **reference.statistics(frame),
            })
    grid = pd.DataFrame(rows)
    if not np.isfinite(grid["sharpe_ratio"]).all():
        raise ValueError("A CDF candidate has an undefined Sharpe ratio.")
    grid = grid.sort_values(
        "sharpe_ratio", ascending=False, kind="stable"
    ).reset_index(drop=True)
    grid.insert(0, "sharpe_rank", np.arange(1, len(grid) + 1))
    return grid


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)["5Y future"]
    grid = search(prices)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_FILE, index=False, float_format="%.8f")
    grid.head(1).to_csv(BEST_FILE, index=False, float_format="%.8f")
    print(grid.to_string(index=False))
    print(f"\nFull grid: {OUTPUT_FILE}\nBest candidate: {BEST_FILE}")


if __name__ == "__main__":
    main()
