"""Generate a full-dataset Sharpe grid for normal-CDF strategy settings."""

from pathlib import Path

import pandas as pd

from backtest_2y_10y_spread_strategy import load_spread
from backtest_5y_trend_strategy import calculate_statistics, load_10y_prices
from backtest_normal_cdf_signal import (
    AVERAGED_SIGNAL_LOOKBACK_DAYS,
    NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
    STRATEGIES,
    run_backtest,
)


OUTPUT_FILE = Path(__file__).with_name(
    "results_refining_parameter"
) / "normal_cdf_sharpe_grid_full_dataset.csv"
LOOKBACK_DAYS = (32, 64, 126, 252, 504)
REBALANCE_FREQUENCIES = ("daily", "weekly", "yearly")


def main() -> None:
    instruments = (
        (
            STRATEGIES["10y_future"]["instrument"],
            load_10y_prices()["price"],
        ),
        (
            STRATEGIES["10y_2y_spread"]["instrument"],
            load_spread(require_positive_10y_minus_2y=False)[
                "weighted_spread"
            ],
        ),
    )
    rows = []
    for instrument, prices in instruments:
        for rebalance_frequency in REBALANCE_FREQUENCIES:
            row: dict[str, object] = {
                "instrument": instrument,
                "rebalance_frequency": (
                    "year" if rebalance_frequency == "yearly" else rebalance_frequency
                ),
            }
            for lookback_days in LOOKBACK_DAYS:
                results = run_backtest(
                    prices,
                    lookback_days=lookback_days,
                    rebalance_frequency=rebalance_frequency,
                    position_change_threshold_fraction=0.0,
                    signal_cutoff=0.0,
                    signal_mode="single",
                    volatility_lookback_days=NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
                )
                row[f"T={lookback_days}"] = calculate_statistics(results)[
                    "sharpe_ratio"
                ]

            average_results = run_backtest(
                prices,
                lookback_days=LOOKBACK_DAYS[0],
                rebalance_frequency=rebalance_frequency,
                position_change_threshold_fraction=0.0,
                signal_cutoff=0.0,
                signal_mode="average",
                volatility_lookback_days=NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
            )
            row[
                f"Average T ({','.join(str(days) for days in AVERAGED_SIGNAL_LOOKBACK_DAYS)})"
            ] = calculate_statistics(average_results)["sharpe_ratio"]
            rows.append(row)

    grid = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")
    print(grid.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSharpe grid CSV: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
