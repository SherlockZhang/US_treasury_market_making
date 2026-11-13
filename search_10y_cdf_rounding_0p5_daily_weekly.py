"""Rank 10Y CDF lookbacks separately by scaling and rebalance frequency."""

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference


INSTRUMENT = "10Y future"
ROUNDING = 0.5
TARGET_ANNUALIZED_VOLATILITY = 0.05
FREQUENCIES: tuple[reference.cdf.RebalanceFrequency, ...] = ("daily", "weekly")
OUTPUT_FILE = (
    reference.RESULTS
    / "10y_cdf_scaled_unscaled_rounding_0p5_daily_weekly_grid_2016_2025.csv"
)
BEST_FILE = (
    reference.RESULTS
    / "10y_cdf_rounding_0p5_target_vol_5pct_best_by_mode_2016_2025.csv"
)


def search(prices: pd.Series) -> pd.DataFrame:
    training_prices = prices.loc[reference.START : reference.TRAIN_END]
    parts = []
    for scaled in (True, False):
        rows = []
        for frequency in FREQUENCIES:
            for lookback in reference.CDF_LOOKBACKS:
                frame = reference.cdf.run_backtest(
                    training_prices,
                    lookback_days=lookback,
                    rebalance_frequency=frequency,
                    position_change_threshold_fraction=reference.CDF_REBALANCE_THRESHOLD,
                    signal_cutoff=reference.CDF_SIGNAL_CUTOFF,
                    signal_mode="rounded",
                    volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
                    output_start_date=reference.START,
                    position_rounding_increment=ROUNDING,
                    use_volatility_scaling=scaled,
                    target_annualized_volatility=TARGET_ANNUALIZED_VOLATILITY,
                    trading_cost_ratio=reference.cdf.TRADING_COST_RATIOS[INSTRUMENT],
                )
                stats = reference.statistics(frame)
                if not np.isfinite(stats["sharpe_ratio"]):
                    raise ValueError(
                        f"Undefined Sharpe for scaled={scaled}, frequency={frequency}, "
                        f"lookback={lookback}."
                    )
                rows.append(
                    {
                        "instrument": INSTRUMENT,
                        "signal_lookback_days": lookback,
                        "position_rounding_increment": ROUNDING,
                        "rebalance_frequency": frequency,
                        "volatility_scaling": scaled,
                        "target_annualized_volatility_pct": (
                            TARGET_ANNUALIZED_VOLATILITY * 100 if scaled else np.nan
                        ),
                        "volatility_lookback_days": reference.CDF_VOLATILITY_LOOKBACK,
                        "signal_cutoff": reference.CDF_SIGNAL_CUTOFF,
                        "rebalance_threshold_fraction": reference.CDF_REBALANCE_THRESHOLD,
                        "trading_cost_ratio": reference.cdf.TRADING_COST_RATIOS[INSTRUMENT],
                        "position_turnover": float(frame["position_change"].sum()),
                        "trading_cost_accumulated": float(frame["trading_cost"].sum()),
                        **stats,
                    }
                )
        model = pd.DataFrame(rows).sort_values(
            "sharpe_ratio", ascending=False, kind="stable"
        ).reset_index(drop=True)
        model.insert(
            0, "sharpe_rank",
            model.groupby("rebalance_frequency", sort=False).cumcount() + 1,
        )
        parts.append(model)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    grid = search(prices)
    reference.RESULTS.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_FILE, index=False, float_format="%.8f")
    grid.loc[grid["sharpe_rank"].eq(1)].to_csv(
        BEST_FILE, index=False, float_format="%.8f"
    )
    print(grid.to_string(index=False))
    print(
        f"\nRows: {len(grid)}; scaled target vol: "
        f"{TARGET_ANNUALIZED_VOLATILITY:.0%}; rounding: {ROUNDING:g}"
    )
    print(f"Full grid: {OUTPUT_FILE}\nBest per scaling and frequency: {BEST_FILE}")


if __name__ == "__main__":
    main()
