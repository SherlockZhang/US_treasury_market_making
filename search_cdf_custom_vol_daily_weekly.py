"""Search scaled and unscaled CDF models with instrument-specific sizing."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference


CONFIGURATIONS = {
    "2Y future": (0.015, 1.0),
    "5Y future": (0.03, 0.5),
    "10Y future": (0.05, 0.5),
}
FREQUENCIES: tuple[reference.cdf.RebalanceFrequency, ...] = ("daily", "weekly")
OUTPUT_DIR = reference.RESULTS / "cdf_custom_vol_daily_weekly_grid_2016_2025"


def search(
    prices_by_instrument: dict[str, pd.Series],
    rounding_increments: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Rank each instrument and scaling mode on 2016-2025 net Sharpe."""
    rows = []
    for instrument, prices in prices_by_instrument.items():
        target, default_rounding = CONFIGURATIONS[instrument]
        cost_ratio = reference.cdf.TRADING_COST_RATIOS[instrument]
        training_prices = prices.loc[reference.START:reference.TRAIN_END].rename(instrument)
        for scaled in (True, False):
            for frequency in FREQUENCIES:
                increments = rounding_increments if rounding_increments is not None else (default_rounding,)
                for lookback, rounding in product(reference.CDF_LOOKBACKS, increments):
                    frame = reference.cdf.run_backtest(
                        training_prices,
                        lookback_days=lookback,
                        rebalance_frequency=frequency,
                        position_change_threshold_fraction=0.0,
                        signal_cutoff=0.0,
                        signal_mode="rounded",
                        volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
                        output_start_date=reference.START,
                        position_rounding_increment=rounding,
                        use_volatility_scaling=scaled,
                        target_annualized_volatility=target,
                        trading_cost_ratio=cost_ratio,
                    )
                    stats = reference.statistics(frame)
                    sharpe = stats["sharpe_ratio"]
                    volatility = frame["strategy_return"].std(ddof=1)
                    if not np.isfinite(sharpe) and (not np.isfinite(volatility) or volatility != 0):
                        raise ValueError(
                            f"Unexpected undefined Sharpe for {instrument}, scaled={scaled}, "
                            f"frequency={frequency}, T={lookback}, rounding={rounding}."
                        )
                    rows.append({
                        "instrument": instrument,
                        "volatility_scaling": scaled,
                        "rebalance_frequency": frequency,
                        "signal_lookback_days": lookback,
                        "volatility_lookback_days": reference.CDF_VOLATILITY_LOOKBACK,
                        "target_annualized_volatility_pct": target * 100 if scaled else np.nan,
                        "position_rounding_increment": rounding,
                        "signal_cutoff": 0.0,
                        "rebalance_threshold_fraction": 0.0,
                        "trading_cost_ratio": cost_ratio,
                        "position_turnover": float(frame["position_change"].sum()),
                        "trading_cost_accumulated": float(frame["trading_cost"].sum()),
                        "sharpe_status": "ok" if np.isfinite(sharpe) else "undefined_zero_volatility",
                        **stats,
                    })
    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        ["instrument", "volatility_scaling", "sharpe_ratio"],
        ascending=[True, False, False], kind="stable",
    ).reset_index(drop=True)
    grid.insert(
        0, "sharpe_rank",
        grid["sharpe_ratio"].notna().groupby(
            [grid["instrument"], grid["volatility_scaling"]]
        ).cumsum().where(grid["sharpe_ratio"].notna()).astype("Int64"),
    )
    grid["selected_best"] = grid["sharpe_rank"].eq(1).fillna(False)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    grid = search({instrument: prices[instrument] for instrument in CONFIGURATIONS})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "parameter_grid.csv", index=False, float_format="%.10f")
    best = grid.loc[grid["selected_best"]]
    best.to_csv(args.output_dir / "best_parameters.csv", index=False, float_format="%.10f")
    print(best.to_string(index=False))
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
