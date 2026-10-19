"""Backtest a normal-CDF momentum signal on 5Y and 10Y-2Y prices."""

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_2y_10y_spread_strategy import load_spread
from backtest_5y_trend_strategy import (
    TARGET_ANNUALIZED_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_LOOKBACK_DAYS,
    calculate_annual_statistics,
    calculate_statistics,
    load_5y_yield,
    load_prices,
    load_risk_free_returns,
    load_spx_index,
    run_backtest as run_trend_backtest,
    summarize_5y_yield_change,
    summarize_spx_change,
)


OUTPUT_DIR = Path(__file__).with_name("results")
DEFAULT_LOOKBACK_DAYS = 252

STRATEGIES = {
    "5y_future": {
        "instrument": "5Y future",
        "stem": "5y_future_normal_cdf_signal",
        "color": "#245a84",
    },
    "10y_2y_spread": {
        "instrument": "10Y-2Y futures spread",
        "stem": "10y_2y_spread_normal_cdf_signal",
        "color": "#8c4a2f",
    },
}


def positive_int(value: str) -> int:
    """Parse a positive integer lookback for the command-line option."""
    try:
        days = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "lookback days must be a positive integer"
        ) from error
    if days <= 0:
        raise argparse.ArgumentTypeError("lookback days must be a positive integer")
    return days


def calculate_normal_cdf_signal(
    daily_returns: pd.Series, lookback_days: int
) -> pd.DataFrame:
    """Calculate rolling t-statistics and map them to signals in [-1, 1]."""
    if lookback_days <= 0:
        raise ValueError("Lookback days must be a positive integer.")

    rolling_returns = daily_returns.rolling(
        window=lookback_days, min_periods=lookback_days
    )
    average_return = rolling_returns.mean()
    return_volatility = rolling_returns.std(ddof=1)
    t_stat = pd.Series(
        np.sqrt(lookback_days) * average_return / return_volatility,
        index=daily_returns.index,
    )
    signal = pd.Series(
        2 * norm.cdf(t_stat) - 1, index=daily_returns.index
    )
    signal = signal.where(return_volatility.gt(0))

    return pd.DataFrame(
        {
            "average_return": average_return,
            "lookback_volatility": return_volatility,
            "t_stat": t_stat,
            "signal": signal,
        },
        index=daily_returns.index,
    )


def run_backtest(
    prices: pd.Series, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """Apply the normal-CDF signal with weekly 10% volatility-target sizing."""
    if lookback_days <= 0:
        raise ValueError("Lookback days must be a positive integer.")
    if prices.empty:
        raise ValueError("No prices are available for this backtest.")
    if prices.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Backtest prices must be finite, non-missing, and positive.")
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")

    prices = prices.sort_index()
    results = prices.rename("price").to_frame()
    dates = pd.DatetimeIndex(results.index)
    results.index = dates
    results["daily_return"] = results["price"].pct_change(fill_method=None)
    signal_data = calculate_normal_cdf_signal(
        results["daily_return"], lookback_days
    )
    results = results.join(signal_data)
    results["realized_volatility"] = (
        results["daily_return"]
        .rolling(
            VOLATILITY_LOOKBACK_DAYS,
            min_periods=VOLATILITY_LOOKBACK_DAYS,
        )
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    eligible = results["signal"].notna() & results["realized_volatility"].gt(0)
    desired_position = pd.Series(np.nan, index=results.index, dtype=float)
    desired_position.loc[eligible] = (
        results.loc[eligible, "signal"]
        * TARGET_ANNUALIZED_VOLATILITY
        / results.loc[eligible, "realized_volatility"]
    )
    monday_target = desired_position.where(dates.dayofweek == 0)
    position_at_close = monday_target.ffill().fillna(0.0)
    results["position"] = position_at_close.shift(1).fillna(0.0)
    results["strategy_return"] = (
        results["position"] * results["daily_return"]
    ).fillna(0.0)

    results["wealth"] = (1 + results["strategy_return"]).cumprod()
    if (results["wealth"] <= 0).any():
        raise ValueError(
            "The strategy's compounded wealth reached zero or less; "
            "check for extreme returns or volatility-target leverage."
        )
    results["cumulative_pnl_pct"] = (results["wealth"] - 1) * 100
    results["running_peak"] = results["wealth"].cummax()
    results["drawdown"] = results["wealth"] / results["running_peak"] - 1
    results["yearly_wealth"] = results.groupby(dates.year)[
        "strategy_return"
    ].transform(lambda returns: (1 + returns).cumprod())
    results["yearly_cumulative_pnl_pct"] = (
        results["yearly_wealth"] - 1
    ) * 100
    return results


def calculate_stats(
    results: pd.DataFrame,
    instrument: str,
    risk_free_returns: pd.Series,
    lookback_days: int,
) -> pd.DataFrame:
    """Calculate risk-free-adjusted full-period and annual statistics."""
    full_stats = calculate_statistics(results, risk_free_returns)
    annual_stats = calculate_annual_statistics(results, risk_free_returns)
    full_stats["instrument"] = instrument
    full_stats["signal_lookback_days"] = lookback_days
    for annual_stat in annual_stats:
        annual_stat["instrument"] = instrument
        annual_stat["signal_lookback_days"] = lookback_days

    yields = load_5y_yield()
    spx = load_spx_index()
    yield_dates = pd.DatetimeIndex(yields.index)
    spx_dates = pd.DatetimeIndex(spx.index)
    full_stats.update(summarize_5y_yield_change(yields))
    full_stats.update(summarize_spx_change(spx))
    for annual_stat in annual_stats:
        year = int(cast(str, annual_stat["period"]))
        annual_yields = yields.loc[yield_dates.year == year]
        annual_spx = spx.loc[spx_dates.year == year]
        annual_stat.update(summarize_5y_yield_change(annual_yields))
        annual_stat.update(summarize_spx_change(annual_spx))
    return pd.DataFrame([full_stats, *annual_stats])


def save_plots(
    results: pd.DataFrame,
    prices: pd.Series,
    instrument: str,
    color: str,
    lookback_days: int,
    plot_path: Path,
    comparison_path: Path,
) -> None:
    """Save cumulative PnL and comparison against the original trend strategy."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        color=color,
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"{instrument} Normal-CDF Signal (T={lookback_days}) — "
        "Full-Period Cumulative PnL"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    original_results = run_trend_backtest(prices.rename("price").to_frame())
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        original_results.index,
        original_results["cumulative_pnl_pct"],
        label="1-year moving-average strategy",
        linewidth=1.5,
    )
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        label=f"Normal-CDF signal (T={lookback_days})",
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(f"{instrument} Trend Strategies — Cumulative PnL Comparison")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)


def run_instrument(
    prices: pd.Series,
    strategy: dict[str, str],
    risk_free_returns: pd.Series,
    lookback_days: int,
) -> None:
    """Run one instrument and save daily, summary, and plot outputs."""
    instrument = strategy["instrument"]
    stem = f"{strategy['stem']}_{lookback_days}d"
    results = run_backtest(prices, lookback_days)
    stats_frame = calculate_stats(
        results, instrument, risk_free_returns, lookback_days
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / f"{stem}_daily.csv"
    stats_path = OUTPUT_DIR / f"{stem}_stats.csv"
    simple_stats_path = OUTPUT_DIR / f"{stem}_stats_simple.csv"
    plot_path = OUTPUT_DIR / f"{stem}_cumulative_pnl.png"
    comparison_path = OUTPUT_DIR / f"{stem}_vs_original_pnl.png"

    results.to_csv(daily_path, index_label="Date", float_format="%.8f")
    stats_frame.to_csv(stats_path, index=False, float_format="%.6f")
    stats_frame.loc[
        :,
        [
            "period",
            "signal_lookback_days",
            "sharpe_ratio",
            "max_drawdown_pct",
            "5y_yield_change_percentage_points",
            "spx_change_index_points",
        ],
    ].to_csv(simple_stats_path, index=False, float_format="%.6f")
    save_plots(
        results,
        prices,
        instrument,
        strategy["color"],
        lookback_days,
        plot_path,
        comparison_path,
    )

    full_stats = stats_frame.iloc[0]
    print(
        f"{instrument} (T={lookback_days}): "
        f"{full_stats['data_start_date']} to {full_stats['data_end_date']}, "
        f"Sharpe {full_stats['sharpe_ratio']:.4f}, max drawdown "
        f"{full_stats['max_drawdown_pct']:.2f}%, total return "
        f"{full_stats['total_return_pct']:.2f}%"
    )
    print(f"Daily results: {daily_path}")
    print(f"Stats: {stats_path}")
    print(f"Simplified stats: {simple_stats_path}")
    print(f"PnL plot: {plot_path}")
    print(f"Comparison plot: {comparison_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookback-days",
        type=positive_int,
        default=DEFAULT_LOOKBACK_DAYS,
        metavar="T",
        help=f"Rolling return lookback in trading days (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    args = parser.parse_args()

    risk_free_returns = load_risk_free_returns()
    five_year_prices = load_prices()["price"]
    spread_prices = load_spread()["spread"]
    run_instrument(
        five_year_prices,
        STRATEGIES["5y_future"],
        risk_free_returns,
        args.lookback_days,
    )
    run_instrument(
        spread_prices,
        STRATEGIES["10y_2y_spread"],
        risk_free_returns,
        args.lookback_days,
    )


if __name__ == "__main__":
    main()
