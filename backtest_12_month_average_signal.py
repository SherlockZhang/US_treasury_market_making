"""Backtest 12-month average-signal strategies on 5Y and 10Y-2Y prices."""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_2y_10y_spread_strategy import load_spread
from backtest_5y_trend_strategy import (
    calculate_annual_statistics,
    calculate_statistics,
    load_5y_yield,
    load_prices,
    load_spx_index,
    run_backtest as run_trend_backtest,
    summarize_5y_yield_change,
    summarize_spx_change,
)


OUTPUT_DIR = Path(__file__).with_name("results")
TRADING_DAYS_PER_YEAR = 252
VOLATILITY_LOOKBACK_DAYS = 63
TARGET_ANNUALIZED_VOLATILITY = 0.10
SIGNAL_LOOKBACK_MONTHS = 12

STRATEGIES = {
    "5y_future": {
        "instrument": "5Y future",
        "daily_file": "5y_future_12_month_average_signal_daily.csv",
        "stats_file": "5y_future_12_month_average_signal_stats.csv",
        "simple_stats_file": "5y_future_12_month_average_signal_stats_simple.csv",
        "plot_file": "5y_future_12_month_average_signal_cumulative_pnl.png",
        "comparison_plot_file": (
            "5y_future_trend_vs_12_month_average_signal_pnl.png"
        ),
    },
    "10y_2y_spread": {
        "instrument": "10Y-2Y futures spread",
        "daily_file": "10y_2y_spread_12_month_average_signal_daily.csv",
        "stats_file": "10y_2y_spread_12_month_average_signal_stats.csv",
        "simple_stats_file": (
            "10y_2y_spread_12_month_average_signal_stats_simple.csv"
        ),
        "plot_file": (
            "10y_2y_spread_12_month_average_signal_cumulative_pnl.png"
        ),
        "comparison_plot_file": (
            "10y_2y_spread_trend_vs_12_month_average_signal_pnl.png"
        ),
    },
}


def calculate_12_month_average_signal(prices: pd.Series) -> pd.Series:
    """Average 12 directional signals using prior observations at calendar lags."""
    prices = prices.sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")

    dates = prices.index
    price_values = prices.to_numpy(dtype=float)
    signals = np.full(len(prices), np.nan, dtype=float)
    for current_position, current_date in enumerate(dates):
        current_price = price_values[current_position]
        lag_signals = []
        for months in range(1, SIGNAL_LOOKBACK_MONTHS + 1):
            lookback_date = current_date - pd.DateOffset(months=months)
            lag_position = dates.searchsorted(lookback_date, side="right") - 1
            if lag_position < 0:
                break
            lag_signals.append(
                1.0 if current_price > price_values[lag_position] else -1.0
            )
        if len(lag_signals) == SIGNAL_LOOKBACK_MONTHS:
            signals[current_position] = sum(lag_signals) / SIGNAL_LOOKBACK_MONTHS

    return pd.Series(signals, index=prices.index, name="average_signal")


def run_backtest(prices: pd.Series) -> pd.DataFrame:
    """Apply weekly positions and the existing 10% volatility target."""
    if prices.empty:
        raise ValueError("No prices are available for this backtest.")
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Backtest prices must be non-missing and positive.")

    results = prices.rename("price").to_frame()
    results["daily_return"] = results["price"].pct_change()
    results["average_signal"] = calculate_12_month_average_signal(results["price"])
    results["realized_volatility"] = (
        results["daily_return"]
        .rolling(VOLATILITY_LOOKBACK_DAYS, min_periods=VOLATILITY_LOOKBACK_DAYS)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    eligible = (
        results["average_signal"].notna()
        & results["realized_volatility"].gt(0)
    )
    desired_position = pd.Series(np.nan, index=results.index, dtype=float)
    desired_position.loc[eligible] = (
        results.loc[eligible, "average_signal"]
        * TARGET_ANNUALIZED_VOLATILITY
        / results.loc[eligible, "realized_volatility"]
    )
    monday_target = desired_position.where(results.index.dayofweek == 0)
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
    results["yearly_wealth"] = results.groupby(results.index.year)[
        "strategy_return"
    ].transform(lambda returns: (1 + returns).cumprod())
    results["yearly_cumulative_pnl_pct"] = (
        results["yearly_wealth"] - 1
    ) * 100
    return results


def calculate_stats(
    results: pd.DataFrame, instrument: str
) -> pd.DataFrame:
    """Calculate full-period and annual stats with yield and SPX movements."""
    full_stats = calculate_statistics(results)
    annual_stats = calculate_annual_statistics(results)
    full_stats["instrument"] = instrument
    for annual_stat in annual_stats:
        annual_stat["instrument"] = instrument

    yields = load_5y_yield()
    spx = load_spx_index()
    full_stats.update(summarize_5y_yield_change(yields))
    full_stats.update(summarize_spx_change(spx))
    for annual_stat in annual_stats:
        year = int(annual_stat["period"])
        annual_stats_yields = yields.loc[yields.index.year == year]
        annual_stats_spx = spx.loc[spx.index.year == year]
        annual_stat.update(summarize_5y_yield_change(annual_stats_yields))
        annual_stat.update(summarize_spx_change(annual_stats_spx))
    return pd.DataFrame([full_stats, *annual_stats])


def save_plot(results: pd.DataFrame, path: Path, instrument: str) -> None:
    """Save the strategy's full-period cumulative PnL chart."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(results.index, results["cumulative_pnl_pct"], linewidth=1.5)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"{instrument} 12-Month Average Signal — Full-Period Cumulative PnL"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_comparison_plot(
    results: pd.DataFrame,
    prices: pd.Series,
    path: Path,
    instrument: str,
) -> None:
    """Compare the one-year moving-average and 12-month averaged signals."""
    import matplotlib.pyplot as plt

    previous_results = run_trend_backtest(
        prices.rename("price").to_frame()
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        previous_results.index,
        previous_results["cumulative_pnl_pct"],
        label="1-year moving-average signal",
        linewidth=1.5,
    )
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        label="12-month average signal",
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(f"{instrument} Trend Strategies — Cumulative PnL Comparison")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_instrument(
    prices: pd.Series, strategy: dict[str, str]
) -> None:
    """Backtest one instrument and save its daily, summary, and plot outputs."""
    instrument = strategy["instrument"]
    results = run_backtest(prices)
    stats_frame = calculate_stats(results, instrument)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / strategy["daily_file"]
    stats_path = OUTPUT_DIR / strategy["stats_file"]
    simple_stats_path = OUTPUT_DIR / strategy["simple_stats_file"]
    plot_path = OUTPUT_DIR / strategy["plot_file"]
    comparison_plot_path = OUTPUT_DIR / strategy["comparison_plot_file"]

    results.to_csv(daily_path, index_label="Date", float_format="%.8f")
    stats_frame.to_csv(stats_path, index=False, float_format="%.6f")
    stats_frame.loc[
        :,
        [
            "period",
            "sharpe_ratio",
            "max_drawdown_pct",
            "5y_yield_change_percentage_points",
            "spx_change_index_points",
        ],
    ].to_csv(simple_stats_path, index=False, float_format="%.6f")
    save_plot(results, plot_path, instrument)
    save_comparison_plot(
        results, prices, comparison_plot_path, instrument
    )

    full_stats = stats_frame.iloc[0]
    print(
        f"{instrument}: {full_stats['data_start_date']} to "
        f"{full_stats['data_end_date']}, Sharpe "
        f"{full_stats['sharpe_ratio']:.4f}, max drawdown "
        f"{full_stats['max_drawdown_pct']:.2f}%, total return "
        f"{full_stats['total_return_pct']:.2f}%"
    )
    print(f"Daily results: {daily_path}")
    print(f"Stats: {stats_path}")
    print(f"Simplified stats: {simple_stats_path}")
    print(f"PnL plot: {plot_path}")
    print(f"Strategy comparison plot: {comparison_plot_path}")


def main() -> None:
    five_year_prices = load_prices()["price"]
    spread_prices = load_spread()["spread"]
    run_instrument(five_year_prices, STRATEGIES["5y_future"])
    run_instrument(spread_prices, STRATEGIES["10y_2y_spread"])


if __name__ == "__main__":
    main()
