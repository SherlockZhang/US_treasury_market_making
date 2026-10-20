"""Backtest factor-modified trend strategies on 5Y and 10Y-2Y prices."""

import argparse
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_2y_10y_spread_strategy import load_spread
from backtest_5y_trend_strategy import (
    TARGET_ANNUALIZED_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    TREND_LOOKBACK_DAYS,
    VOLATILITY_LOOKBACK_DAYS,
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
INPUT_FILE = Path(__file__).with_name("US_future_data.xlsx")
FactorName = Literal["spx", "cpi"]
STRATEGIES = {
    "5y_future": {
        "instrument": "5Y future",
        "stem": "factor_modified_5y_future",
        "color": "#245a84",
    },
    "10y_2y_spread": {
        "instrument": "10Y-2Y futures spread",
        "stem": "factor_modified_10y_2y_spread",
        "color": "#8c4a2f",
    },
}


def calculate_spx_12_month_return(
    dates: pd.DatetimeIndex, spx_index: pd.Series
) -> pd.Series:
    """Calculate trailing returns using the latest SPX values at calendar lags."""
    spx_index = spx_index.sort_index()
    if spx_index.empty:
        raise ValueError("No SPX observations are available.")
    if spx_index.index.has_duplicates:
        raise ValueError("SPX data contains duplicate dates.")
    if (spx_index <= 0).any():
        raise ValueError("SPX index levels must be positive.")

    spx_dates = spx_index.index
    spx_values = spx_index.to_numpy(dtype=float)
    returns = np.full(len(dates), np.nan, dtype=float)
    for position, date in enumerate(dates):
        current_position = spx_dates.searchsorted(date, side="right") - 1
        lookback_date = date - pd.DateOffset(years=1)
        prior_position = spx_dates.searchsorted(lookback_date, side="right") - 1
        if current_position >= 0 and prior_position >= 0:
            returns[position] = (
                spx_values[current_position] / spx_values[prior_position] - 1
            )

    return pd.Series(returns, index=dates, name="spx_12_month_return")


def load_cpi_index() -> pd.Series:
    """Load monthly CPI observations from the main workbook."""
    frame = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")
    required_columns = {"Date.3", "CPI"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE.name}: "
            f"{sorted(missing_columns)}"
        )

    cpi = frame.loc[:, ["Date.3", "CPI"]].copy()
    cpi["Date.3"] = pd.to_datetime(cpi["Date.3"], errors="raise")
    cpi["CPI"] = pd.to_numeric(cpi["CPI"], errors="raise")
    incomplete_rows = cpi.isna().any(axis=1) & cpi.notna().any(axis=1)
    if incomplete_rows.any():
        raise ValueError("The CPI date and value columns cannot be partially blank.")
    cpi = cpi.dropna(how="all")
    if cpi.empty:
        raise ValueError("No CPI observations were found.")
    if cpi["Date.3"].duplicated().any():
        raise ValueError("CPI data contains duplicate dates.")

    return (
        cpi.sort_values("Date.3")
        .set_index("Date.3")["CPI"]
        .rename("cpi")
    )


def align_cpi_scores(
    dates: pd.DatetimeIndex, cpi: pd.Series
) -> pd.DataFrame:
    """Align the latest available CPI score and its trailing six-release mean."""
    cpi = cpi.sort_index()
    if cpi.empty:
        raise ValueError("No CPI observations are available.")
    if cpi.index.has_duplicates:
        raise ValueError("CPI data contains duplicate dates.")
    if cpi.isna().any():
        raise ValueError("CPI observations cannot contain missing values.")

    cpi_score = -(cpi - 2).clip(lower=0)
    cpi_score_mean = cpi_score.rolling(window=6, min_periods=6).mean()
    source_positions = cpi.index.searchsorted(dates, side="right") - 1
    available = source_positions >= 0

    aligned_values = {
        "cpi_observation_date": np.full(
            len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        ),
        "cpi": np.full(len(dates), np.nan),
        "cpi_score": np.full(len(dates), np.nan),
        "cpi_6_month_score_mean": np.full(len(dates), np.nan),
    }
    if available.any():
        observation_positions = source_positions[available]
        aligned_values["cpi_observation_date"][available] = cpi.index.to_numpy()[
            observation_positions
        ]
        aligned_values["cpi"][available] = cpi.to_numpy(dtype=float)[
            observation_positions
        ]
        aligned_values["cpi_score"][available] = cpi_score.to_numpy(dtype=float)[
            observation_positions
        ]
        aligned_values["cpi_6_month_score_mean"][available] = (
            cpi_score_mean.to_numpy(dtype=float)[observation_positions]
        )
    return pd.DataFrame(aligned_values, index=dates)


def run_backtest(
    prices: pd.Series, factor_data: pd.Series, factor: FactorName
) -> pd.DataFrame:
    """Apply a selected factor modifier to the trend signal and size by volatility."""
    if factor not in ("spx", "cpi"):
        raise ValueError(f"Unsupported factor: {factor}")
    if prices.empty:
        raise ValueError("No prices are available for this backtest.")
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Backtest prices must be non-missing and positive.")
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")

    prices = prices.sort_index()
    dates = pd.DatetimeIndex(prices.index)
    results = prices.rename("price").to_frame()
    results["daily_return"] = results["price"].pct_change()
    results["moving_average"] = results["price"].rolling(
        TREND_LOOKBACK_DAYS, min_periods=TREND_LOOKBACK_DAYS
    ).mean()
    results["signal"] = np.select(
        [
            results["price"] > results["moving_average"],
            results["price"] < results["moving_average"],
        ],
        [1.0, -1.0],
        default=0.0,
    )
    if factor == "spx":
        results["spx_12_month_return"] = calculate_spx_12_month_return(
            dates, factor_data
        )
        results["factor_multiplier"] = np.where(
            (results["spx_12_month_return"] < -0.1)
            & (results["signal"] > 0),
            2.0,
            1.0,
        )
        factor_available = results["spx_12_month_return"].notna()
    else:
        results = results.join(align_cpi_scores(dates, factor_data))
        results["factor_multiplier"] = np.where(
            results["cpi_score"] < results["cpi_6_month_score_mean"],
            1.05,
            1.0,
        )
        factor_available = results["cpi_6_month_score_mean"].notna()
    results["modified_signal"] = (
        results["signal"] * results["factor_multiplier"]
    )
    results["realized_volatility"] = (
        results["daily_return"]
        .rolling(VOLATILITY_LOOKBACK_DAYS, min_periods=VOLATILITY_LOOKBACK_DAYS)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    eligible = (
        results["moving_average"].notna()
        & factor_available
        & results["realized_volatility"].gt(0)
    )
    desired_position = pd.Series(np.nan, index=dates, dtype=float)
    desired_position.loc[eligible] = (
        results.loc[eligible, "modified_signal"]
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
    factor: FactorName,
) -> pd.DataFrame:
    """Calculate full-period and annual stats plus 5Y yield and SPX changes."""
    full_stats = calculate_statistics(results)
    annual_stats = calculate_annual_statistics(results)
    full_stats["instrument"] = instrument
    full_stats["factor"] = factor
    for annual_stat in annual_stats:
        annual_stat["instrument"] = instrument
        annual_stat["factor"] = factor

    yields = load_5y_yield()
    spx_index = load_spx_index()
    yield_dates = pd.DatetimeIndex(yields.index)
    spx_dates = pd.DatetimeIndex(spx_index.index)
    full_stats.update(summarize_5y_yield_change(yields))
    full_stats.update(summarize_spx_change(spx_index))
    for annual_stat in annual_stats:
        year = int(cast(int, annual_stat["period"]))
        annual_yields = yields.loc[yield_dates.year == year]
        annual_spx = spx_index.loc[spx_dates.year == year]
        annual_stat.update(summarize_5y_yield_change(annual_yields))
        annual_stat.update(summarize_spx_change(annual_spx))
    return pd.DataFrame([full_stats, *annual_stats])


def save_pnl_plot(
    results: pd.DataFrame,
    prices: pd.Series,
    path: Path,
    comparison_path: Path,
    instrument: str,
    color: str,
    factor: FactorName,
) -> None:
    """Save cumulative PnL and a comparison with the original trend strategy."""
    previous_results = run_trend_backtest(prices.rename("price").to_frame())

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        color=color,
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"{instrument} {factor.upper()}-Modified — Full-Period Cumulative PnL"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        previous_results.index,
        previous_results["cumulative_pnl_pct"],
        label="Original 1-year moving-average strategy",
        linewidth=1.5,
    )
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        label=f"{factor.upper()}-modified strategy",
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"{instrument} Trend Strategies ({factor.upper()} factor) — "
        "Cumulative PnL Comparison"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)


def run_instrument(
    prices: pd.Series,
    factor_data: pd.Series,
    strategy: dict[str, str],
    factor: FactorName,
) -> None:
    """Backtest one instrument and save daily, summary, and plot outputs."""
    instrument = strategy["instrument"]
    stem = strategy["stem"] if factor == "spx" else f"cpi_{strategy['stem']}"
    results = run_backtest(prices, factor_data, factor)
    stats_frame = calculate_stats(results, instrument, factor)

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
            "factor",
            "sharpe_ratio",
            "max_drawdown_pct",
            "5y_yield_change_percentage_points",
            "spx_change_index_points",
        ],
    ].to_csv(simple_stats_path, index=False, float_format="%.6f")
    save_pnl_plot(
        results,
        prices,
        plot_path,
        comparison_path,
        instrument,
        strategy["color"],
        factor,
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
    print(f"Comparison plot: {comparison_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factor",
        choices=("spx", "cpi"),
        default="spx",
        help="Select SPX 12-month return or monthly CPI as the signal factor.",
    )
    factor: FactorName = parser.parse_args().factor
    factor_data = load_spx_index() if factor == "spx" else load_cpi_index()
    five_year_prices = load_prices()["price"]
    spread_prices = load_spread()["spread"]
    run_instrument(
        five_year_prices, factor_data, STRATEGIES["5y_future"], factor
    )
    run_instrument(
        spread_prices, factor_data, STRATEGIES["10y_2y_spread"], factor
    )


if __name__ == "__main__":
    main()
