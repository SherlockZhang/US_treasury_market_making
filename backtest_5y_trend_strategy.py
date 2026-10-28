"""Backtest a weekly rebalanced, volatility-targeted 5Y futures trend strategy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).with_name("US_future_data.xlsx")
SHEET_NAME = "Sheet2"
OUTPUT_DIR = Path(__file__).with_name("results")
DAILY_OUTPUT_FILE = OUTPUT_DIR / "5y_trend_backtest_daily.csv"
STATS_OUTPUT_FILE = OUTPUT_DIR / "5y_trend_backtest_stats.csv"
SIMPLIFIED_STATS_OUTPUT_FILE = OUTPUT_DIR / "5y_trend_backtest_stats_simple.csv"
PLOT_OUTPUT_FILE = OUTPUT_DIR / "5y_trend_cumulative_pnl.png"

TRADING_DAYS_PER_YEAR = 252
TREND_LOOKBACK_DAYS = 252
VOLATILITY_LOOKBACK_DAYS = 63
TARGET_ANNUALIZED_VOLATILITY = 0.10


def parse_excel_dates(values: pd.Series) -> pd.Series:
    """Parse Excel dates stored as serial numbers or native date values."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(
            values,
            unit="D",
            origin="1899-12-30",
            errors="raise",
        )

    def parse_value(value: object) -> pd.Timestamp | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.to_datetime(
                value,
                unit="D",
                origin="1899-12-30",
                errors="raise",
            )
        return pd.to_datetime(value, errors="raise")

    return pd.Series(
        values.map(parse_value),
        index=values.index,
        name=values.name,
        dtype="datetime64[ns]",
    )


def load_prices() -> pd.DataFrame:
    """Load and validate the 5Y futures price series."""
    frame = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    required_columns = {"Date", "5Y future"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE.name}: "
            f"{sorted(missing_columns)}"
        )

    prices = frame.loc[:, ["Date", "5Y future"]].copy()
    prices["Date"] = parse_excel_dates(prices["Date"])
    prices["5Y future"] = pd.to_numeric(prices["5Y future"], errors="raise")
    incomplete_rows = prices.isna().any(axis=1) & prices.notna().any(axis=1)
    if incomplete_rows.any():
        raise ValueError("The 5Y futures date and price columns cannot contain blanks.")
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("No 5Y futures price observations were found.")
    if prices["Date"].duplicated().any():
        raise ValueError("The 5Y futures data contains duplicate dates.")
    if (prices["5Y future"] <= 0).any():
        raise ValueError("5Y futures prices must be positive.")

    prices = prices.sort_values("Date").set_index("Date")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("The 5Y futures dates must be in ascending order.")
    return prices.rename(columns={"5Y future": "price"})


def load_5y_yield() -> pd.Series:
    """Load the 5Y yield observations, indexed by their own observation dates."""
    frame = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    required_columns = {"Date .1", "5Y Yield"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE.name}: "
            f"{sorted(missing_columns)}"
        )

    yields = frame.loc[:, ["Date .1", "5Y Yield"]].copy()
    yields["Date .1"] = parse_excel_dates(yields["Date .1"])
    yields["5Y Yield"] = pd.to_numeric(yields["5Y Yield"], errors="raise")
    incomplete_rows = yields.isna().any(axis=1) & yields.notna().any(axis=1)
    if incomplete_rows.any():
        raise ValueError(
            "The 5Y yield date and value columns cannot contain partially blank rows."
        )
    yields = yields.dropna(how="all")
    if yields.empty:
        raise ValueError("No 5Y yield observations were found.")
    if yields["Date .1"].duplicated().any():
        raise ValueError("The 5Y yield data contains duplicate dates.")

    return (
        yields.sort_values("Date .1")
        .set_index("Date .1")["5Y Yield"]
        .rename("5y_yield_pct")
    )


def summarize_5y_yield_change(yields: pd.Series) -> dict[str, object]:
    """Return the signed first-to-last yield movement in percentage points."""
    if yields.empty:
        raise ValueError("No 5Y yield observations are available for the period.")
    yields = yields.sort_index()
    first_yield = yields.iloc[0]
    last_yield = yields.iloc[-1]
    return {
        "5y_yield_start_date": yields.index[0].date(),
        "5y_yield_end_date": yields.index[-1].date(),
        "5y_yield_start_pct": first_yield,
        "5y_yield_end_pct": last_yield,
        "5y_yield_change_percentage_points": last_yield - first_yield,
    }


def load_spx_index() -> pd.Series:
    """Load SPX index observations, indexed by their observation dates."""
    frame = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    required_columns = {"Date.2", "SPX"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE.name}: "
            f"{sorted(missing_columns)}"
        )

    spx = frame.loc[:, ["Date.2", "SPX"]].copy()
    spx["Date.2"] = parse_excel_dates(spx["Date.2"])
    spx["SPX"] = pd.to_numeric(spx["SPX"], errors="raise")
    incomplete_rows = spx.isna().any(axis=1) & spx.notna().any(axis=1)
    if incomplete_rows.any():
        raise ValueError("The SPX date and index columns cannot be partially blank.")
    spx = spx.dropna(how="all")
    if spx.empty:
        raise ValueError("No SPX index observations were found.")
    if spx["Date.2"].duplicated().any():
        raise ValueError("The SPX data contains duplicate dates.")
    if (spx["SPX"] <= 0).any():
        raise ValueError("SPX index levels must be positive.")

    return (
        spx.sort_values("Date.2")
        .set_index("Date.2")["SPX"]
        .rename("spx_index")
    )


def summarize_spx_change(spx: pd.Series) -> dict[str, object]:
    """Return the signed first-to-last SPX index-point change."""
    if spx.empty:
        raise ValueError("No SPX observations are available for the period.")
    spx = spx.sort_index()
    first_spx = spx.iloc[0]
    last_spx = spx.iloc[-1]
    return {
        "spx_start_date": spx.index[0].date(),
        "spx_end_date": spx.index[-1].date(),
        "spx_start_index": first_spx,
        "spx_end_index": last_spx,
        "spx_change_index_points": last_spx - first_spx,
    }


def run_backtest(prices: pd.DataFrame) -> pd.DataFrame:
    """Create daily strategy returns using signals rebalanced on Mondays."""
    results = prices.copy()
    results["daily_return"] = results["price"].pct_change()
    results["moving_average"] = results["price"].rolling(
        TREND_LOOKBACK_DAYS, min_periods=TREND_LOOKBACK_DAYS
    ).mean()
    results["realized_volatility"] = (
        results["daily_return"]
        .rolling(VOLATILITY_LOOKBACK_DAYS, min_periods=VOLATILITY_LOOKBACK_DAYS)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    results["signal"] = np.select(
        [
            results["price"] > results["moving_average"],
            results["price"] < results["moving_average"],
        ],
        [1.0, -1.0],
        default=0.0,
    )
    eligible = (
        results["moving_average"].notna()
        & results["realized_volatility"].gt(0)
    )
    desired_position = pd.Series(np.nan, index=results.index, dtype=float)
    desired_position.loc[eligible] = (
        results.loc[eligible, "signal"]
        * TARGET_ANNUALIZED_VOLATILITY
        / results.loc[eligible, "realized_volatility"]
    )

    monday_rebalance = results.index.dayofweek == 0
    monday_targets = desired_position.where(monday_rebalance)
    position_at_close = monday_targets.ffill().fillna(0.0)
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


def calculate_statistics(
    results: pd.DataFrame,
) -> dict[str, object]:
    """Summarize performance using strategy returns for the Sharpe ratio."""
    active_dates = results.index[results["position"].ne(0)]
    if len(active_dates) == 0:
        raise ValueError("No active positions were generated by the backtest.")
    trading = results.loc[active_dates[0] :].copy()
    if len(trading) < 2:
        raise ValueError("At least two backtest observations after entry are required.")

    returns = trading["strategy_return"]
    daily_volatility = returns.std(ddof=1)
    if not np.isfinite(daily_volatility) or daily_volatility == 0:
        raise ValueError("Strategy return volatility must be positive.")

    drawdown = results["drawdown"]
    trough_date = drawdown.idxmin()
    peak_date = results.loc[:trough_date, "wealth"].idxmax()
    recovery_dates = results.index[
        (results.index > trough_date)
        & (results["wealth"] >= results.loc[peak_date, "wealth"])
    ]
    recovery_date = recovery_dates[0].date() if len(recovery_dates) else None
    annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "period": "Full dataset",
        "instrument": "5Y future",
        "data_start_date": results.index.min().date(),
        "data_end_date": results.index.max().date(),
        "observations": len(results),
        "active_start_date": trading.index.min().date(),
        "active_end_date": trading.index.max().date(),
        "active_observations": len(trading),
        "target_annualized_volatility_pct": TARGET_ANNUALIZED_VOLATILITY * 100,
        "total_return_pct": (results["wealth"].iloc[-1] - 1) * 100,
        "annualized_return_pct": annualized_return * 100,
        "annualized_volatility_pct": annualized_volatility * 100,
        "sharpe_ratio": returns.mean() / daily_volatility * np.sqrt(
            TRADING_DAYS_PER_YEAR
        ),
        "max_drawdown_pct": drawdown.min() * 100,
        "max_drawdown_peak_date": peak_date.date(),
        "max_drawdown_trough_date": trough_date.date(),
        "max_drawdown_recovery_date": recovery_date,
        "average_absolute_position": trading["position"].abs().mean(),
        "maximum_absolute_position": trading["position"].abs().max(),
    }


def calculate_annual_statistics(
    results: pd.DataFrame,
) -> list[dict[str, object]]:
    """Calculate calendar-year metrics with return-based Sharpe ratios."""
    annual_stats = []
    for year, yearly in results.groupby(results.index.year, sort=True):
        returns = yearly["strategy_return"]
        daily_volatility = returns.std(ddof=1)
        sharpe_ratio = (
            returns.mean()
            / daily_volatility
            * np.sqrt(TRADING_DAYS_PER_YEAR)
            if np.isfinite(daily_volatility)
            and daily_volatility > 0
            else np.nan
        )

        wealth = (1 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        if drawdown.min() < 0:
            trough_date = drawdown.idxmin()
            peak_date = wealth.loc[:trough_date].idxmax()
            recovery_dates = yearly.index[
                (yearly.index > trough_date) & (wealth >= wealth.loc[peak_date])
            ]
            recovery_date = (
                recovery_dates[0].date() if len(recovery_dates) else None
            )
            peak_date_value = peak_date.date()
            trough_date_value = trough_date.date()
        else:
            peak_date_value = None
            trough_date_value = None
            recovery_date = None

        annual_stats.append(
            {
                "period": str(year),
                "instrument": "5Y future",
                "data_start_date": yearly.index.min().date(),
                "data_end_date": yearly.index.max().date(),
                "observations": len(yearly),
                "active_start_date": (
                    yearly.index[yearly["position"].ne(0)].min().date()
                    if yearly["position"].ne(0).any()
                    else None
                ),
                "active_end_date": (
                    yearly.index[yearly["position"].ne(0)].max().date()
                    if yearly["position"].ne(0).any()
                    else None
                ),
                "active_observations": int(yearly["position"].ne(0).sum()),
                "target_annualized_volatility_pct": (
                    TARGET_ANNUALIZED_VOLATILITY * 100
                ),
                "total_return_pct": (wealth.iloc[-1] - 1) * 100,
                "annualized_return_pct": (
                    returns.mean() * TRADING_DAYS_PER_YEAR * 100
                ),
                "annualized_volatility_pct": (
                    daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
                    if np.isfinite(daily_volatility)
                    else np.nan
                ),
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": drawdown.min() * 100,
                "max_drawdown_peak_date": peak_date_value,
                "max_drawdown_trough_date": trough_date_value,
                "max_drawdown_recovery_date": recovery_date,
                "average_absolute_position": yearly["position"].abs().mean(),
                "maximum_absolute_position": yearly["position"].abs().max(),
            }
        )
    return annual_stats


def save_cumulative_pnl_plot(results: pd.DataFrame) -> None:
    """Save the cumulative PnL curve for the full backtest period."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        color="#1f4e79",
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title("5Y Treasury Futures Trend Strategy — Full-Period Cumulative PnL")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(PLOT_OUTPUT_FILE, dpi=150)
    plt.close(figure)


def main() -> None:
    prices = load_prices()
    yields = load_5y_yield()
    spx = load_spx_index()
    results = run_backtest(prices)
    stats = calculate_statistics(results)
    annual_stats = calculate_annual_statistics(results)
    stats.update(summarize_5y_yield_change(yields))
    stats.update(summarize_spx_change(spx))
    for annual_stat in annual_stats:
        year = int(annual_stat["period"])
        annual_yields = yields.loc[yields.index.year == year]
        annual_stat.update(summarize_5y_yield_change(annual_yields))
        annual_spx = spx.loc[spx.index.year == year]
        annual_stat.update(summarize_spx_change(annual_spx))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(DAILY_OUTPUT_FILE, index_label="Date", float_format="%.8f")
    stats_frame = pd.DataFrame([stats, *annual_stats])
    stats_frame.to_csv(
        STATS_OUTPUT_FILE, index=False, float_format="%.6f"
    )
    stats_frame.loc[
        :,
        [
            "period",
            "sharpe_ratio",
            "max_drawdown_pct",
            "5y_yield_change_percentage_points",
            "spx_change_index_points",
        ],
    ].to_csv(SIMPLIFIED_STATS_OUTPUT_FILE, index=False, float_format="%.6f")
    save_cumulative_pnl_plot(results)

    print(
        f"Backtest period: {stats['data_start_date']} to "
        f"{stats['data_end_date']} ({stats['observations']} observations)"
    )
    print(
        f"Active trading period: {stats['active_start_date']} to "
        f"{stats['active_end_date']}"
    )
    print(f"Sharpe ratio: {stats['sharpe_ratio']:.4f}")
    print(f"Maximum drawdown: {stats['max_drawdown_pct']:.2f}%")
    print(f"Total return: {stats['total_return_pct']:.2f}%")
    print("\nCalendar-year performance:")
    for annual_stat in annual_stats:
        print(
            f"{annual_stat['period']}: return "
            f"{annual_stat['total_return_pct']:.2f}%, Sharpe "
            f"{annual_stat['sharpe_ratio']:.4f}, max drawdown "
            f"{annual_stat['max_drawdown_pct']:.2f}%, 5Y yield change "
            f"{annual_stat['5y_yield_change_percentage_points']:+.4f} "
            "percentage points"
        )
    print(f"Daily backtest saved to {DAILY_OUTPUT_FILE}")
    print(f"Statistics saved to {STATS_OUTPUT_FILE}")
    print(f"Simplified statistics saved to {SIMPLIFIED_STATS_OUTPUT_FILE}")
    print(f"Cumulative PnL plot saved to {PLOT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
