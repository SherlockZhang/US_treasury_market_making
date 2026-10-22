"""Backtest a weekly trend strategy on the 10Y-minus-2Y futures spread."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest_5y_trend_strategy import (
    calculate_annual_statistics,
    calculate_statistics,
    load_5y_yield,
    load_spx_index,
    parse_excel_dates,
    run_backtest,
    summarize_spx_change,
    summarize_5y_yield_change,
)


INPUT_FILE = Path(__file__).with_name("US_future_data.xlsx")
SHEET_NAME = "Sheet1"
OUTPUT_DIR = Path(__file__).with_name("results")
DAILY_OUTPUT_FILE = OUTPUT_DIR / "2y_10y_spread_backtest_daily.csv"
STATS_OUTPUT_FILE = OUTPUT_DIR / "2y_10y_spread_backtest_stats.csv"
SIMPLIFIED_STATS_OUTPUT_FILE = (
    OUTPUT_DIR / "2y_10y_spread_backtest_stats_simple.csv"
)
PLOT_OUTPUT_FILE = OUTPUT_DIR / "2y_10y_spread_cumulative_pnl.png"

TWO_YEAR_CONTRACTS = 5
TEN_YEAR_CONTRACTS = 3
INSTRUMENT_NAME = "5x2Y - 3x10Y futures basket"


def load_spread() -> pd.DataFrame:
    """Load both futures and calculate the standard and weighted price series."""
    frame = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    required_columns = {"Date.1", "2Y future", "Date ", "10Y future"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE.name}: "
            f"{sorted(missing_columns)}"
        )

    def load_leg(date_column: str, price_column: str) -> pd.Series:
        leg = frame.loc[:, [date_column, price_column]].copy()
        leg[date_column] = parse_excel_dates(leg[date_column])
        leg[price_column] = pd.to_numeric(leg[price_column], errors="raise")
        incomplete_rows = leg.isna().any(axis=1) & leg.notna().any(axis=1)
        if incomplete_rows.any():
            raise ValueError(
                f"The {price_column} date and price columns contain "
                "partially blank rows."
            )
        leg = leg.dropna(how="all")
        if leg.empty:
            raise ValueError(f"No observations found for {price_column}.")
        if leg[date_column].duplicated().any():
            raise ValueError(f"The {price_column} data contains duplicate dates.")
        if (leg[price_column] <= 0).any():
            raise ValueError(f"{price_column} prices must be positive.")
        return leg.set_index(date_column)[price_column].sort_index()

    two_year = load_leg("Date.1", "2Y future").rename("2y_future")
    ten_year = load_leg("Date ", "10Y future").rename("10y_future")
    prices = pd.concat([two_year, ten_year], axis=1, join="inner").dropna()
    if prices.empty:
        raise ValueError("No dates have both 2Y and 10Y futures prices.")

    prices["spread"] = prices["10y_future"] - prices["2y_future"]
    if (prices["spread"] <= 0).any():
        raise ValueError(
            "The 10Y-minus-2Y spread must remain positive to use "
            "percentage returns consistently."
        )
    prices["weighted_spread"] = (
        TWO_YEAR_CONTRACTS * prices["2y_future"]
        - TEN_YEAR_CONTRACTS * prices["10y_future"]
    )
    if (prices["weighted_spread"] <= 0).any():
        raise ValueError(
            "The 5x2Y-minus-3x10Y weighted futures price must remain positive "
            "to calculate percentage returns consistently."
        )
    prices.index.name = "Date"
    return prices


def calculate_spread_statistics(
    results: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    """Calculate full-period and calendar-year metrics for the spread."""
    full_stats = calculate_statistics(results)
    annual_stats = calculate_annual_statistics(results)
    full_stats["instrument"] = INSTRUMENT_NAME
    for annual_stat in annual_stats:
        annual_stat["instrument"] = INSTRUMENT_NAME
    return full_stats, annual_stats


def save_cumulative_pnl_plot(results: pd.DataFrame) -> None:
    """Save the cumulative PnL curve for the full spread backtest period."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        color="#8c4a2f",
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        "5x2Y-minus-3x10Y Treasury Futures Basket Trend Strategy — "
        "Full-Period Cumulative PnL"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(PLOT_OUTPUT_FILE, dpi=150)
    plt.close(figure)


def main() -> None:
    prices = load_spread()
    results = run_backtest(
        prices.loc[:, ["weighted_spread"]].rename(
            columns={"weighted_spread": "price"}
        )
    )
    results = results.rename(columns={"price": "weighted_spread"})
    results = results.join(prices.loc[:, ["2y_future", "10y_future"]])
    full_stats, annual_stats = calculate_spread_statistics(results)
    yields = load_5y_yield()
    spx = load_spx_index()
    full_stats.update(summarize_5y_yield_change(yields))
    full_stats.update(summarize_spx_change(spx))
    for annual_stat in annual_stats:
        year = int(annual_stat["period"])
        annual_yields = yields.loc[yields.index.year == year]
        annual_stat.update(summarize_5y_yield_change(annual_yields))
        annual_spx = spx.loc[spx.index.year == year]
        annual_stat.update(summarize_spx_change(annual_spx))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(DAILY_OUTPUT_FILE, index_label="Date", float_format="%.8f")
    stats_frame = pd.DataFrame([full_stats, *annual_stats])
    stats_frame.to_csv(STATS_OUTPUT_FILE, index=False, float_format="%.6f")
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
        f"Backtest period: {full_stats['data_start_date']} to "
        f"{full_stats['data_end_date']} ({full_stats['observations']} "
        "aligned observations)"
    )
    print(f"Sharpe ratio: {full_stats['sharpe_ratio']:.4f}")
    print(f"Maximum drawdown: {full_stats['max_drawdown_pct']:.2f}%")
    print(f"Total return: {full_stats['total_return_pct']:.2f}%")
    print("\nCalendar-year performance:")
    for annual_stat in annual_stats:
        print(
            f"{annual_stat['period']}: return "
            f"{annual_stat['total_return_pct']:.2f}%, Sharpe "
            f"{annual_stat['sharpe_ratio']:.4f}, max drawdown "
            f"{annual_stat['max_drawdown_pct']:.2f}%"
        )
    print(f"Daily backtest saved to {DAILY_OUTPUT_FILE}")
    print(f"Statistics saved to {STATS_OUTPUT_FILE}")
    print(f"Simplified statistics saved to {SIMPLIFIED_STATS_OUTPUT_FILE}")
    print(f"Cumulative PnL plot saved to {PLOT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
