"""Calculate monthly DBMF Price returns and their positive-return win rate."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).with_name("future_price.csv")
OUTPUT_DIR = Path(__file__).with_name("component_analysis") / "results"
DEFAULT_START_DATE = pd.Timestamp("2016-01-01")
ANNUAL_STATS_START_YEAR = 2020
PRICE_COLUMN = "DBMF Price"


def parse_start_date(value: str) -> pd.Timestamp:
    """Parse a YYYY-MM-DD start date."""
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "start date must use YYYY-MM-DD format"
        ) from error
    if parsed.strftime("%Y-%m-%d") != value:
        raise argparse.ArgumentTypeError("start date must use YYYY-MM-DD format")
    return parsed


def load_dbmf_prices(start_date: pd.Timestamp) -> pd.Series:
    """Load DBMF Price using the date column two columns before its price."""
    frame = pd.read_csv(INPUT_FILE)
    columns = list(frame.columns)
    if PRICE_COLUMN not in columns:
        raise ValueError(f"Missing required {PRICE_COLUMN!r} column in {INPUT_FILE.name}.")

    price_position = columns.index(PRICE_COLUMN)
    date_position = price_position - 2
    if date_position < 0:
        raise ValueError(f"No date column two columns before {PRICE_COLUMN!r}.")
    date_column = columns[date_position]
    if not isinstance(date_column, str) or not date_column.strip().startswith("Date"):
        raise ValueError(
            f"Expected a Date column two columns before {PRICE_COLUMN!r}."
        )

    observations = frame.loc[:, [date_column, PRICE_COLUMN]].copy()
    date_values = observations[date_column]
    numeric_dates = pd.to_numeric(date_values, errors="coerce")
    numeric_mask = date_values.notna() & numeric_dates.notna()
    parsed_dates = pd.Series(
        pd.NaT,
        index=observations.index,
        dtype="datetime64[ns]",
    )
    if numeric_mask.any():
        parsed_dates.loc[numeric_mask] = pd.to_datetime(
            numeric_dates.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="raise",
        )
    text_mask = date_values.notna() & ~numeric_mask
    if text_mask.any():
        parsed_dates.loc[text_mask] = pd.to_datetime(
            date_values.loc[text_mask],
            errors="raise",
        )
    observations[date_column] = parsed_dates
    observations[PRICE_COLUMN] = pd.to_numeric(
        observations[PRICE_COLUMN],
        errors="raise",
    )

    partially_blank = observations.isna().any(axis=1) & observations.notna().any(
        axis=1
    )
    if partially_blank.any():
        raise ValueError(f"{PRICE_COLUMN} date and price columns contain blanks.")
    observations = observations.dropna(how="all")
    if observations.empty:
        raise ValueError(f"No observations found for {PRICE_COLUMN!r}.")
    if observations[date_column].duplicated().any():
        raise ValueError(f"{PRICE_COLUMN} contains duplicate dates.")
    prices = observations.sort_values(date_column).set_index(date_column)[PRICE_COLUMN]
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"{PRICE_COLUMN} values must be finite and positive.")

    prices = prices.loc[prices.index >= start_date]
    if prices.empty:
        raise ValueError(
            f"No {PRICE_COLUMN} observations are available on or after "
            f"{start_date:%Y-%m-%d}."
        )
    return prices.rename(PRICE_COLUMN)


def calculate_monthly_returns(prices: pd.Series) -> pd.DataFrame:
    """Use each month's last observed price to calculate monthly returns."""
    monthly_prices = prices.resample("ME").last().dropna()
    monthly_observation_dates = prices.index.to_series().resample("ME").last().dropna()
    monthly_returns = monthly_prices.pct_change(fill_method=None).mul(100).dropna()
    if monthly_returns.empty:
        raise ValueError("At least two monthly DBMF prices are required.")

    results = pd.DataFrame(
        {
            "month_end_price_date": monthly_observation_dates.reindex(
                monthly_returns.index
            ).to_numpy(),
            "month_end_price": monthly_prices.reindex(monthly_returns.index).to_numpy(),
            "monthly_return_pct": monthly_returns.to_numpy(),
        },
        index=monthly_returns.index,
    )
    results.index.name = "month_end"
    results["month"] = results.index.strftime("%Y-%m")
    results["positive_return"] = results["monthly_return_pct"] > 0
    return results.loc[
        :,
        [
            "month",
            "month_end_price_date",
            "month_end_price",
            "monthly_return_pct",
            "positive_return",
        ],
    ]


def calculate_annual_statistics(
    prices: pd.Series,
    start_year: int = ANNUAL_STATS_START_YEAR,
) -> pd.DataFrame:
    """Calculate calendar-year return, annualized volatility, Sharpe, and drawdown."""
    daily_returns = prices.pct_change(fill_method=None).dropna()
    annual_rows = []
    for year, returns in daily_returns.groupby(daily_returns.index.year):
        if year < start_year:
            continue
        wealth = (1 + returns).cumprod()
        wealth_with_initial_value = pd.concat(
            [pd.Series([1.0], index=[returns.index[0] - pd.Timedelta(days=1)]), wealth]
        )
        drawdown = (
            wealth_with_initial_value / wealth_with_initial_value.cummax() - 1
        )
        daily_volatility = returns.std(ddof=1)
        sharpe_ratio = (
            returns.mean() / daily_volatility * np.sqrt(252)
            if np.isfinite(daily_volatility) and daily_volatility > 0
            else np.nan
        )
        annual_rows.append(
            {
                "year": year,
                "start_date": returns.index[0].date(),
                "end_date": returns.index[-1].date(),
                "trading_days": len(returns),
                "annual_return_pct": ((1 + returns).prod() - 1) * 100,
                "annual_volatility_pct": daily_volatility * np.sqrt(252) * 100,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": drawdown.min() * 100,
            }
        )
    if not annual_rows:
        raise ValueError(f"No daily DBMF returns are available from {start_year}.")
    return pd.DataFrame(annual_rows)


def save_monthly_returns_plot(
    monthly_returns: pd.DataFrame,
    win_rate: float,
    output_path: Path,
) -> None:
    """Plot each monthly DBMF return, distinguishing wins from losses."""
    returns = monthly_returns["monthly_return_pct"]
    colors = np.where(returns > 0, "#2E7D32", "#C62828")
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.bar(returns.index, returns, width=24, color=colors, alpha=0.85)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title(
        f"DBMF Monthly Returns ({monthly_returns['month'].iloc[0]} to "
        f"{monthly_returns['month'].iloc[-1]}) — Win Rate {win_rate:.2%}"
    )
    axis.set_xlabel("Month")
    axis.set_ylabel("Monthly return (%)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=parse_start_date,
        default=DEFAULT_START_DATE,
        help=(
            "First DBMF observation to include (YYYY-MM-DD; "
            f"default: {DEFAULT_START_DATE:%Y-%m-%d})."
        ),
    )
    args = parser.parse_args()

    prices = load_dbmf_prices(args.start_date)
    monthly_returns = calculate_monthly_returns(prices)
    positive_count = int(monthly_returns["positive_return"].sum())
    total_count = len(monthly_returns)
    win_rate = positive_count / total_count
    annual_statistics = calculate_annual_statistics(prices)
    start_tag = args.start_date.strftime("%Y%m%d")
    monthly_returns_file = (
        OUTPUT_DIR / f"dbmf_monthly_returns_from{start_tag}.csv"
    )
    summary_file = (
        OUTPUT_DIR / f"dbmf_monthly_returns_summary_from{start_tag}.csv"
    )
    plot_file = OUTPUT_DIR / f"dbmf_monthly_returns_from{start_tag}.png"
    annual_stats_file = (
        OUTPUT_DIR / f"dbmf_annual_statistics_from{ANNUAL_STATS_START_YEAR}.csv"
    )
    summary = pd.DataFrame(
        [
            {
                "requested_start_date": args.start_date.date(),
                "first_price_date": prices.index[0].date(),
                "last_price_date": prices.index[-1].date(),
                "first_monthly_return": monthly_returns["month"].iloc[0],
                "last_monthly_return": monthly_returns["month"].iloc[-1],
                "positive_return_count": positive_count,
                "total_monthly_return_count": total_count,
                "win_rate": win_rate,
                "win_rate_pct": win_rate * 100,
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly_returns.to_csv(
        monthly_returns_file,
        index_label="month_end",
        float_format="%.8f",
    )
    summary.to_csv(summary_file, index=False, float_format="%.6f")
    annual_statistics.to_csv(
        annual_stats_file,
        index=False,
        float_format="%.6f",
    )
    save_monthly_returns_plot(monthly_returns, win_rate, plot_file)

    print(
        f"DBMF Price observations: {prices.index[0]:%Y-%m-%d} to "
        f"{prices.index[-1]:%Y-%m-%d}"
    )
    if prices.index[0] > args.start_date:
        print(
            f"No DBMF Price data is available from {args.start_date:%Y-%m-%d}; "
            f"the available history starts on {prices.index[0]:%Y-%m-%d}."
        )
    print(
        f"Monthly returns: {monthly_returns['month'].iloc[0]} to "
        f"{monthly_returns['month'].iloc[-1]}"
    )
    print(
        f"Positive months: {positive_count}/{total_count}; "
        f"win rate: {win_rate:.2%}"
    )
    print(f"Monthly returns: {monthly_returns_file}")
    print(f"Summary: {summary_file}")
    print(f"Monthly return plot: {plot_file}")
    print(f"Annual statistics: {annual_stats_file}")


if __name__ == "__main__":
    main()
