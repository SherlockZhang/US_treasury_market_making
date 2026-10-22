"""Analyze ticker-return correlations with the 5Y future and weighted 2Y/10Y basket."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).with_name("market_data.xlsx")
SHEET_NAME = "Sheet1"
OUTPUT_DIR = Path(__file__).with_name("results")
CORRELATION_OUTPUT_FILE = OUTPUT_DIR / "market_data_target_return_correlations.csv"
ROLLING_WINDOW_OBSERVATIONS = 252

FIVE_YEAR_TARGET = "5Y future"
WEIGHTED_BASKET_TARGET = "5x2Y - 3x10Y futures basket"
TWO_YEAR_TICKER = "2Y future"
TEN_YEAR_TICKER = "10Y future"
FUTURES_TICKERS = {
    FIVE_YEAR_TARGET,
    TWO_YEAR_TICKER,
    TEN_YEAR_TICKER,
}


def load_ticker_series() -> dict[str, pd.Series]:
    """Load ticker series from adjacent date/value columns."""
    frame = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    prices: dict[str, pd.Series] = {}
    columns = list(frame.columns)

    for column_index, ticker in enumerate(columns):
        if not isinstance(ticker, str) or ticker.startswith("Unnamed:"):
            continue
        if column_index == 0:
            continue
        date_column = columns[column_index - 1]
        if not isinstance(date_column, str) or not date_column.strip().startswith(
            "Date"
        ):
            continue
        ticker_frame = frame.loc[:, [date_column, ticker]].copy()
        ticker_frame[date_column] = pd.to_datetime(
            ticker_frame[date_column], errors="raise"
        )
        ticker_frame[ticker] = pd.to_numeric(ticker_frame[ticker], errors="raise")
        incomplete_rows = ticker_frame.isna().any(axis=1) & ticker_frame.notna().any(
            axis=1
        )
        if incomplete_rows.any():
            raise ValueError(
                f"{ticker} has rows with only a date or only a price."
            )
        ticker_frame = ticker_frame.dropna(how="all")
        if ticker_frame.empty:
            raise ValueError(f"No observations found for {ticker}.")
        if ticker_frame[date_column].duplicated().any():
            raise ValueError(f"{ticker} contains duplicate dates.")
        if not np.isfinite(ticker_frame[ticker]).all():
            raise ValueError(f"{ticker} values must be finite.")
        prices[ticker] = (
            ticker_frame.sort_values(date_column)
            .set_index(date_column)[ticker]
            .rename(ticker)
        )

    required_tickers = {
        FIVE_YEAR_TARGET,
        TWO_YEAR_TICKER,
        TEN_YEAR_TICKER,
    }
    missing_tickers = required_tickers.difference(prices)
    if missing_tickers:
        raise ValueError(
            f"Missing required target ticker data: {sorted(missing_tickers)}"
        )
    return prices


def build_target_prices(prices: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Create target price series, aligning the two futures legs by date."""
    two_year, ten_year = prices[TWO_YEAR_TICKER].align(
        prices[TEN_YEAR_TICKER], join="inner"
    )
    weighted_basket = 5 * two_year - 3 * ten_year
    if weighted_basket.empty:
        raise ValueError("The 2Y and 10Y futures have no overlapping dates.")
    if not np.isfinite(weighted_basket).all() or (weighted_basket <= 0).any():
        raise ValueError(
            "The 5x2Y-minus-3x10Y basket must have finite positive values "
            "for percentage returns."
        )
    weighted_basket.name = WEIGHTED_BASKET_TARGET
    return {
        FIVE_YEAR_TARGET: prices[FIVE_YEAR_TARGET],
        WEIGHTED_BASKET_TARGET: weighted_basket,
    }


def calculate_returns(
    prices: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Calculate percentage returns separately for each ticker price series."""
    returns = {
        ticker: series.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        for ticker, series in prices.items()
    }
    return returns


def correlation_for_pair(first: pd.Series, second: pd.Series) -> float:
    paired_returns = pd.concat([first, second], axis=1, join="inner").dropna()
    if len(paired_returns) < 2:
        raise ValueError("At least two paired return observations are required.")
    if (paired_returns.std(ddof=1) == 0).any():
        raise ValueError("Both return series must have positive variance.")
    correlation = paired_returns.iloc[:, 0].corr(paired_returns.iloc[:, 1])
    if not np.isfinite(correlation):
        raise ValueError("The Pearson correlation could not be calculated.")
    return float(correlation)


def calculate_target_correlations(
    ticker_returns: dict[str, pd.Series],
    target_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """Calculate ticker-to-target correlations, omitting the target pair."""
    correlations = pd.DataFrame(
        index=list(ticker_returns),
        columns=list(target_returns),
        dtype=float,
    )
    for ticker, returns in ticker_returns.items():
        for target, target_return in target_returns.items():
            if (
                ticker in FUTURES_TICKERS
                and (
                    target == WEIGHTED_BASKET_TARGET
                    and ticker == FIVE_YEAR_TARGET
                    or target == FIVE_YEAR_TARGET
                    and ticker in FUTURES_TICKERS
                )
            ):
                continue
            correlations.loc[ticker, target] = correlation_for_pair(
                returns, target_return
            )
    correlations.index.name = "ticker"
    return correlations


def save_rolling_correlation_plot(
    ticker_returns: dict[str, pd.Series],
    target_return: pd.Series,
    target_name: str,
    output_path: Path,
) -> None:
    """Plot each eligible ticker's rolling correlation with one target."""
    figure, axis = plt.subplots(figsize=(15, 8))
    plotted = 0
    for ticker, returns in ticker_returns.items():
        if ticker in FUTURES_TICKERS:
            continue
        paired_returns = pd.concat(
            [returns.rename(ticker), target_return.rename(target_name)],
            axis=1,
            join="inner",
        ).dropna()
        if len(paired_returns) < ROLLING_WINDOW_OBSERVATIONS:
            continue
        rolling_correlation = paired_returns.iloc[:, 0].rolling(
            window=ROLLING_WINDOW_OBSERVATIONS,
            min_periods=ROLLING_WINDOW_OBSERVATIONS,
        ).corr(paired_returns.iloc[:, 1])
        if rolling_correlation.notna().any():
            axis.plot(
                rolling_correlation.index,
                rolling_correlation,
                linewidth=1.0,
                label=ticker,
            )
            plotted += 1

    if plotted == 0:
        plt.close(figure)
        raise ValueError(
            f"No ticker has enough paired observations for a "
            f"{ROLLING_WINDOW_OBSERVATIONS}-observation rolling correlation "
            f"with {target_name}."
        )

    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"252-Observation Rolling Return Correlation with {target_name}"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Pearson correlation")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    figure.tight_layout(rect=(0, 0, 0.82, 1))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    ticker_prices = load_ticker_series()
    target_prices = build_target_prices(ticker_prices)
    ticker_returns = calculate_returns(ticker_prices)
    target_returns = calculate_returns(target_prices)

    correlations = calculate_target_correlations(
        ticker_returns, target_returns
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(CORRELATION_OUTPUT_FILE, float_format="%.8f")

    plot_paths = {
        FIVE_YEAR_TARGET: OUTPUT_DIR / "market_data_rolling_correlation_5y_future.png",
        WEIGHTED_BASKET_TARGET: (
            OUTPUT_DIR / "market_data_rolling_correlation_5x2y_minus_3x10y.png"
        ),
    }
    for target, output_path in plot_paths.items():
        save_rolling_correlation_plot(
            ticker_returns,
            target_returns[target],
            target,
            output_path,
        )

    print(
        f"Correlations calculated for all {len(ticker_returns)} ticker series "
        "using percentage changes."
    )
    print(
        "Direct correlations among the 5Y, 2Y, and 10Y futures are omitted, "
        "as is the 5Y-versus-weighted-basket target correlation."
    )
    print(f"Correlation matrix: {CORRELATION_OUTPUT_FILE}")
    for output_path in plot_paths.values():
        print(f"Rolling correlation plot: {output_path}")


if __name__ == "__main__":
    main()
