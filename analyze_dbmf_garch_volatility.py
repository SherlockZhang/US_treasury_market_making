"""Calculate EWMA volatility and plot DBMF price with SPX and IEF volatility."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Configuration
INPUT_FILE = Path(__file__).with_name("DBMF_COMP.csv")
OUTPUT_DIR = Path(__file__).with_name("vol_analysis")
OUTPUT_PLOT = OUTPUT_DIR / "DBMF_COMP_ewma_volatility.png"
DATE_COLUMN = "Date"
PRICE_COLUMNS = ["DBMF", "SPX", "IEF"]
ANNUALIZATION_DAYS = 252
EWMA_LAMBDA = 0.97


# Data loading and processing
def load_prices():
    """Load and validate price history for DBMF, SPX, and IEF."""
    data = pd.read_csv(INPUT_FILE, parse_dates=[DATE_COLUMN])
    required_columns = [DATE_COLUMN, *PRICE_COLUMNS]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {', '.join(missing_columns)}"
        )

    prices = data[required_columns].copy()
    prices = prices.dropna(subset=[DATE_COLUMN]).set_index(DATE_COLUMN).sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    if prices.index.has_duplicates:
        raise ValueError("The input CSV contains duplicate dates.")
    if prices["DBMF"].dropna().empty:
        raise ValueError("The input CSV does not contain valid DBMF prices.")
    for ticker in PRICE_COLUMNS:
        if prices[ticker].dropna().empty:
            raise ValueError(f"The input CSV does not contain valid {ticker} prices.")
    return prices


def calculate_returns(prices):
    """Calculate returns, skipping missing observations within each price series."""
    returns = prices.pct_change(fill_method=None) * 100
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    for ticker in PRICE_COLUMNS:
        series = returns[ticker].dropna()
        if len(series) < 3:
            raise ValueError(f"Not enough valid returns to calculate EWMA for {ticker}.")
    return returns


# Calculations
def calculate_ewma_volatility(returns, ticker):
    """Calculate annualized EWMA conditional volatility using prior returns."""
    valid_returns = returns.dropna().astype(float)
    values = valid_returns.to_numpy()
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError(f"Not enough finite returns to calculate EWMA for {ticker}.")

    conditional_variance = np.empty_like(values)
    conditional_variance[0] = np.var(values, ddof=0)
    for index in range(1, len(values)):
        conditional_variance[index] = (
            EWMA_LAMBDA * conditional_variance[index - 1]
            + (1 - EWMA_LAMBDA) * values[index - 1] ** 2
        )
    if (conditional_variance <= 0).any():
        raise ValueError(f"EWMA variance is not positive for {ticker}.")

    return pd.Series(
        np.sqrt(conditional_variance) * np.sqrt(ANNUALIZATION_DAYS),
        index=valid_returns.index,
        name=f"{ticker} annualized volatility (%)",
    )


# Output
def plot_price_and_volatility(prices, volatilities):
    """Plot DBMF's compounded return index and annualized EWMA volatilities."""
    complete_volatilities = pd.concat(volatilities, axis=1)
    figure, price_axis = plt.subplots(figsize=(13, 7))
    volatility_axis = price_axis.twinx()

    dbmf_prices = prices["DBMF"].dropna()
    dbmf_returns = dbmf_prices.pct_change(fill_method=None).dropna()
    indexed_price = (1 + dbmf_returns).cumprod() * 100
    indexed_price = pd.concat(
        [pd.Series([100.0], index=[dbmf_prices.index[0]]), indexed_price]
    )
    price_axis.plot(
        indexed_price.index,
        indexed_price,
        color="#1f77b4",
        linewidth=1.5,
        label="DBMF indexed price (start = 100)",
    )
    colors = {"SPX": "#C62828", "IEF": "#EF9A9A"}
    for ticker in ("SPX", "IEF"):
        volatility_axis.plot(
            complete_volatilities.index,
            complete_volatilities[f"{ticker} annualized volatility (%)"],
            color=colors[ticker],
            linewidth=1.1,
            label=f"{ticker} EWMA volatility",
        )

    price_axis.set_title(f"DBMF Price and EWMA Volatility (lambda = {EWMA_LAMBDA:.2f})")
    price_axis.set_xlabel("Date")
    price_axis.set_ylabel("DBMF indexed price (start = 100)")
    volatility_axis.set_ylabel("Annualized EWMA volatility (%)")
    volatility_axis.set_yscale("log")
    price_axis.grid(True, alpha=0.25)

    lines, labels = price_axis.get_legend_handles_labels()
    volatility_lines, volatility_labels = (
        volatility_axis.get_legend_handles_labels()
    )
    price_axis.legend(
        lines + volatility_lines,
        labels + volatility_labels,
        loc="upper left",
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PLOT, dpi=150)
    plt.close(figure)
    print(f"Saved DBMF price and EWMA volatility plot to {OUTPUT_PLOT}")


# Main analysis
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    prices = load_prices()
    returns = calculate_returns(prices)
    volatilities = []

    for ticker in PRICE_COLUMNS:
        volatility = calculate_ewma_volatility(returns[ticker], ticker)
        volatilities.append(volatility)
        print(
            f"{ticker} EWMA volatility: lambda={EWMA_LAMBDA:.2f}, "
            f"latest annualized volatility={volatility.iloc[-1]:.2f}%"
        )

    plot_price_and_volatility(prices, volatilities)


if __name__ == "__main__":
    main()
