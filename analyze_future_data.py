"""Clean futures prices, calculate aligned returns, and fit a DBMF ridge model."""

import argparse
import re
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Configuration
INPUT_FILE = Path(__file__).with_name("future_price.csv")
OUTPUT_DIR = Path(__file__).with_name("component_analysis") / "results"
RETURN_FILE = OUTPUT_DIR / "future_return.csv"
PRICE_PLOT_FILE = OUTPUT_DIR / "future_price_series.png"
REGRESSION_PLOT_FILE = OUTPUT_DIR / "future_dbmf_ridge_fit.png"
REGRESSION_STATS_FILE = OUTPUT_DIR / "future_ridge_stats.csv"
TARGET_COLUMN = "DBMF"
TREASURY_FUTURES_COLUMNS = {"2Y US", "20Y US", "10Y US"}
RIDGE_ALPHA = 1


# Data loading and processing
def parse_treasury_futures_price(value, column):
    """Convert a Treasury futures quote in points and 32nds to a decimal price."""
    quote = value.strip()
    match = re.fullmatch(
        r"(\d+)(?:-(\d{1,2})(?:\s+(\d+)/(\d+)|(\+))?)?",
        quote,
    )
    if match is None:
        raise ValueError(f"Invalid Treasury futures quote in {column}: {value!r}")

    points = int(match.group(1))
    thirty_seconds = int(match.group(2) or 0)
    if thirty_seconds >= 32:
        raise ValueError(f"Invalid 32nds in Treasury futures quote: {value!r}")

    if match.group(3):
        denominator = int(match.group(4))
        if denominator == 0:
            raise ValueError(f"Invalid fractional 32nds in Treasury futures quote: {value!r}")
        fraction = Fraction(int(match.group(3)), denominator)
    elif match.group(5):
        fraction = Fraction(1, 2)
    else:
        fraction = Fraction(0)
    if fraction >= 1:
        raise ValueError(f"Invalid fractional 32nds in Treasury futures quote: {value!r}")
    return points + (thirty_seconds + float(fraction)) / 32


def load_price_series():
    """Load independent date/price pairs from the repeated-header source CSV."""
    raw = pd.read_csv(INPUT_FILE, header=None, dtype=str, keep_default_na=False)
    header = raw.iloc[0].str.strip()
    price_series = {}

    for date_position, name in enumerate(header):
        if name.casefold() != "date":
            continue
        price_position = date_position + 1
        if price_position >= len(header):
            raise ValueError("A Date column in the input CSV has no price column.")
        ticker = header.iloc[price_position]
        if not ticker:
            raise ValueError("A Date column in the input CSV has an empty price header.")
        if ticker in price_series:
            raise ValueError(f"Duplicate price series in input CSV: {ticker}")

        dates = pd.to_datetime(raw.iloc[1:, date_position], errors="coerce")
        raw_prices = raw.iloc[1:, price_position].str.strip()
        has_price = raw_prices.ne("")
        invalid_dates = raw.iloc[1:, date_position].str.strip().ne("") & dates.isna()
        if invalid_dates.any():
            invalid_value = raw.iloc[1:, date_position][invalid_dates].iloc[0]
            raise ValueError(f"Invalid date in {ticker} series: {invalid_value!r}")

        if ticker in TREASURY_FUTURES_COLUMNS:
            numeric_prices = raw_prices.map(
                lambda value: (
                    parse_treasury_futures_price(value, ticker) if value else np.nan
                )
            )
        else:
            numeric_prices = pd.to_numeric(raw_prices, errors="coerce")

        invalid_prices = has_price & numeric_prices.isna()
        if invalid_prices.any():
            invalid_value = raw_prices[invalid_prices].iloc[0]
            raise ValueError(f"Invalid price in {ticker} series: {invalid_value!r}")

        series = pd.Series(numeric_prices.to_numpy(), index=dates, name=ticker)
        series = series.loc[series.index.notna()].dropna().sort_index()
        if series.index.has_duplicates:
            raise ValueError(f"Duplicate dates in {ticker} series.")
        if series.empty:
            raise ValueError(f"No valid prices found for {ticker}.")
        price_series[ticker] = series

    if TARGET_COLUMN not in price_series:
        raise ValueError(f"Required target price series {TARGET_COLUMN!r} is missing.")
    if len(price_series) < 2:
        raise ValueError("At least two price series are required for the analysis.")
    return pd.concat(price_series.values(), axis=1, sort=True).sort_index()


def calculate_aligned_returns(prices):
    """Calculate percentage returns and retain dates shared by every series."""
    return_data = pd.concat(
        [
            series.dropna().pct_change(fill_method=None).dropna().rename(series.name)
            for _, series in prices.items()
        ],
        axis=1,
        join="inner",
    ).dropna()
    if return_data.empty:
        raise ValueError("No dates have valid returns for all price series.")
    return return_data * 100


# Calculations
def fit_ridge_regression(return_data):
    """Fit standardized ridge regression to DBMF returns."""
    predictors = return_data.drop(columns=TARGET_COLUMN)
    target = return_data[TARGET_COLUMN]
    predictor_mean = predictors.mean()
    predictor_scale = predictors.std(ddof=0)
    if (predictor_scale == 0).any():
        constant_predictors = predictor_scale.index[predictor_scale == 0].tolist()
        raise ValueError(
            f"Cannot fit ridge regression with constant predictors: {constant_predictors}"
        )

    standardized = (predictors - predictor_mean) / predictor_scale
    centered_target = target - target.mean()
    x = standardized.to_numpy()
    y = centered_target.to_numpy()
    gram = x.T @ x
    penalized_gram = gram + len(x) * RIDGE_ALPHA * np.eye(x.shape[1])
    coefficients_standardized = np.linalg.solve(
        penalized_gram / len(x),
        x.T @ y / len(x),
    )
    coefficients = coefficients_standardized / predictor_scale.to_numpy()
    intercept = target.mean() - predictor_mean.to_numpy() @ coefficients
    fitted = pd.Series(
        intercept + predictors.to_numpy() @ coefficients,
        index=return_data.index,
        name="ridge_fitted_return",
    )
    residuals = target - fitted
    total_sum_squares = ((target - target.mean()) ** 2).sum()
    if total_sum_squares == 0:
        raise ValueError("DBMF returns have zero variance; R-squared is undefined.")

    stats = {
        "observations": len(return_data),
        "ridge_alpha": RIDGE_ALPHA,
        "intercept_percent": intercept,
        "r_squared": 1 - (residuals**2).sum() / total_sum_squares,
        "rmse_percent": np.sqrt((residuals**2).mean()),
    }
    for column, coefficient in zip(predictors.columns, coefficients):
        safe_column = re.sub(r"\W+", "_", column).strip("_")
        stats[f"coefficient_{safe_column}"] = coefficient
    return fitted, pd.Series(
        coefficients, index=predictors.columns
    ), stats


# Output
def run_output_path(path, window):
    """Add a regression-window suffix to an output filename."""
    suffix = f"{window}d" if window is not None else "full"
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def plot_price_series(prices, window):
    """Plot complete-date prices indexed to 100 at the first shared observation."""
    complete_prices = prices.dropna(how="any")
    if complete_prices.empty:
        raise ValueError("No dates have valid prices for every series to plot.")
    rebased = complete_prices.div(complete_prices.iloc[0], axis="columns") * 100
    figure, axis = plt.subplots(figsize=(13, 7))
    for ticker in rebased.columns:
        axis.plot(rebased.index, rebased[ticker], linewidth=1.2, label=ticker)
    axis.set_title("Futures and Market Price Series (Rebased to 100)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Price index (first valid observation = 100)")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    output_path = run_output_path(PRICE_PLOT_FILE, window)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved rebased price series plot to {output_path}")


def plot_regression_fit(return_data, fitted, window):
    """Plot raw DBMF returns and the in-sample ridge-fitted return series."""
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        return_data.index,
        return_data[TARGET_COLUMN],
        color="#9A9A9A",
        linewidth=1,
        alpha=0.7,
        label="DBMF raw return",
    )
    axis.plot(
        fitted.index,
        fitted,
        color="#176B73",
        linewidth=1.4,
        label="Ridge fitted return",
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("DBMF Raw and Ridge-Fitted Returns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Return (%)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = run_output_path(REGRESSION_PLOT_FILE, window)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved DBMF ridge regression plot to {output_path}")


def plot_compounded_returns(return_data, fitted, window):
    """Plot raw and fitted return paths, each compounded from an initial value of 100."""
    raw_price = (1 + return_data[TARGET_COLUMN] / 100).cumprod() * 100
    fitted_price = (1 + fitted / 100).cumprod() * 100
    initial_date = return_data.index[0] - pd.Timedelta(days=1)
    raw_price = pd.concat([pd.Series([100.0], index=[initial_date]), raw_price])
    fitted_price = pd.concat(
        [pd.Series([100.0], index=[initial_date]), fitted_price]
    )

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(raw_price.index, raw_price, color="#9A9A9A", linewidth=1.2, label="DBMF raw returns")
    axis.plot(
        fitted_price.index,
        fitted_price,
        color="#176B73",
        linewidth=1.4,
        label="Ridge fitted returns",
    )
    axis.set_title("DBMF Price Paths from Raw and Ridge-Fitted Returns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Indexed value (start = 100)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = run_output_path(
        REGRESSION_PLOT_FILE.with_name("future_dbmf_compounded_prices.png"),
        window,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved compounded DBMF price plot to {output_path}")


def save_results(return_data, stats, window):
    """Save windowed aligned returns and ridge regression statistics."""
    return_path = run_output_path(RETURN_FILE, window)
    stats_path = run_output_path(REGRESSION_STATS_FILE, window)
    return_data.to_csv(return_path, index_label="Date", float_format="%.3f")
    pd.DataFrame([stats]).to_csv(
        stats_path, index=False, float_format="%.3f"
    )
    print(f"Saved aligned returns to {return_path}")
    print(f"Saved ridge regression statistics to {stats_path}")


# Main analysis
def positive_window(value):
    """Parse a positive number of aligned return observations."""
    window = int(value)
    if window <= 0:
        raise argparse.ArgumentTypeError("window must be a positive integer")
    return window


def main(window=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    prices = load_price_series()
    return_data = calculate_aligned_returns(prices)
    if window is not None:
        if window > len(return_data):
            raise ValueError(
                f"Requested {window} return observations, but only "
                f"{len(return_data)} are available."
            )
        return_data = return_data.tail(window)

    fitted, _, stats = fit_ridge_regression(return_data)
    save_results(return_data, stats, window)
    plot_price_series(prices, window)
    plot_regression_fit(return_data, fitted, window)
    plot_compounded_returns(return_data, fitted, window)
    print(
        f"Ridge regression ({'last ' + str(window) + ' observations' if window else 'full history'}): "
        f"n={stats['observations']}, "
        f"R²={stats['r_squared']:.3f}, "
        f"RMSE={stats['rmse_percent']:.3f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze aligned futures returns with a DBMF ridge regression."
    )
    parser.add_argument(
        "--window",
        type=positive_window,
        help="fit on the last N aligned return observations (for example, --window 60)",
    )
    main(parser.parse_args().window)
