"""Compare DBMF with a portfolio weighted by ridge regression coefficients."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_future_data import (
    TARGET_COLUMN,
    calculate_aligned_returns,
    fit_ridge_regression,
    load_price_series,
)


OUTPUT_DIR = Path(__file__).with_name("results")
PLOT_FILE = OUTPUT_DIR / "future_dbmf_vs_coefficient_portfolio.png"
PATHS_FILE = OUTPUT_DIR / "future_dbmf_vs_coefficient_portfolio_paths.csv"
WEIGHTS_FILE = OUTPUT_DIR / "future_dbmf_ridge_component_weights.csv"


def run_output_path(path, window):
    suffix = f"{window}d" if window is not None else "full"
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def positive_window(value):
    window = int(value)
    if window <= 0:
        raise argparse.ArgumentTypeError("window must be a positive integer")
    return window


def calculate_portfolio_paths(return_data, coefficients):
    """Apply ridge coefficients to raw component returns and compound both paths."""
    portfolio_returns = return_data[coefficients.index].mul(coefficients).sum(axis=1)
    dbmf_returns = return_data[TARGET_COLUMN]
    portfolio_prices = (1 + portfolio_returns / 100).cumprod() * 100
    dbmf_prices = (1 + dbmf_returns / 100).cumprod() * 100
    initial_date = return_data.index[0] - pd.Timedelta(days=1)
    portfolio_prices = pd.concat(
        [pd.Series([100.0], index=[initial_date]), portfolio_prices]
    )
    dbmf_prices = pd.concat(
        [pd.Series([100.0], index=[initial_date]), dbmf_prices]
    )
    paths = pd.DataFrame(
        {
            "DBMF raw return (%)": dbmf_returns,
            "coefficient portfolio return (%)": portfolio_returns,
            "DBMF indexed price": dbmf_prices.reindex(return_data.index),
            "coefficient portfolio indexed price": portfolio_prices.reindex(
                return_data.index
            ),
        },
        index=return_data.index,
    )
    return paths, dbmf_prices, portfolio_prices


def save_outputs(
    coefficients, paths, dbmf_prices, portfolio_prices, window
):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_path = run_output_path(PLOT_FILE, window)
    paths_path = run_output_path(PATHS_FILE, window)
    weights_path = run_output_path(WEIGHTS_FILE, window)

    paths.to_csv(paths_path, index_label="Date", float_format="%.3f")
    coefficients.rename("ridge_coefficient_weight").to_csv(
        weights_path, index_label="component", float_format="%.3f"
    )

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        dbmf_prices.index,
        dbmf_prices,
        color="#9A9A9A",
        linewidth=1.3,
        label="DBMF raw returns",
    )
    axis.plot(
        portfolio_prices.index,
        portfolio_prices,
        color="#176B73",
        linewidth=1.4,
        label="Ridge-coefficient portfolio",
    )
    axis.set_title("DBMF vs Portfolio Using Ridge Regression Coefficients")
    axis.set_xlabel("Date")
    axis.set_ylabel("Indexed value (start = 100)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Saved portfolio comparison plot to {plot_path}")
    print(f"Saved daily portfolio paths to {paths_path}")
    print(f"Saved ridge component weights to {weights_path}")


def main(window=None):
    return_data = calculate_aligned_returns(load_price_series())
    if window is not None:
        if window > len(return_data):
            raise ValueError(
                f"Requested {window} return observations, but only "
                f"{len(return_data)} are available."
            )
        return_data = return_data.tail(window)

    _, coefficients, stats = fit_ridge_regression(return_data)
    paths, dbmf_prices, portfolio_prices = calculate_portfolio_paths(
        return_data, coefficients
    )
    save_outputs(
        coefficients,
        paths,
        dbmf_prices,
        portfolio_prices,
        window,
    )
    print(
        f"Component analysis ({'last ' + str(window) + ' observations' if window else 'full history'}): "
        f"n={stats['observations']}, portfolio ending value="
        f"{portfolio_prices.iloc[-1]:.3f}, DBMF ending value={dbmf_prices.iloc[-1]:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare DBMF to a ridge-coefficient component portfolio."
    )
    parser.add_argument(
        "--window",
        type=positive_window,
        help="fit on the last N aligned return observations (for example, --window 60)",
    )
    main(parser.parse_args().window)
