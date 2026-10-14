"""Run weekly rolling ridge regressions of DBMF returns on futures factors."""

import matplotlib.pyplot as plt
import pandas as pd

from analyze_future_data import (
    OUTPUT_DIR,
    REGRESSION_TICKERS,
    calculate_aligned_returns,
    fit_ridge_regression,
    load_price_series,
)


WINDOW_SIZE = 60
RETURNS_FILE = OUTPUT_DIR / "future_rolling_60d_tuesday_returns.csv"
RESULTS_FILE = OUTPUT_DIR / "future_rolling_60d_tuesday_regression.csv"
WEIGHTS_PLOT_FILE = OUTPUT_DIR / "future_rolling_60d_tuesday_weights.png"
R_SQUARED_PLOT_FILE = OUTPUT_DIR / "future_rolling_60d_tuesday_r_squared.png"


def calculate_rolling_regressions(return_data):
    """Fit one model per Tuesday using that date and the prior 59 returns."""
    predictors = REGRESSION_TICKERS
    results = []

    for end_position, date in enumerate(return_data.index):
        if date.weekday() != 1 or end_position + 1 < WINDOW_SIZE:
            continue

        window = return_data.iloc[end_position - WINDOW_SIZE + 1 : end_position + 1]
        _, coefficients, stats = fit_ridge_regression(window)
        result = {
            "date": date,
            "observations": stats["observations"],
            "intercept_percent": stats["intercept_percent"],
            "r_squared": stats["r_squared"],
        }
        result.update(
            {
                f"weight_{column}": coefficients[column]
                for column in predictors
            }
        )
        results.append(result)

    if not results:
        raise ValueError(
            "No Tuesday regression could be fit with at least "
            f"{WINDOW_SIZE} aligned daily return observations."
        )
    return pd.DataFrame(results).set_index("date")


def save_outputs(return_data, regression_results):
    """Save aligned returns, rolling statistics, and their time-series plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return_data.to_csv(RETURNS_FILE, index_label="Date", float_format="%.3f")
    regression_results.to_csv(
        RESULTS_FILE, index_label="Date", float_format="%.3f"
    )

    weight_columns = [
        column for column in regression_results.columns if column.startswith("weight_")
    ]
    figure, axis = plt.subplots(figsize=(13, 7))
    for column in weight_columns:
        axis.plot(
            regression_results.index,
            regression_results[column],
            linewidth=1.2,
            label=column.removeprefix("weight_"),
        )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title(f"Rolling {WINDOW_SIZE}-Day Tuesday Ridge Regression Weights")
    axis.set_xlabel("Regression date")
    axis.set_ylabel("Ridge coefficient")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(WEIGHTS_PLOT_FILE, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        regression_results.index,
        regression_results["r_squared"],
        color="#176B73",
        linewidth=1.4,
    )
    axis.set_title(f"Rolling {WINDOW_SIZE}-Day Tuesday Ridge Regression R-squared")
    axis.set_xlabel("Regression date")
    axis.set_ylabel("R-squared")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(R_SQUARED_PLOT_FILE, dpi=150)
    plt.close(figure)

    print(f"Saved aligned returns to {RETURNS_FILE}")
    print(f"Saved rolling regression results to {RESULTS_FILE}")
    print(f"Saved rolling factor weights plot to {WEIGHTS_PLOT_FILE}")
    print(f"Saved rolling R-squared plot to {R_SQUARED_PLOT_FILE}")


def main():
    prices = load_price_series()
    return_data = calculate_aligned_returns(prices)
    regression_results = calculate_rolling_regressions(return_data)
    save_outputs(return_data, regression_results)
    print(
        f"Completed {len(regression_results)} Tuesday regressions "
        f"using {WINDOW_SIZE} aligned return observations each."
    )


if __name__ == "__main__":
    main()
