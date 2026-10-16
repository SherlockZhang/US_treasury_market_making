"""Calculate correlation between 5Y futures returns and 5Y yield changes."""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_5y_trend_strategy import load_5y_yield, load_prices


OUTPUT_DIR = Path(__file__).with_name("results")
OUTPUT_FILE = OUTPUT_DIR / "5y_future_yield_correlation.csv"


def calculate_correlation() -> dict[str, object]:
    """Calculate Pearson correlation on date-aligned daily changes."""
    futures_returns = load_prices()["price"].pct_change().rename(
        "5y_future_daily_return"
    )
    yield_changes = load_5y_yield().diff().rename("5y_yield_change_percentage_points")
    paired_changes = pd.concat(
        [futures_returns, yield_changes], axis=1, join="inner"
    ).dropna()
    if len(paired_changes) < 2:
        raise ValueError("At least two paired daily observations are required.")
    if (paired_changes.std(ddof=1) == 0).any():
        raise ValueError("Both daily change series must have positive variance.")

    correlation = paired_changes.iloc[:, 0].corr(paired_changes.iloc[:, 1])
    if not np.isfinite(correlation):
        raise ValueError("The Pearson correlation could not be calculated.")

    return {
        "correlation_method": "Pearson",
        "5y_future_measure": "daily percentage return",
        "5y_yield_measure": "daily change in percentage points",
        "observations": len(paired_changes),
        "start_date": paired_changes.index.min().date(),
        "end_date": paired_changes.index.max().date(),
        "correlation": correlation,
    }


def main() -> None:
    stats = calculate_correlation()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(OUTPUT_FILE, index=False, float_format="%.8f")
    print(
        f"5Y futures return / 5Y yield change correlation: "
        f"{stats['correlation']:.6f}"
    )
    print(
        f"Paired observations: {stats['observations']} "
        f"({stats['start_date']} to {stats['end_date']})"
    )
    print(f"Correlation statistics saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
