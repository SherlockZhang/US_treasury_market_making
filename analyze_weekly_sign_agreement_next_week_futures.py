"""Analyze filtered weekly market-return sign agreement with Treasury futures."""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_5y_trend_strategy import parse_excel_dates


ROOT = Path(__file__).resolve().parent
MARKET_DATA_FILE = ROOT / "market_data.xlsx"
FUTURE_DATA_FILE = ROOT / "US_future_data.xlsx"
OUTPUT_DIR = ROOT / "results_refining_parameter"
SHEET_NAME = "Sheet1"

MARKET_VARIABLES = (
    "TFF2GLFN Index",
    "XAU BGN Curncy",
    "EURUSD BGN Curncy",
    "GSTMTAIP Index",
    "CTA US Equity",
    "TFF2HLFN Index",
    "UKX",
    "JPYUSD BGN Curncy",
    "S5INFT Index",
    "XBTUSD BGN Curncy",
    "DAX",
    "NDX",
    "BBDXY Index",
    "TFF2ILFN Index",
    "DJI",
    "SPX",
    "CL1 COMB Comdty",
    "SOFRRATE Index",
)
TARGETS = {
    "5Y future": "5y",
    "10Y future": "10y",
}
FILTER_RULE = (
    "abs(market weekly return - full-sample mean) > "
    "full-sample standard deviation"
)
OUTPUT_COLUMNS = (
    "market_variable",
    "target",
    "full_sample_weekly_return_mean",
    "full_sample_weekly_return_std_ddof1",
    "filter_rule",
    "total_paired_weeks",
    "filtered_weeks",
    "both_positive_count",
    "both_negative_count",
    "same_nonzero_sign_count",
    "different_sign_or_zero_count_filtered",
    "same_sign_pct_of_filtered_weeks",
    "sample_start_week",
    "sample_end_week",
)


def load_value_series(frame: pd.DataFrame, value_column: str) -> pd.Series:
    """Load a value column and its adjacent Excel date column."""
    if value_column not in frame.columns:
        raise ValueError(f"Missing required column {value_column!r}.")

    column_index = frame.columns.get_loc(value_column)
    if column_index == 0:
        raise ValueError(f"Missing adjacent date column for {value_column!r}.")
    date_column = frame.columns[column_index - 1]
    if not isinstance(date_column, str) or not date_column.strip().startswith(
        "Date"
    ):
        raise ValueError(f"Missing adjacent date column for {value_column!r}.")

    observations = frame.loc[:, [date_column, value_column]].copy()
    observations[date_column] = parse_excel_dates(observations[date_column])
    observations[value_column] = pd.to_numeric(
        observations[value_column], errors="raise"
    )
    incomplete_rows = observations.isna().any(axis=1) & observations.notna().any(
        axis=1
    )
    if incomplete_rows.any():
        raise ValueError(f"{value_column} has rows with only a date or value.")
    observations = observations.dropna(how="all")
    if observations.empty:
        raise ValueError(f"No observations found for {value_column}.")
    if observations[date_column].duplicated().any():
        raise ValueError(f"{value_column} contains duplicate dates.")
    if not np.isfinite(observations[value_column].dropna()).all():
        raise ValueError(f"{value_column} values must be finite.")

    return (
        observations.sort_values(date_column)
        .set_index(date_column)[value_column]
        .rename(value_column)
    )


def analyze_target(
    target_name: str,
    market_series: dict[str, pd.Series],
    target_prices: pd.Series,
) -> pd.DataFrame:
    """Build the 1-standard-deviation sign-agreement summary for one future."""
    target_weekly_returns = (
        target_prices.resample("W-FRI").last().pct_change(fill_method=None).shift(-1)
    )
    rows = []

    for market_variable, market_prices in market_series.items():
        market_returns = (
            market_prices.resample("W-FRI").last().pct_change(fill_method=None)
        )
        paired = pd.concat(
            [
                market_returns.rename("market_return"),
                target_weekly_returns.rename("next_week_target_return"),
            ],
            axis=1,
            join="inner",
            sort=True,
        ).dropna()
        if paired.empty:
            raise ValueError(
                f"No paired weekly returns for {market_variable} and {target_name}."
            )

        mean = market_returns.mean()
        standard_deviation = market_returns.std(ddof=1)
        if not np.isfinite(mean) or not np.isfinite(standard_deviation):
            raise ValueError(f"Invalid return statistics for {market_variable}.")

        filtered = paired.loc[
            (paired["market_return"] - mean).abs() > standard_deviation
        ]
        same_sign = (
            (filtered["market_return"] > 0) & (filtered["next_week_target_return"] > 0)
        ) | (
            (filtered["market_return"] < 0)
            & (filtered["next_week_target_return"] < 0)
        )
        same_sign_count = int(same_sign.sum())
        filtered_count = len(filtered)

        rows.append(
            {
                "market_variable": market_variable,
                "target": target_name,
                "full_sample_weekly_return_mean": mean,
                "full_sample_weekly_return_std_ddof1": standard_deviation,
                "filter_rule": FILTER_RULE,
                "total_paired_weeks": len(paired),
                "filtered_weeks": filtered_count,
                "both_positive_count": int(
                    (
                        (filtered["market_return"] > 0)
                        & (filtered["next_week_target_return"] > 0)
                    ).sum()
                ),
                "both_negative_count": int(
                    (
                        (filtered["market_return"] < 0)
                        & (filtered["next_week_target_return"] < 0)
                    ).sum()
                ),
                "same_nonzero_sign_count": same_sign_count,
                "different_sign_or_zero_count_filtered": (
                    filtered_count - same_sign_count
                ),
                "same_sign_pct_of_filtered_weeks": (
                    same_sign_count / filtered_count * 100
                    if filtered_count
                    else np.nan
                ),
                "sample_start_week": (
                    filtered.index.min().strftime("%Y-%m-%d")
                    if filtered_count
                    else ""
                ),
                "sample_end_week": (
                    filtered.index.max().strftime("%Y-%m-%d")
                    if filtered_count
                    else ""
                ),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    market_frame = pd.read_excel(MARKET_DATA_FILE, sheet_name=SHEET_NAME)
    future_frame = pd.read_excel(FUTURE_DATA_FILE, sheet_name=SHEET_NAME)

    missing_markets = set(MARKET_VARIABLES).difference(market_frame.columns)
    if missing_markets:
        raise ValueError(f"Missing market variables: {sorted(missing_markets)}")

    market_series = {
        ticker: load_value_series(market_frame, ticker)
        for ticker in MARKET_VARIABLES
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, filename_suffix in TARGETS.items():
        target_prices = load_value_series(future_frame, target_name)
        summary = analyze_target(target_name, market_series, target_prices)
        output_file = (
            OUTPUT_DIR
            / f"weekly_sign_agreement_market_variables_vs_next_week_{filename_suffix}_1std_filtered.csv"
        )
        summary.to_csv(output_file, index=False, float_format="%.8f")
        print(f"Saved {len(summary)} market-variable rows to {output_file}")


if __name__ == "__main__":
    main()
