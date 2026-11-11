"""Compare 10Y filtered winners with the specified scaled CDF grid candidate."""

import argparse
from pathlib import Path

import pandas as pd

import analyze_2y_best_four_gross_win_filtered as report
import analyze_best_four_futures_strategies as reference


INSTRUMENT = "10Y future"
OUTPUT = reference.RESULTS / "10y_four_gross_filtered_weekly_t64_rounding_0p5_2016_2025_and_2026"


def selected_candidate() -> pd.Series:
    grid = pd.read_csv(report.search.OUTPUT_DIR / "cdf_parameter_grid.csv")
    candidates = grid.loc[
        grid["instrument"].eq(INSTRUMENT)
        & grid["volatility_scaling"].eq(True)
        & grid["rebalance_frequency"].eq("weekly")
        & grid["signal_lookback_days"].eq(64)
        & grid["position_rounding_increment"].eq(0.5)
    ]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one 10Y scaled CDF candidate with the requested settings.")
    candidate = candidates.iloc[0]
    if candidate["sharpe_rank"] != 3 or not candidate["eligible"]:
        raise ValueError("Requested 10Y scaled CDF candidate is not the third eligible grid model.")
    return candidate


def save_report(directory: Path) -> None:
    candidate = selected_candidate()
    report.save_report(directory, INSTRUMENT, scaled_candidate=candidate)
    pd.DataFrame([candidate]).to_csv(
        directory / "scaled_cdf_selected_grid_row.csv", index=False, float_format="%.10f"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    save_report(args.output_dir)


if __name__ == "__main__":
    main()
