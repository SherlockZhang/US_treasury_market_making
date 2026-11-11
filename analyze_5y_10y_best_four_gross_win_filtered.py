"""Backtest the four gross-win-ratio-filtered winners for 5Y and 10Y futures."""

import argparse
from pathlib import Path

import analyze_2y_best_four_gross_win_filtered as report
import analyze_best_four_futures_strategies as reference


INSTRUMENTS = ("5Y future", "10Y future")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=reference.RESULTS)
    args = parser.parse_args()
    for instrument in INSTRUMENTS:
        stem = instrument.split()[0].lower()
        directory = args.output_root / f"{stem}_best_four_gross_win_filtered_2016_2025_and_2026"
        report.save_report(directory, instrument)


if __name__ == "__main__":
    main()
