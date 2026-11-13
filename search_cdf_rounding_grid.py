"""Search CDF position-rounding increments alongside lookback and frequency."""

import argparse
from pathlib import Path

import search_cdf_custom_vol_daily_weekly as previous


ROUNDING_INCREMENTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
OUTPUT_DIR = previous.reference.RESULTS / "cdf_custom_vol_rounding_daily_weekly_grid_2016_2025"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    prices = previous.reference.cdf.load_single_future_prices(include_dbmf=False)
    grid = previous.search(
        {instrument: prices[instrument] for instrument in previous.CONFIGURATIONS},
        rounding_increments=ROUNDING_INCREMENTS,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "parameter_grid.csv", index=False, float_format="%.10f")
    best = grid.loc[grid["selected_best"]]
    best.to_csv(args.output_dir / "best_parameters.csv", index=False, float_format="%.10f")
    undefined = grid.loc[grid["sharpe_status"].ne("ok")]
    print(best.to_string(index=False))
    print(f"\n{len(undefined)} candidates have undefined Sharpe due to zero return volatility.")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
