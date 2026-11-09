"""Reproduce the 2Y four-strategy comparison with a 1.5% scaled-CDF target."""

import argparse

import pandas as pd

import analyze_best_four_futures_strategies as reference


INSTRUMENT = "2Y future"
TARGET = 0.015
ROUNDING = 0.5
PARAMETERS = {
    "Normal CDF scaled": {"lookback": 64},
    "Normal CDF unscaled": {"lookback": 252},
    "EWMA": {"n": 5, "m": 40},
    "KAMA": {"fast": 5, "slow": 25, "er": 25, "m": 3},
}
SOURCE = reference.RESULTS / "2y_best_four_strategies_rounded_0p5_2016_2025_and_2026"
OUTPUT = reference.RESULTS / "2y_four_strategies_cdf_rounded_0p5_target_vol_1p5pct_2016_2025_and_2026"


def validate_reference() -> None:
    summary = pd.read_csv(SOURCE / "summary.csv")
    labels = reference.parameter_labels(PARAMETERS, ROUNDING, ROUNDING)
    for period in ("2016_2025", "2026"):
        for name in reference.STRATEGIES:
            rows = summary.loc[
                summary["instrument"].eq(INSTRUMENT)
                & summary["period"].eq(period)
                & summary["strategy"].eq(name)
            ]
            if len(rows) != 1 or rows.iloc[0]["parameters"] != labels[name]:
                raise ValueError(f"Reference parameters do not match {period}: {name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounding",
        type=float,
        choices=reference.cdf.POSITION_ROUNDING_INCREMENTS,
        default=ROUNDING,
        help="Position rounding for both CDF models.",
    )
    args = parser.parse_args()
    validate_reference()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    if args.rounding == ROUNDING:
        output = OUTPUT
    else:
        tag = f"{args.rounding:g}".replace(".", "p")
        output = reference.RESULTS / (
            f"2y_four_strategies_cdf_rounded_{tag}_target_vol_1p5pct_2016_2025_and_2026"
        )
    reference.save_report(INSTRUMENT, prices, PARAMETERS, output, args.rounding, args.rounding, TARGET)


if __name__ == "__main__":
    main()
