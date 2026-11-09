"""Reproduce four-strategy reports with selected expanded-grid CDF models."""

import argparse
from typing import TypedDict

import pandas as pd

import analyze_best_four_futures_strategies as reference


class Configuration(TypedDict):
    rounding: float
    parameters: dict[str, dict[str, int]]


CONFIGURATIONS: dict[str, Configuration] = {
    "5Y future": {
        "rounding": 0.4,
        "parameters": {
            "Normal CDF scaled": {"lookback": 64},
            "Normal CDF unscaled": {"lookback": 252},
            "EWMA": {"n": 55, "m": 20},
            "KAMA": {"fast": 5, "slow": 25, "er": 25, "m": 3},
        },
    },
    "10Y future": {
        "rounding": 0.2,
        "parameters": {
            "Normal CDF scaled": {"lookback": 32},
            "Normal CDF unscaled": {"lookback": 64},
            "EWMA": {"n": 35, "m": 15},
            "KAMA": {"fast": 5, "slow": 20, "er": 25, "m": 3},
        },
    },
}

REQUESTED_CONFIGURATIONS: dict[str, Configuration] = {
    instrument: {
        "rounding": {"5Y future": 0.2, "10Y future": 0.1}[instrument],
        "parameters": {
            **config["parameters"],
            "Normal CDF scaled": {"lookback": 64},
        },
    }
    for instrument, config in CONFIGURATIONS.items()
}


def generate(instrument: str, prices: pd.Series, config: Configuration) -> None:
    rounding = config["rounding"]
    parameters = config["parameters"]
    stem = instrument.split()[0].lower()
    grid = pd.read_csv(reference.RESULTS / f"{stem}_scaled_cdf_rounding_grid_2016_2025.csv")
    selected = (
        grid["instrument"].eq(instrument)
        & grid["signal_lookback_days"].eq(parameters["Normal CDF scaled"]["lookback"])
        & grid["position_rounding_increment"].eq(rounding)
    )
    if selected.sum() != 1 or not grid["sharpe_ratio"].is_monotonic_decreasing:
        raise ValueError(f"Saved grid does not uniquely contain the requested {instrument} model.")

    original_directory = reference.RESULTS / f"{stem}_best_four_strategies_rounded_0p5_2016_2025_and_2026"
    original_summary = pd.read_csv(original_directory / "summary.csv")
    original_labels = reference.parameter_labels(parameters)
    for strategy in reference.STRATEGIES[1:]:
        rows = original_summary.loc[
            original_summary["instrument"].eq(instrument)
            & original_summary["period"].eq("2016_2025")
            & original_summary["strategy"].eq(strategy)
        ]
        if len(rows) != 1 or rows.iloc[0]["parameters"] != original_labels[strategy]:
            raise ValueError(f"Reference parameters do not match for {instrument}: {strategy}.")

    rounding_tag = str(rounding).replace(".", "p")
    directory = reference.RESULTS / (
        f"{stem}_best_four_strategies_scaled_rounded_{rounding_tag}_2016_2025_and_2026"
    )
    directory.mkdir(parents=True, exist_ok=True)
    grid["selected_for_report"] = selected
    grid.to_csv(directory / "rounded_scaled_cdf_training_grid.csv", index=False, float_format="%.8f")
    reference.save_report(instrument, prices, parameters, directory, rounding)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        choices=("grid-winners", "requested-t64"),
        default="grid-winners",
        help="Use the previous grid winners or the requested T=64 models.",
    )
    args = parser.parse_args()
    configurations = (
        CONFIGURATIONS if args.selection == "grid-winners" else REQUESTED_CONFIGURATIONS
    )
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    for instrument, config in configurations.items():
        generate(instrument, prices[instrument], config)


if __name__ == "__main__":
    main()
