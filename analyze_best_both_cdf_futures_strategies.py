"""Reproduce the four-strategy reports with both CDF models rounded per future."""

import pandas as pd

import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as previous


def selected_grid(instrument: str, config: previous.Configuration) -> pd.DataFrame:
    stem = instrument.split()[0].lower()
    tag = str(config["rounding"]).replace(".", "p")
    grid = pd.read_csv(
        reference.RESULTS
        / f"{stem}_cdf_scaled_unscaled_rounding_{tag}_daily_weekly_grid_2016_2025.csv"
    )
    grid["selected_for_report"] = False
    for scaled, name in ((True, reference.STRATEGIES[0]), (False, reference.STRATEGIES[1])):
        candidates = grid.loc[grid["volatility_scaling"].eq(scaled)]
        if (
            len(candidates) != len(reference.CDF_LOOKBACKS) * 2
            or not candidates["sharpe_ratio"].is_monotonic_decreasing
            or candidates["sharpe_rank"].eq(1).sum() != 1
        ):
            raise ValueError(f"Invalid training grid ranking for {instrument}: {name}.")
        winner = candidates.iloc[0]
        if (
            winner["sharpe_rank"] != 1
            or winner["instrument"] != instrument
            or winner["position_rounding_increment"] != config["rounding"]
            or winner["signal_lookback_days"] != config["parameters"][name]["lookback"]
            or winner["rebalance_frequency"] != "weekly"
            or winner["end_date"] != "2025-12-31"
            or winner["signal_cutoff"] != 0
            or winner["rebalance_threshold_fraction"] != 0
        ):
            raise ValueError(f"Training grid winner does not match {instrument}: {name}.")
        grid.loc[winner.name, "selected_for_report"] = True
    return grid


def generate(instrument: str, prices: pd.Series, config: previous.Configuration) -> None:
    grid = selected_grid(instrument, config)
    stem = instrument.split()[0].lower()
    original_directory = reference.RESULTS / f"{stem}_best_four_strategies_rounded_0p5_2016_2025_and_2026"
    original_summary = pd.read_csv(original_directory / "summary.csv")
    labels = reference.parameter_labels(config["parameters"])
    for name in reference.STRATEGIES[2:]:
        rows = original_summary.loc[
            original_summary["instrument"].eq(instrument)
            & original_summary["period"].eq("2016_2025")
            & original_summary["strategy"].eq(name)
        ]
        if len(rows) != 1 or rows.iloc[0]["parameters"] != labels[name]:
            raise ValueError(f"Reference parameters do not match for {instrument}: {name}.")

    rounding = config["rounding"]
    tag = str(rounding).replace(".", "p")
    directory = reference.RESULTS / f"{stem}_best_four_strategies_cdf_rounded_{tag}_2016_2025_and_2026"
    directory.mkdir(parents=True, exist_ok=True)
    grid.to_csv(directory / "rounded_cdf_training_grid.csv", index=False, float_format="%.8f")
    reference.save_report(instrument, prices, config["parameters"], directory, rounding, rounding)


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    for instrument, config in previous.REQUESTED_CONFIGURATIONS.items():
        generate(instrument, prices[instrument], config)


if __name__ == "__main__":
    main()
