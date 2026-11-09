"""Re-run the previous four-strategy comparison with new scaled-CDF targets."""

import argparse

import analyze_best_both_cdf_futures_strategies as previous
import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as configs


TARGETS = {"5Y future": 0.03, "10Y future": 0.05}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounding",
        type=float,
        choices=reference.cdf.POSITION_ROUNDING_INCREMENTS,
        help="Override the position rounding for both CDF models.",
    )
    args = parser.parse_args()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    for instrument, target in TARGETS.items():
        config = configs.REQUESTED_CONFIGURATIONS[instrument]
        grid = previous.selected_grid(instrument, config)
        stem = instrument.split()[0].lower()
        rounding = config["rounding"] if args.rounding is None else args.rounding
        tag = str(rounding).replace(".", "p")
        target_pct = round(target * 100)
        directory = reference.RESULTS / (
            f"{stem}_four_strategies_cdf_rounded_{tag}_target_vol_{target_pct}pct_2016_2025_and_2026"
        )
        directory.mkdir(parents=True, exist_ok=True)
        grid.to_csv(directory / "prior_1pct_cdf_selection_grid.csv", index=False, float_format="%.8f")
        reference.save_report(
            instrument, prices[instrument], config["parameters"], directory,
            rounding, rounding, target,
        )


if __name__ == "__main__":
    main()
