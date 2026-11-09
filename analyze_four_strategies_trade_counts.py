"""Reproduce the three four-strategy reports with position-change counts."""

import analyze_2y_four_strategies_target_volatility as two
import analyze_best_both_cdf_futures_strategies as selection
import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as configurations
import analyze_four_strategies_custom_volatility as targets


def main() -> None:
    two.validate_reference()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    two_output = reference.RESULTS / (
        "2y_four_strategies_cdf_rounded_1_target_vol_1p5pct_2016_2025_and_2026_trade_counts"
    )
    reference.save_report(two.INSTRUMENT, prices[two.INSTRUMENT], two.PARAMETERS,
                          two_output, 1.0, 1.0, two.TARGET)

    for instrument, target in targets.TARGETS.items():
        config = configurations.REQUESTED_CONFIGURATIONS[instrument]
        grid = selection.selected_grid(instrument, config)
        stem = instrument.split()[0].lower()
        directory = reference.RESULTS / (
            f"{stem}_four_strategies_cdf_rounded_0p5_target_vol_"
            f"{round(target * 100)}pct_2016_2025_and_2026_trade_counts"
        )
        directory.mkdir(parents=True, exist_ok=True)
        grid.to_csv(directory / "prior_1pct_cdf_selection_grid.csv",
                    index=False, float_format="%.8f")
        reference.save_report(instrument, prices[instrument], config["parameters"],
                              directory, 0.5, 0.5, target)


if __name__ == "__main__":
    main()
