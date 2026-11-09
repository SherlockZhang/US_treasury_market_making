"""Compare 2026 market-factor effects using the latest four-strategy settings."""

import pandas as pd

import analyze_2y_four_strategies_target_volatility as two
import analyze_best_cdf_market_factor_comparison as comparison
import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as configs
import analyze_four_strategies_custom_volatility as targets


OUTPUT_DIR = reference.RESULTS / "cdf_four_strategy_model_2026_market_factor_custom_vol_rounding"
SETTINGS = {
    "2Y future": (two.PARAMETERS["Normal CDF scaled"]["lookback"], two.TARGET, 1.0,
                  "2y_four_strategies_cdf_rounded_1_target_vol_1p5pct_2016_2025_and_2026"),
    "5Y future": (configs.REQUESTED_CONFIGURATIONS["5Y future"]["parameters"]["Normal CDF scaled"]["lookback"],
                  targets.TARGETS["5Y future"], 0.5,
                  "5y_four_strategies_cdf_rounded_0p5_target_vol_3pct_2016_2025_and_2026"),
    "10Y future": (configs.REQUESTED_CONFIGURATIONS["10Y future"]["parameters"]["Normal CDF scaled"]["lookback"],
                   targets.TARGETS["10Y future"], 0.5,
                   "10y_four_strategies_cdf_rounded_0p5_target_vol_5pct_2016_2025_and_2026"),
}


def main() -> None:
    prices_by_instrument = reference.cdf.load_single_future_prices(include_dbmf=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for instrument, (lookback, target, rounding, prior_folder) in SETTINGS.items():
        frames = comparison.backtest_pair(
            instrument, prices_by_instrument[instrument], lookback, target, rounding
        )
        stem = instrument.split()[0].lower()
        for mode, suffix in (("none", "without"), ("instrument", "with")):
            frame = frames[mode]
            frame.to_csv(
                OUTPUT_DIR / f"{stem}_{suffix}_market_factors_daily.csv",
                index_label="date", float_format="%.10f",
            )
            summary = comparison.summarize(
                instrument, mode, lookback, frame, target, rounding,
                lookback_selection="retained from prior four-strategy comparison; not re-optimized",
            )
            summary["reference_four_strategy_report"] = prior_folder
            summaries.append(summary)
        comparison.save_plots(instrument, frames, lookback, OUTPUT_DIR)
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "summary.csv", index=False, float_format="%.8f"
    )
    print(f"Market-factor comparison: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
