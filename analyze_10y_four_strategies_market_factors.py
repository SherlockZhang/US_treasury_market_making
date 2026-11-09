"""Compare 10Y strategies out of sample with factors on the scaled CDF only."""

import pandas as pd

import analyze_best_cdf_market_factor_comparison as factors
import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as configs
import analyze_custom_vol_cdf_market_factor_comparison as custom


INSTRUMENT = "10Y future"
LOOKBACK, TARGET, ROUNDING, PRIOR_FOLDER = custom.SETTINGS[INSTRUMENT]
PARAMETERS = configs.REQUESTED_CONFIGURATIONS[INSTRUMENT]["parameters"]
OUTPUT_DIR = reference.RESULTS / "10y_four_strategies_2026_scaled_cdf_market_factors_target_vol_5pct_rounded_0p5"


def backtest(prices: pd.Series) -> dict[str, pd.DataFrame]:
    if PARAMETERS[reference.STRATEGIES[0]]["lookback"] != LOOKBACK:
        raise ValueError("Scaled CDF lookback does not match the previous factor comparison.")
    frames = reference.run_period(
        prices.loc[:reference.TEST_END], PARAMETERS, reference.TEST_START,
        ROUNDING, ROUNDING, TARGET,
    )
    frames[reference.STRATEGIES[0]] = factors.backtest_pair(
        INSTRUMENT, prices, LOOKBACK, TARGET, ROUNDING
    )["instrument"]
    return frames


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    frames = backtest(prices)
    labels = reference.parameter_labels(PARAMETERS, ROUNDING, ROUNDING, TARGET)
    labels[reference.STRATEGIES[0]] += "; instrument market factors"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, slug in zip(reference.STRATEGIES, reference.SLUGS):
        frame = frames[name]
        position_change = frame["position"].diff()
        position_change.iloc[0] = frame["position"].iloc[0]
        frame["position_changed"] = position_change.ne(0)
        frame.to_csv(
            OUTPUT_DIR / f"2026_{slug}_daily.csv",
            index_label="date", float_format="%.10f",
        )
        summaries.append({
            "period": "2026",
            "instrument": INSTRUMENT,
            "strategy": name,
            "parameters": labels[name],
            "market_factor_mode": "instrument" if name == reference.STRATEGIES[0] else "none",
            "market_factor_profile": factors.profile_description(INSTRUMENT)
            if name == reference.STRATEGIES[0] else "",
            **reference.statistics(frame),
            "trading_volume": int(frame["position_changed"].sum()),
        })
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "summary.csv", index=False, float_format="%.8f"
    )
    reference.save_plots(frames, labels, INSTRUMENT, "2026", OUTPUT_DIR)
    print(f"{INSTRUMENT}: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
