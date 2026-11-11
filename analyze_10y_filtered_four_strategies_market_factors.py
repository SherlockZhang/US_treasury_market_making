"""Compare 10Y filtered strategies in 2026 with factors on scaled CDF only."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_10y_four_weekly_t64_rounding_0p5 as selected
import analyze_2y_cost_adjusted_best_strategies as selection
import analyze_2y_expanded_cdf_grid_winners as expanded
import analyze_best_cdf_market_factor_comparison as factors
import analyze_best_four_futures_strategies as reference
import search_best_four_win_ratio_grid as search
import search_cdf_custom_vol_daily_weekly as cdf_grid


INSTRUMENT = selected.INSTRUMENT
OUTPUT = reference.RESULTS / "10y_four_gross_filtered_scaled_market_factors_2026"


def backtest(prices: pd.Series) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    grid_dir = search.OUTPUT_DIR
    cdf_path = grid_dir / "cdf_best_parameters.csv"
    ewma_path = grid_dir / "ewma_best_parameters.csv"
    kama_path = grid_dir / "kama_best_parameters.csv"
    parameters, frequencies, rounding, _ = expanded.selected_parameters(
        INSTRUMENT, cdf_best=cdf_path, ewma_best=ewma_path, kama_best=kama_path,
    )
    scaled = selected.selected_candidate()
    winners = (
        scaled, selection.selected_row(cdf_path, False, INSTRUMENT),
        selection.selected_row(ewma_path, instrument=INSTRUMENT),
        selection.selected_row(kama_path, instrument=INSTRUMENT),
    )
    if scaled["rebalance_frequency"] != "weekly":
        raise ValueError("The factor comparison requires the selected weekly scaled model.")
    parameters[reference.STRATEGIES[0]]["lookback"] = int(scaled["signal_lookback_days"])
    frequencies = ("weekly", frequencies[1])
    rounding = (float(scaled["position_rounding_increment"]), rounding[1])
    target, _ = cdf_grid.CONFIGURATIONS[INSTRUMENT]
    prices = prices.rename(INSTRUMENT)
    training = reference.run_period(
        prices.loc[:reference.TRAIN_END], parameters, reference.START,
        rounding[0], rounding[1], target, frequencies,
    )
    for name, row in zip(reference.STRATEGIES, winners):
        if not row["eligible"] or row["selection_status"] != "eligible" or row["win_ratio_pct"] <= 50:
            raise ValueError(f"{name} is not an eligible selected grid candidate.")
        stats = reference.statistics(training[name])
        for metric in ("win_ratio_pct", "sharpe_ratio"):
            if not np.isclose(stats[metric], row[metric], rtol=0, atol=1e-8):
                raise ValueError(f"{name} training {metric} does not match the grid.")
    frames = reference.run_period(
        prices.loc[:reference.TEST_END], parameters, reference.TEST_START,
        rounding[0], rounding[1], target, frequencies,
    )
    pair = factors.backtest_pair(
        INSTRUMENT, prices, int(scaled["signal_lookback_days"]), target, rounding[0]
    )
    pd.testing.assert_series_equal(
        frames[reference.STRATEGIES[0]]["strategy_return"],
        pair["none"]["strategy_return"],
    )
    frames[reference.STRATEGIES[0]] = pair["instrument"]
    labels = reference.parameter_labels(parameters, rounding[0], rounding[1], target, frequencies)
    labels[reference.STRATEGIES[0]] += "; instrument market factors"
    return frames, labels


def save_report(directory: Path) -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    frames, labels = backtest(prices)
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, slug in zip(reference.STRATEGIES, reference.SLUGS):
        frame = frames[name]
        frame["position_changed"] = frame["position_change"].ne(0)
        frame.to_csv(directory / f"2026_{slug}_daily.csv", index_label="date", float_format="%.10f")
        summaries.append({
            "period": "2026", "instrument": INSTRUMENT, "strategy": name,
            "parameters": labels[name],
            "market_factor_mode": "instrument" if name == reference.STRATEGIES[0] else "none",
            "market_factor_profile": factors.profile_description(INSTRUMENT)
            if name == reference.STRATEGIES[0] else "",
            **reference.statistics(frame), **reference.trading_statistics(frame),
        })
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False, float_format="%.10f")
    reference.save_plots(frames, labels, INSTRUMENT, "2026", directory)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nOutputs: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    save_report(args.output_dir)


if __name__ == "__main__":
    main()
