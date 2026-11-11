"""Compare the selected 2026 scaled CDF models with and without market factors."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_10y_four_weekly_t64_rounding_0p5 as ten_year
import analyze_2y_cost_adjusted_best_strategies as selection
import analyze_best_cdf_market_factor_comparison as comparison
import analyze_best_four_futures_strategies as reference
import search_best_four_win_ratio_grid as search
import search_cdf_custom_vol_daily_weekly as cdf_grid


OUTPUT = reference.RESULTS / "cdf_2026_gross_filtered_winners_market_factor_comparison"


def selected_row(instrument: str) -> pd.Series:
    if instrument == "10Y future":
        row = ten_year.selected_candidate()
    else:
        row = selection.selected_row(search.OUTPUT_DIR / "cdf_best_parameters.csv", True, instrument)
    target, _ = cdf_grid.CONFIGURATIONS[instrument]
    if (
        not row["eligible"] or row["selection_status"] != "eligible" or row["win_ratio_pct"] <= 50
        or row["rebalance_frequency"] != "weekly"
        or row["sharpe_status"] != "ok"
        or row["volatility_lookback_days"] != reference.CDF_VOLATILITY_LOOKBACK
        or row["signal_cutoff"] != 0 or row["rebalance_threshold_fraction"] != 0
        or row["position_rounding_increment"] not in reference.cdf.POSITION_ROUNDING_INCREMENTS
        or not np.isclose(row["target_annualized_volatility_pct"], target * 100)
        or not np.isclose(row["trading_cost_ratio"], reference.cdf.TRADING_COST_RATIOS[instrument], rtol=0, atol=1e-12)
        or row["end_date"] != "2025-12-31"
    ):
        raise ValueError(f"{instrument} selected settings do not match the factor comparison.")
    return row


def save_report(directory: Path) -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    summaries = []
    coverage = []
    directory.mkdir(parents=True, exist_ok=True)
    for instrument in comparison.INSTRUMENTS:
        row = selected_row(instrument)
        target = float(row["target_annualized_volatility_pct"]) / 100
        rounding = float(row["position_rounding_increment"])
        lookback = int(row["signal_lookback_days"])
        training = reference.cdf.run_backtest(
            prices[instrument].loc[reference.START:reference.TRAIN_END],
            lookback_days=lookback,
            rebalance_frequency="weekly",
            position_change_threshold_fraction=0.0, signal_cutoff=0.0,
            signal_mode="rounded", volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
            output_start_date=reference.START, position_rounding_increment=rounding,
            use_volatility_scaling=True, target_annualized_volatility=target,
            trading_cost_ratio=float(row["trading_cost_ratio"]),
        )
        training_stats = reference.statistics(training)
        for metric in ("sharpe_ratio", "win_ratio_pct"):
            if not np.isclose(training_stats[metric], row[metric], rtol=0, atol=1e-8):
                raise ValueError(f"{instrument} training {metric} does not match the selected grid row.")
        frames = comparison.backtest_pair(instrument, prices[instrument], lookback, target, rounding)
        stem = instrument.split()[0].lower()
        for mode, suffix in (("none", "without"), ("instrument", "with")):
            frame = frames[mode]
            frame["position_changed"] = frame["position_change"].ne(0)
            frame.to_csv(directory / f"{stem}_{suffix}_market_factors_daily.csv",
                         index_label="date", float_format="%.10f")
            summary = comparison.summarize(
                instrument, mode, lookback, frame, target, rounding,
                lookback_selection=(
                    "2016-2025 gross-win-filtered grid; user-selected rank 3"
                    if instrument == "10Y future"
                    else "2016-2025 gross-win-filtered grid winner"
                ),
            )
            summary.update(reference.trading_statistics(frame))
            summary["trading_cost_ratio"] = float(row["trading_cost_ratio"])
            summary["selected_grid_sharpe_rank"] = int(row["sharpe_rank"])
            summaries.append(summary)
        end = frames["none"].index[-1]
        for spec in reference.cdf.INSTRUMENT_MARKET_FACTOR_PROFILES[instrument]:
            series = reference.cdf.load_market_data_series(spec.ticker).loc[:end]
            if series.empty:
                raise ValueError(f"No {spec.ticker} market-factor data available by {end.date()}.")
            latest = series.index[-1]
            coverage.append({
                "instrument": instrument, "ticker": spec.ticker,
                "first_observation": series.index[0].date(),
                "last_observation": latest.date(), "backtest_end": end.date(),
                "calendar_days_since_last_observation": (end - latest).days,
            })
        comparison.save_plots(instrument, frames, lookback, directory)
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False, float_format="%.10f")
    pd.DataFrame(coverage).to_csv(directory / "market_factor_data_coverage.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nOutputs: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    save_report(args.output_dir)


if __name__ == "__main__":
    main()
