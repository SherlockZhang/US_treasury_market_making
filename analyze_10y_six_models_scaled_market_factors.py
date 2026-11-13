"""Compare six 10Y rank-sum winners in 2026, with factors on scaled CDF models."""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_10y_six_best_models_normal_grid as charts
import analyze_2y_all_rank_sum_six_models as trends
import analyze_2y_four_cdf_modes_cap3 as report
import analyze_5y_10y_all_rank_sum_six_models as previous
import analyze_best_cdf_market_factor_comparison as factor_description
import analyze_best_four_futures_strategies as reference
import analyze_rank_sum_scaled_cdf_market_factors as factors


INSTRUMENT = "10Y future"
SOURCE = previous.OUTPUTS[INSTRUMENT]
OUTPUT = reference.RESULTS / "10y_six_models_scaled_market_factors_all_rank_sum_2026"


def verify_saved_frame(frame: pd.DataFrame, path: Path) -> None:
    saved = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if not frame.index.equals(saved.index):
        raise ValueError(f"Backtest dates do not match {path}.")
    for column in ("position", "strategy_return", "gross_strategy_return", "trading_cost"):
        if not np.allclose(frame[column], saved[column], rtol=0, atol=1e-9):
            raise ValueError(f"Backtest {column} does not match {path}.")


def build_frames(prices: pd.Series) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    winners = pd.read_csv(SOURCE / "cdf_best_parameters.csv")
    expected_modes = {(scaled, frequency) for scaled in (True, False) for frequency in factors.FREQUENCIES}
    actual_modes = set(zip(winners["volatility_scaling"], winners["rebalance_frequency"]))
    if len(winners) != 4 or actual_modes != expected_modes or not winners["selected_best"].eq(True).all():
        raise ValueError("Expected one rank-sum winner for each 10Y CDF mode.")
    ewma = trends.selected_trend_winner("EWMA", INSTRUMENT)
    kama = trends.selected_trend_winner("KAMA", INSTRUMENT)
    for model, row, keys in (
        ("ewma", ewma, ("n", "m")),
        ("kama", kama, ("alpha_fast_n", "alpha_slow_n", "er_lookback_days", "kama_extreme_window_m")),
    ):
        saved = pd.read_csv(SOURCE / f"{model}_best_parameters.csv")
        if len(saved) != 1 or any(saved.iloc[0][key] != row[key] for key in keys):
            raise ValueError(f"The current {model.upper()} winner differs from the prior report.")
    frames, labels = report.build_frames(
        prices, winners, reference.TEST_START, reference.TEST_END,
        instrument=INSTRUMENT, target=0.05, position_cap=None,
        ewma_winner=ewma, kama_winner=kama,
    )
    for name, frame in frames.items():
        slug = charts.MODEL_SLUGS.get(name, name.lower().replace(" ", "_").replace("=", ""))
        verify_saved_frame(frame, SOURCE / f"2026_{slug}_daily.csv")

    active_prices = prices.loc[reference.START:reference.TEST_END]
    factor_data = tuple(
        (spec, reference.cdf.calculate_instrument_market_factor(spec, active_prices.index))
        for spec in reference.cdf.INSTRUMENT_MARKET_FACTOR_PROFILES[INSTRUMENT]
    )
    for frequency in factors.FREQUENCIES:
        row = factors.selected_row(INSTRUMENT, frequency)
        name = charts.model_name(row)
        pair = factors.backtest_pair(INSTRUMENT, frequency, active_prices, row, factor_data)
        factors.verify_baseline(INSTRUMENT, frequency, int(row["signal_lookback_days"]), pair["none"])
        frame = pair["instrument"]
        verify_saved_frame(
            frame, factors.OUTPUT / frequency / "10y_with_market_factors_daily.csv",
        )
        frame["position_changed"] = frame["position_change"].ne(0)
        frames[name] = frame
        labels[name] += "; instrument market factors"
    return frames, labels


def save_report(directory: Path = OUTPUT) -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    frames, labels = build_frames(prices)
    directory.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, frame in frames.items():
        slug = charts.MODEL_SLUGS.get(name, name.lower().replace(" ", "_").replace("=", ""))
        frame.to_csv(directory / f"2026_{slug}_daily.csv", index_label="date", float_format="%.10f")
        has_factors = name.startswith("Normal CDF scaled ")
        summary.append({
            "period": "2026",
            "instrument": INSTRUMENT,
            "strategy": name,
            "parameters": labels[name],
            "market_factor_mode": "instrument" if has_factors else "none",
            "market_factor_profile": factor_description.profile_description(INSTRUMENT) if has_factors else "",
            **reference.statistics(frame),
            **reference.trading_statistics(frame),
        })
    pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False, float_format="%.10f")
    for model in ("cdf", "ewma", "kama"):
        pd.read_csv(SOURCE / f"{model}_best_parameters.csv").to_csv(
            directory / f"{model}_best_parameters.csv", index=False, float_format="%.10f",
        )
    coverage = pd.read_csv(factors.OUTPUT / "market_factor_data_coverage.csv")
    coverage = coverage.loc[coverage["instrument"].eq(INSTRUMENT)]
    if len(coverage) != len(reference.cdf.INSTRUMENT_MARKET_FACTOR_PROFILES[INSTRUMENT]):
        raise ValueError("10Y market-factor coverage is incomplete.")
    coverage.to_csv(directory / "market_factor_data_coverage.csv", index=False)
    charts.save_plots(frames, labels, "2026", directory, instrument=INSTRUMENT)
    print(f"{INSTRUMENT}: {directory}")


if __name__ == "__main__":
    save_report()
