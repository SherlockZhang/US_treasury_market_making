"""Backtest 2Y cost-adjusted winners including the expanded CDF rounding grid."""

import argparse
from pathlib import Path

import numpy as np

import analyze_2y_cost_adjusted_best_strategies as previous
import analyze_best_four_futures_strategies as reference
import search_cdf_custom_vol_daily_weekly as cdf_grid


CDF_BEST = reference.RESULTS / "cdf_custom_vol_rounding_daily_weekly_grid_2016_2025" / "best_parameters.csv"
OUTPUT = reference.RESULTS / "2y_best_four_expanded_rounding_grid_2016_2025_and_2026"


def selected_parameters(
    instrument: str = previous.INSTRUMENT,
    *,
    cdf_best: Path = CDF_BEST,
    ewma_best: Path = previous.EWMA_BEST,
    kama_best: Path = previous.KAMA_BEST,
) -> tuple[
    dict[str, dict[str, int]],
    tuple[reference.cdf.RebalanceFrequency, reference.cdf.RebalanceFrequency],
    tuple[float, float],
    dict[str, float],
]:
    if instrument not in cdf_grid.CONFIGURATIONS:
        raise ValueError(f"Unsupported instrument: {instrument}.")
    target, _ = cdf_grid.CONFIGURATIONS[instrument]
    scaled = previous.selected_row(cdf_best, True, instrument)
    unscaled = previous.selected_row(cdf_best, False, instrument)
    ewma = previous.selected_row(ewma_best, instrument=instrument)
    kama = previous.selected_row(kama_best, instrument=instrument)
    for row in (scaled, unscaled):
        if (
            row["sharpe_status"] != "ok"
            or row["position_rounding_increment"] not in reference.cdf.POSITION_ROUNDING_INCREMENTS
            or row["volatility_lookback_days"] != reference.CDF_VOLATILITY_LOOKBACK
            or row["signal_cutoff"] != 0
            or row["rebalance_threshold_fraction"] != 0
        ):
            raise ValueError("Selected CDF grid settings do not match the report.")
    if not np.isclose(scaled["target_annualized_volatility_pct"], target * 100):
        raise ValueError("Selected scaled CDF target does not match the report.")
    scaled_frequency = scaled["rebalance_frequency"]
    unscaled_frequency = unscaled["rebalance_frequency"]
    if scaled_frequency not in ("daily", "weekly") or unscaled_frequency not in ("daily", "weekly"):
        raise ValueError("Selected CDF rebalance frequency is unsupported.")
    parameters = {
        "Normal CDF scaled": {"lookback": int(scaled["signal_lookback_days"])},
        "Normal CDF unscaled": {"lookback": int(unscaled["signal_lookback_days"])},
        "EWMA": {"n": int(ewma["n"]), "m": int(ewma["m"])},
        "KAMA": {
            "fast": int(kama["alpha_fast_n"]), "slow": int(kama["alpha_slow_n"]),
            "er": int(kama["er_lookback_days"]), "m": int(kama["kama_extreme_window_m"]),
        },
    }
    sharpes = {
        name: float(row["sharpe_ratio"])
        for name, row in zip(reference.STRATEGIES, (scaled, unscaled, ewma, kama))
    }
    return (
        parameters, (scaled_frequency, unscaled_frequency),
        (float(scaled["position_rounding_increment"]), float(unscaled["position_rounding_increment"])),
        sharpes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    parameters, frequencies, rounding, sharpes = selected_parameters()
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[previous.INSTRUMENT]
    frames = reference.run_period(
        prices.loc[:reference.TRAIN_END], parameters, reference.START,
        rounding[0], rounding[1], previous.TARGET, frequencies,
    )
    for name, frame in frames.items():
        actual = reference.statistics(frame)["sharpe_ratio"]
        if not np.isclose(actual, sharpes[name], rtol=0, atol=1e-8):
            raise ValueError(f"{name} training Sharpe {actual} does not match grid winner {sharpes[name]}.")
    reference.save_report(
        previous.INSTRUMENT, prices, parameters, args.output_dir,
        rounding[0], rounding[1], previous.TARGET, frequencies,
    )


if __name__ == "__main__":
    main()
