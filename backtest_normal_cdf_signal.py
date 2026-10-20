"""Backtest a normal-CDF momentum signal on 5Y and 10Y-2Y prices."""

import argparse
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_2y_10y_spread_strategy import load_spread
from backtest_5y_trend_strategy import (
    TARGET_ANNUALIZED_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    calculate_annual_statistics,
    calculate_statistics,
    load_5y_yield,
    load_prices,
    load_spx_index,
    run_backtest as run_trend_backtest,
    summarize_5y_yield_change,
    summarize_spx_change,
)


OUTPUT_DIR = Path(__file__).with_name("results_refining_parameter")
DEFAULT_LOOKBACK_DAYS = 252
AVERAGED_SIGNAL_LOOKBACK_DAYS = (32, 64, 126, 252, 504)
NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS = 63
NORMAL_CDF_SIGNAL_CUTOFF = 0.1
REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION = 0.25
RebalanceFrequency = Literal["daily", "weekly", "monthly"]
SignalMode = Literal["single", "average"]
DEFAULT_REBALANCE_FREQUENCY: RebalanceFrequency = "weekly"

STRATEGIES = {
    "5y_future": {
        "instrument": "5Y future",
        "stem": "5y_future_normal_cdf_signal",
        "color": "#245a84",
    },
    "10y_2y_spread": {
        "instrument": "5x2Y - 3x10Y futures basket",
        "stem": "5x2y_minus_3x10y_normal_cdf_signal",
        "color": "#8c4a2f",
    },
}


def positive_int(value: str) -> int:
    """Parse a positive integer lookback for the command-line option."""
    try:
        days = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "lookback days must be a positive integer"
        ) from error
    if days <= 0:
        raise argparse.ArgumentTypeError("lookback days must be a positive integer")
    return days


def nonnegative_float(value: str) -> float:
    """Parse a finite non-negative position-change threshold fraction."""
    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "threshold fraction must be a finite non-negative number"
        ) from error
    if not np.isfinite(fraction) or fraction < 0:
        raise argparse.ArgumentTypeError(
            "threshold fraction must be a finite non-negative number"
        )
    return fraction


def calculate_normal_cdf_signal(
    daily_returns: pd.Series,
    lookback_days: int,
    signal_cutoff: float = NORMAL_CDF_SIGNAL_CUTOFF,
) -> pd.DataFrame:
    """Calculate rolling t-statistics and map them to signals in [-1, 1]."""
    if lookback_days <= 0:
        raise ValueError("Lookback days must be a positive integer.")
    if not np.isfinite(signal_cutoff) or signal_cutoff < 0:
        raise ValueError("Signal cutoff must be finite and non-negative.")

    rolling_returns = daily_returns.rolling(
        window=lookback_days, min_periods=lookback_days
    )
    average_return = rolling_returns.mean()
    return_volatility = rolling_returns.std(ddof=1)
    t_stat = pd.Series(
        np.sqrt(lookback_days) * average_return / return_volatility,
        index=daily_returns.index,
    )
    unfiltered_signal = pd.Series(
        3 * norm.cdf(t_stat) - 1.5, index=daily_returns.index
    )
    unfiltered_signal = unfiltered_signal.where(return_volatility.gt(0))
    signal = unfiltered_signal.where(
        unfiltered_signal.isna()
        | (unfiltered_signal.abs() > signal_cutoff),
        0.0,
    )

    return pd.DataFrame(
        {
            "average_return": average_return,
            "lookback_volatility": return_volatility,
            "t_stat": t_stat,
            "unfiltered_signal": unfiltered_signal,
            "signal": signal,
        },
        index=daily_returns.index,
    )


def run_backtest(
    prices: pd.Series,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    rebalance_frequency: RebalanceFrequency = DEFAULT_REBALANCE_FREQUENCY,
    position_change_threshold_fraction: float = (
        REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION
    ),
    signal_cutoff: float = NORMAL_CDF_SIGNAL_CUTOFF,
    signal_mode: SignalMode = "single",
) -> pd.DataFrame:
    """Apply the signal with daily, Monday-weekly, or month-end volatility sizing."""
    if lookback_days <= 0:
        raise ValueError("Lookback days must be a positive integer.")
    if rebalance_frequency not in ("daily", "weekly", "monthly"):
        raise ValueError(
            f"Unsupported rebalance frequency: {rebalance_frequency}"
        )
    if (
        not np.isfinite(position_change_threshold_fraction)
        or position_change_threshold_fraction < 0
    ):
        raise ValueError(
            "Position-change threshold fraction must be finite and non-negative."
        )
    if not np.isfinite(signal_cutoff) or signal_cutoff < 0:
        raise ValueError("Signal cutoff must be finite and non-negative.")
    if signal_mode not in ("single", "average"):
        raise ValueError(f"Unsupported signal mode: {signal_mode}")
    if prices.empty:
        raise ValueError("No prices are available for this backtest.")
    if prices.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Backtest prices must be finite, non-missing, and positive.")
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")

    prices = prices.sort_index()
    results = prices.rename("price").to_frame()
    dates = pd.DatetimeIndex(results.index)
    results.index = dates
    results["daily_return"] = results["price"].pct_change(fill_method=None)
    signal_data = calculate_normal_cdf_signal(
        results["daily_return"], lookback_days, signal_cutoff
    )
    if signal_mode == "average":
        component_signals = pd.DataFrame(
            {
                f"signal_t{component_lookback}": calculate_normal_cdf_signal(
                    results["daily_return"],
                    component_lookback,
                    signal_cutoff,
                )["signal"]
                for component_lookback in AVERAGED_SIGNAL_LOOKBACK_DAYS
            },
            index=results.index,
        )
        signal_data["signal"] = component_signals.mean(axis=1, skipna=False)
        signal_data = signal_data.join(component_signals)
    results = results.join(signal_data)
    results["realized_volatility"] = (
        results["daily_return"]
        .rolling(
            NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
            min_periods=NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
        )
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    eligible = results["signal"].notna() & results["realized_volatility"].gt(0)
    desired_position = pd.Series(np.nan, index=results.index, dtype=float)
    desired_position.loc[eligible] = (
        results.loc[eligible, "signal"]
        * TARGET_ANNUALIZED_VOLATILITY
        / results.loc[eligible, "realized_volatility"]
    )
    if rebalance_frequency == "daily":
        rebalance_dates = np.ones(len(dates), dtype=bool)
    elif rebalance_frequency == "weekly":
        rebalance_dates = dates.dayofweek == 0
    else:
        months = dates.to_period("M")
        rebalance_dates = np.r_[
            months[:-1] != months[1:],
            True,
        ]
    position_change_threshold = (
        position_change_threshold_fraction
        * TARGET_ANNUALIZED_VOLATILITY
        / results["realized_volatility"]
    )
    position_change_threshold = position_change_threshold.where(rebalance_dates)
    target_position = np.zeros(len(results), dtype=float)
    current_target_position = 0.0
    desired_position_values = desired_position.to_numpy()
    threshold_values = position_change_threshold.to_numpy()
    for index, is_rebalance_date in enumerate(rebalance_dates):
        if not is_rebalance_date:
            target_position[index] = current_target_position
            continue

        candidate = desired_position_values[index]
        threshold = threshold_values[index]
        if np.isfinite(candidate) and np.isfinite(threshold):
            if abs(candidate - current_target_position) >= threshold:
                current_target_position = candidate
        target_position[index] = current_target_position

    results["desired_position"] = desired_position
    results["position_change_threshold"] = position_change_threshold
    results["target_position"] = pd.Series(target_position, index=results.index)
    position_at_close = results["target_position"]
    results["position"] = position_at_close.shift(1).fillna(0.0)
    results["strategy_return"] = (
        results["position"] * results["daily_return"]
    ).fillna(0.0)

    results["wealth"] = (1 + results["strategy_return"]).cumprod()
    if (results["wealth"] <= 0).any():
        raise ValueError(
            "The strategy's compounded wealth reached zero or less; "
            "check for extreme returns or volatility-target leverage."
        )
    results["cumulative_pnl_pct"] = (results["wealth"] - 1) * 100
    results["running_peak"] = results["wealth"].cummax()
    results["drawdown"] = results["wealth"] / results["running_peak"] - 1
    results["yearly_wealth"] = results.groupby(dates.year)[
        "strategy_return"
    ].transform(lambda returns: (1 + returns).cumprod())
    results["yearly_cumulative_pnl_pct"] = (
        results["yearly_wealth"] - 1
    ) * 100
    return results


def calculate_stats(
    results: pd.DataFrame,
    instrument: str,
    lookback_days: int,
    rebalance_frequency: RebalanceFrequency,
    position_change_threshold_fraction: float,
    signal_cutoff: float,
    signal_mode: SignalMode,
) -> pd.DataFrame:
    """Calculate full-period and annual statistics."""
    full_stats = calculate_statistics(results)
    annual_stats = calculate_annual_statistics(results)
    full_stats["instrument"] = instrument
    full_stats["signal_mode"] = signal_mode
    full_stats["signal_lookback_days"] = (
        lookback_days if signal_mode == "single" else np.nan
    )
    full_stats["signal_lookbacks_days"] = ",".join(
        str(days)
        for days in (
            (lookback_days,)
            if signal_mode == "single"
            else AVERAGED_SIGNAL_LOOKBACK_DAYS
        )
    )
    full_stats["volatility_lookback_days"] = (
        NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS
    )
    full_stats["signal_cutoff"] = signal_cutoff
    full_stats["rebalance_position_change_threshold_fraction"] = (
        position_change_threshold_fraction
    )
    full_stats["rebalance_frequency"] = rebalance_frequency
    for annual_stat in annual_stats:
        annual_stat["instrument"] = instrument
        annual_stat["signal_mode"] = signal_mode
        annual_stat["signal_lookback_days"] = (
            lookback_days if signal_mode == "single" else np.nan
        )
        annual_stat["signal_lookbacks_days"] = full_stats[
            "signal_lookbacks_days"
        ]
        annual_stat["volatility_lookback_days"] = (
            NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS
        )
        annual_stat["signal_cutoff"] = signal_cutoff
        annual_stat["rebalance_position_change_threshold_fraction"] = (
            position_change_threshold_fraction
        )
        annual_stat["rebalance_frequency"] = rebalance_frequency

    yields = load_5y_yield()
    spx = load_spx_index()
    yield_dates = pd.DatetimeIndex(yields.index)
    spx_dates = pd.DatetimeIndex(spx.index)
    full_stats.update(summarize_5y_yield_change(yields))
    full_stats.update(summarize_spx_change(spx))
    for annual_stat in annual_stats:
        year = int(cast(str, annual_stat["period"]))
        annual_yields = yields.loc[yield_dates.year == year]
        annual_spx = spx.loc[spx_dates.year == year]
        annual_stat.update(summarize_5y_yield_change(annual_yields))
        annual_stat.update(summarize_spx_change(annual_spx))
    return pd.DataFrame([full_stats, *annual_stats])


def save_plots(
    results: pd.DataFrame,
    prices: pd.Series,
    instrument: str,
    color: str,
    rebalance_frequency: RebalanceFrequency,
    position_change_threshold_fraction: float,
    signal_cutoff: float,
    signal_label: str,
    plot_path: Path,
    comparison_path: Path,
) -> None:
    """Save cumulative PnL and comparison against the original trend strategy."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        color=color,
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"{instrument} Normal-CDF Signal ({signal_label}) — "
        f"{rebalance_frequency.capitalize()} Rebalance — "
        f"{NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS}-Day Volatility Scaling — "
        f"Signal Cutoff ±{signal_cutoff:.1f} — "
        f"{position_change_threshold_fraction:.0%} Rebalance Threshold — "
        "Full-Period Cumulative PnL"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    original_results = run_trend_backtest(prices.rename("price").to_frame())
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        original_results.index,
        original_results["cumulative_pnl_pct"],
        label="1-year moving-average strategy",
        linewidth=1.5,
    )
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        label=(
            f"Normal-CDF signal ({signal_label}, "
            f"{rebalance_frequency} rebalance, |signal|>{signal_cutoff:.1f}, "
            f"{position_change_threshold_fraction:.0%} threshold)"
        ),
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(f"{instrument} Trend Strategies — Cumulative PnL Comparison")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)


def run_instrument(
    prices: pd.Series,
    strategy: dict[str, str],
    lookback_days: int,
    rebalance_frequency: RebalanceFrequency,
    position_change_threshold_fraction: float,
    signal_cutoff: float,
    signal_mode: SignalMode,
) -> None:
    """Run one instrument and save daily, summary, and plot outputs."""
    instrument = strategy["instrument"]
    signal_lookbacks = (
        (lookback_days,)
        if signal_mode == "single"
        else AVERAGED_SIGNAL_LOOKBACK_DAYS
    )
    signal_label = (
        f"T={lookback_days}"
        if signal_mode == "single"
        else f"average T={','.join(str(days) for days in signal_lookbacks)}"
    )
    lookback_tag = (
        f"{lookback_days}d"
        if signal_mode == "single"
        else f"avg{'-'.join(str(days) for days in signal_lookbacks)}d"
    )
    stem = (
        f"{strategy['stem']}_{rebalance_frequency}_{lookback_tag}_"
        f"vol{NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS}d_"
        f"step{int(position_change_threshold_fraction * 100)}pct_"
        f"sig{int(signal_cutoff * 100)}"
    )
    results = run_backtest(
        prices,
        lookback_days,
        rebalance_frequency,
        position_change_threshold_fraction,
        signal_cutoff,
        signal_mode,
    )
    stats_frame = calculate_stats(
        results,
        instrument,
        lookback_days,
        rebalance_frequency,
        position_change_threshold_fraction,
        signal_cutoff,
        signal_mode,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / f"{stem}_daily.csv"
    stats_path = OUTPUT_DIR / f"{stem}_stats.csv"
    simple_stats_path = OUTPUT_DIR / f"{stem}_stats_simple.csv"
    plot_path = OUTPUT_DIR / f"{stem}_cumulative_pnl.png"
    comparison_path = OUTPUT_DIR / f"{stem}_vs_original_pnl.png"

    results.to_csv(daily_path, index_label="Date", float_format="%.8f")
    stats_frame.to_csv(stats_path, index=False, float_format="%.6f")
    stats_frame.loc[
        :,
        [
            "period",
            "signal_mode",
            "signal_lookback_days",
            "signal_lookbacks_days",
            "volatility_lookback_days",
            "signal_cutoff",
            "rebalance_position_change_threshold_fraction",
            "rebalance_frequency",
            "sharpe_ratio",
            "max_drawdown_pct",
            "5y_yield_change_percentage_points",
            "spx_change_index_points",
        ],
    ].to_csv(simple_stats_path, index=False, float_format="%.6f")
    save_plots(
        results,
        prices,
        instrument,
        strategy["color"],
        rebalance_frequency,
        position_change_threshold_fraction,
        signal_cutoff,
        signal_label,
        plot_path,
        comparison_path,
    )

    full_stats = stats_frame.iloc[0]
    print(
        f"{instrument} ({signal_label}, {rebalance_frequency} rebalance): "
        f"{full_stats['data_start_date']} to {full_stats['data_end_date']}, "
        f"Sharpe {full_stats['sharpe_ratio']:.4f}, max drawdown "
        f"{full_stats['max_drawdown_pct']:.2f}%, total return "
        f"{full_stats['total_return_pct']:.2f}%"
    )
    print(f"Daily results: {daily_path}")
    print(f"Stats: {stats_path}")
    print(f"Simplified stats: {simple_stats_path}")
    print(f"PnL plot: {plot_path}")
    print(f"Comparison plot: {comparison_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookback-days",
        type=positive_int,
        default=DEFAULT_LOOKBACK_DAYS,
        metavar="T",
        help=f"Rolling return lookback in trading days (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--signal-mode",
        choices=("single", "average"),
        default="single",
        help=(
            "Use one lookback signal or average signals for T="
            f"{','.join(str(days) for days in AVERAGED_SIGNAL_LOOKBACK_DAYS)} "
            "(default: single)."
        ),
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=("daily", "weekly", "monthly"),
        default=DEFAULT_REBALANCE_FREQUENCY,
        help=(
            "Position rebalance frequency; monthly rebalances on each month's "
            f"last available date (default: {DEFAULT_REBALANCE_FREQUENCY})."
        ),
    )
    parser.add_argument(
        "--rebalance-threshold-fraction",
        type=nonnegative_float,
        default=REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION,
        metavar="FRACTION",
        help=(
            "Minimum position change as a fraction of target volatility divided "
            f"by realized volatility; use 0 to disable (default: "
            f"{REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION})."
        ),
    )
    parser.add_argument(
        "--signal-cutoff",
        type=nonnegative_float,
        default=NORMAL_CDF_SIGNAL_CUTOFF,
        metavar="CUTOFF",
        help=(
            "Minimum absolute normal-CDF signal to retain "
            f"(default: {NORMAL_CDF_SIGNAL_CUTOFF})."
        ),
    )
    args = parser.parse_args()
    rebalance_frequency: RebalanceFrequency = args.rebalance_frequency
    signal_mode: SignalMode = args.signal_mode

    five_year_prices = load_prices()["price"]
    spread_prices = load_spread()["weighted_spread"]
    run_instrument(
        five_year_prices,
        STRATEGIES["5y_future"],
        args.lookback_days,
        rebalance_frequency,
        args.rebalance_threshold_fraction,
        args.signal_cutoff,
        signal_mode,
    )
    run_instrument(
        spread_prices,
        STRATEGIES["10y_2y_spread"],
        args.lookback_days,
        rebalance_frequency,
        args.rebalance_threshold_fraction,
        args.signal_cutoff,
        signal_mode,
    )


if __name__ == "__main__":
    main()
