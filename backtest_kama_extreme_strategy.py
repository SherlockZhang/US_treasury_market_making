"""Backtest a KAMA rolling-extreme strategy on Treasury futures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest_normal_cdf_signal as normal_cdf


DEFAULT_FAST_N = 2
DEFAULT_SLOW_N = 30
DEFAULT_ER_LOOKBACK_DAYS = 30
DEFAULT_EXTREME_LOOKBACK_DAYS = 30
OUTPUT_DIR = Path(__file__).with_name("presentation report")


def positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def calculate_kama(
    prices: pd.Series,
    fast_n: int = DEFAULT_FAST_N,
    slow_n: int = DEFAULT_SLOW_N,
    er_lookback_days: int = DEFAULT_ER_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Calculate Kaufman adaptive moving average and its efficiency ratio."""
    if fast_n <= 0 or slow_n <= 0:
        raise ValueError("KAMA fast and slow periods must be positive integers.")
    if fast_n > slow_n:
        raise ValueError("KAMA fast period must not exceed its slow period.")
    if er_lookback_days <= 0:
        raise ValueError("ER lookback must be a positive integer.")
    if prices.empty:
        raise ValueError("No prices are available for the KAMA calculation.")
    if (
        prices.isna().any()
        or not np.isfinite(prices).all()
        or (prices <= 0).any()
    ):
        raise ValueError("KAMA prices must be finite, non-missing, and positive.")
    if prices.index.has_duplicates:
        raise ValueError("KAMA price data contains duplicate dates.")

    prices = prices.sort_index().rename("price")
    change = prices.diff(er_lookback_days).abs()
    volatility = prices.diff().abs().rolling(
        window=er_lookback_days,
        min_periods=er_lookback_days,
    ).sum()
    efficiency_ratio = (
        change.div(volatility.where(volatility.ne(0)))
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .rename("efficiency_ratio")
    )
    fast_smoothing_constant = 2 / (fast_n + 1)
    slow_smoothing_constant = 2 / (slow_n + 1)
    smoothing_constant = (
        efficiency_ratio * (fast_smoothing_constant - slow_smoothing_constant)
        + slow_smoothing_constant
    ).pow(2).rename("kama_smoothing_constant")

    kama_values = np.empty(len(prices), dtype=float)
    kama_values[0] = prices.iloc[0]
    for index in range(1, len(prices)):
        kama_values[index] = kama_values[index - 1] + smoothing_constant.iloc[
            index
        ] * (prices.iloc[index] - kama_values[index - 1])
    kama = pd.Series(kama_values, index=prices.index, name="kama")
    return pd.concat([kama, efficiency_ratio, smoothing_constant], axis=1)


def run_backtest(
    prices: pd.Series,
    fast_n: int = DEFAULT_FAST_N,
    slow_n: int = DEFAULT_SLOW_N,
    er_lookback_days: int = DEFAULT_ER_LOOKBACK_DAYS,
    extreme_lookback_days: int = DEFAULT_EXTREME_LOOKBACK_DAYS,
    start_date: pd.Timestamp | None = None,
    trading_cost_ratio: float | None = None,
) -> pd.DataFrame:
    """Go long at rolling KAMA highs and short at lows, starting next day."""
    instrument = prices.name
    if extreme_lookback_days <= 0:
        raise ValueError("Extreme lookback must be a positive integer.")
    kama_data = calculate_kama(prices, fast_n, slow_n, er_lookback_days)
    prices = prices.sort_index().rename("price")
    kama = kama_data["kama"]
    rolling_minimum = kama.rolling(
        window=extreme_lookback_days,
        min_periods=extreme_lookback_days,
    ).min().rename("kama_rolling_minimum")
    rolling_maximum = kama.rolling(
        window=extreme_lookback_days,
        min_periods=extreme_lookback_days,
    ).max().rename("kama_rolling_maximum")
    at_minimum = kama.eq(rolling_minimum)
    at_maximum = kama.eq(rolling_maximum)
    short_signal = (at_minimum & ~at_maximum).rename("short_entry_signal")
    long_signal = (at_maximum & ~at_minimum).rename("long_entry_signal")
    entry_signal = pd.Series(0, index=prices.index, dtype=np.int8)
    entry_signal.loc[long_signal] = 1
    entry_signal.loc[short_signal] = -1

    first_position = 0
    if start_date is not None:
        first_position = int(prices.index.searchsorted(start_date))
        if first_position >= len(prices):
            raise ValueError(
                f"No prices are available on or after {start_date:%Y-%m-%d}."
            )

    position = np.zeros(len(prices), dtype=np.int8)
    current_position = 0
    for index in range(first_position, len(prices)):
        if index > first_position:
            previous_signal = entry_signal.iloc[index - 1]
            if current_position == 0:
                current_position = int(previous_signal)
            elif current_position == 1 and previous_signal == -1:
                current_position = 0
            elif current_position == -1 and previous_signal == 1:
                current_position = 0
        elif index > 0:
            current_position = int(entry_signal.iloc[index - 1])
        position[index] = current_position

    results = pd.concat(
        [
            prices,
            kama_data,
            rolling_minimum,
            rolling_maximum,
            long_signal,
            short_signal,
        ],
        axis=1,
    )
    results["position"] = position
    results["daily_return"] = prices.pct_change(fill_method=None)
    normal_cdf.apply_trading_cost(results, instrument, trading_cost_ratio)
    results["wealth"] = (1 + results["strategy_return"]).cumprod()
    if (results["wealth"].iloc[first_position:] <= 0).any():
        raise ValueError(
            "The strategy's compounded wealth reached zero or less; "
            "check for extreme daily returns."
        )
    results["cumulative_pnl_pct"] = (results["wealth"] - 1) * 100
    results["running_peak"] = results["wealth"].cummax()
    results["drawdown"] = results["wealth"] / results["running_peak"] - 1
    return results


def save_outputs(
    instrument: str,
    results: pd.DataFrame,
    fast_n: int,
    slow_n: int,
    er_lookback_days: int,
    extreme_lookback_days: int,
    start_date: pd.Timestamp | None,
) -> dict[str, object]:
    """Save daily results, a cumulative PnL chart, and summary metrics."""
    if start_date is not None:
        results = results.loc[results.index >= start_date].copy()
    if results.empty:
        raise ValueError(f"No {instrument} results fall in the requested period.")

    date_tag = (
        f"from{start_date:%Y%m%d}" if start_date is not None else "full_history"
    )
    stem = (
        f"{instrument.split()[0].lower()}_future_kama_extreme_"
        f"fast{fast_n}_slow{slow_n}_er{er_lookback_days}_"
        f"m{extreme_lookback_days}_{date_tag}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / f"{stem}_daily.csv"
    plot_path = OUTPUT_DIR / f"{stem}_pnl.png"
    results.to_csv(daily_path, index_label="Date", float_format="%.8f")

    figure, (price_axis, pnl_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )
    price_axis.plot(results.index, results["price"], label="Future price")
    price_axis.plot(
        results.index,
        results["kama"],
        label=f"KAMA (fast N={fast_n}, slow N={slow_n}, ER={er_lookback_days})",
    )
    price_axis.set_title(f"{instrument} Price and KAMA")
    price_axis.set_ylabel("Price")
    price_axis.legend()
    price_axis.grid(True, alpha=0.25)
    pnl_axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        label="KAMA extreme strategy",
        color="#245a84",
    )
    pnl_axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    pnl_axis.set_title(
        f"KAMA Extreme Strategy (M={extreme_lookback_days}) — Cumulative PnL"
    )
    pnl_axis.set_xlabel("Date")
    pnl_axis.set_ylabel("Cumulative PnL (%)")
    pnl_axis.legend()
    pnl_axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    metrics = normal_cdf.calculate_comparison_metrics(results["strategy_return"])
    return {
        "instrument": instrument,
        "start_date": results.index[0].date(),
        "end_date": results.index[-1].date(),
        "observations": len(results),
        "kama_fast_n": fast_n,
        "kama_slow_n": slow_n,
        "er_lookback_days": er_lookback_days,
        "extreme_lookback_m": extreme_lookback_days,
        **metrics,
        "daily_results": str(daily_path),
        "pnl_plot": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument",
        choices=tuple(normal_cdf.SINGLE_FUTURE_STRATEGIES),
        nargs="+",
        default=list(normal_cdf.SINGLE_FUTURE_STRATEGIES),
        help="Treasury futures to run (default: all three).",
    )
    parser.add_argument(
        "--fast-n",
        type=positive_int,
        default=DEFAULT_FAST_N,
        help=f"KAMA fast period (default: {DEFAULT_FAST_N}).",
    )
    parser.add_argument(
        "--slow-n",
        type=positive_int,
        default=DEFAULT_SLOW_N,
        help=f"KAMA slow period (default: {DEFAULT_SLOW_N}).",
    )
    parser.add_argument(
        "--er-lookback-days",
        type=positive_int,
        default=DEFAULT_ER_LOOKBACK_DAYS,
        help=f"Efficiency-ratio lookback (default: {DEFAULT_ER_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--extreme-lookback-days",
        type=positive_int,
        default=DEFAULT_EXTREME_LOOKBACK_DAYS,
        metavar="M",
        help=(
            "Trailing KAMA window for high/low entry signals "
            f"(default: {DEFAULT_EXTREME_LOOKBACK_DAYS})."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=normal_cdf.iso_date,
        help=(
            "First output date in YYYY-MM-DD format; earlier data is retained "
            "for indicator warmup (default: full history)."
        ),
    )
    args = parser.parse_args()

    prices_by_name = normal_cdf.load_single_future_prices(include_dbmf=False)
    summary_rows = []
    for instrument in args.instrument:
        results = run_backtest(
            prices_by_name[instrument],
            fast_n=args.fast_n,
            slow_n=args.slow_n,
            er_lookback_days=args.er_lookback_days,
            extreme_lookback_days=args.extreme_lookback_days,
            start_date=args.start_date,
            trading_cost_ratio=normal_cdf.TRADING_COST_RATIOS[instrument],
        )
        summary = save_outputs(
            instrument,
            results,
            args.fast_n,
            args.slow_n,
            args.er_lookback_days,
            args.extreme_lookback_days,
            args.start_date,
        )
        summary_rows.append(summary)
        print(
            f"{instrument}: {summary['start_date']} to {summary['end_date']}, "
            f"total return {summary['total_return_pct']:.2f}%, "
            f"Sharpe {summary['sharpe_ratio']:.4f}"
        )
        print(f"Daily results: {summary['daily_results']}")
        print(f"PnL plot: {summary['pnl_plot']}")

    date_tag = (
        f"from{args.start_date:%Y%m%d}"
        if args.start_date is not None
        else "full_history"
    )
    summary_path = OUTPUT_DIR / (
        f"kama_extreme_strategy_fast{args.fast_n}_slow{args.slow_n}_"
        f"er{args.er_lookback_days}_m{args.extreme_lookback_days}_"
        f"{date_tag}_stats.csv"
    )
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        float_format="%.6f",
    )
    print(f"Summary stats: {summary_path}")


if __name__ == "__main__":
    main()
