"""Reproduce the four-strategy comparison for individual Treasury futures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import backtest_ewma_extreme_strategy as ewma
import backtest_kama_extreme_strategy as kama
import backtest_normal_cdf_signal as cdf


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
START = pd.Timestamp("2016-01-01")
TRAIN_END = pd.Timestamp("2025-12-31")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-12-31")
STRATEGIES = ("Normal CDF scaled", "Normal CDF unscaled", "EWMA", "KAMA")
SLUGS = ("cdf_scaled", "cdf_unscaled", "ewma", "kama")
METRICS = (
    "sharpe_ratio",
    "max_drawdown_pct",
    "annualized_return_pct",
    "annualized_volatility_pct",
)
CDF_LOOKBACKS = (32, 64, 126, 252, 504)
CDF_VOLATILITY_LOOKBACK = 64
CDF_POSITION_ROUNDING = 0.5
CDF_SIGNAL_CUTOFF = 0.0
CDF_REBALANCE_THRESHOLD = 0.0
SCALED_CDF_LOOKBACK_OVERRIDES = {"2Y future": 64, "5Y future": 64}


def selected_row(filename: str, instrument: str) -> pd.Series:
    table = pd.read_csv(RESULTS / filename)
    rows = table.loc[table["instrument"].eq(instrument)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {instrument} row in {filename}.")
    return rows.iloc[0]


def selected_parameters(instrument: str) -> dict[str, dict[str, int]]:
    scaled = selected_row(
        "normal_cdf_best_2y_5y_10y_through_20251231_no_cutoff_no_threshold_stats.csv",
        instrument,
    )
    unscaled = selected_row(
        "normal_cdf_unscaled_rounded_0p5_best_2y_5y_10y_through_20251231_stats.csv",
        instrument,
    )
    for row in (scaled, unscaled):
        if row["rebalance_frequency"] != "weekly":
            raise ValueError(f"Selected CDF model for {instrument} is not weekly.")
        if row["signal_cutoff"] != 0 or row["rebalance_threshold_fraction"] != 0:
            raise ValueError(f"Unexpected CDF model filters for {instrument}.")
    if float(unscaled["position_rounding_increment"]) != 0.5:
        raise ValueError(f"Unexpected CDF rounding for {instrument}.")
    e = selected_row(
        "ewma_best_parameters_2y_5y_10y_through_20251231_extended_stats.csv",
        instrument,
    )
    k = selected_row(
        "kama_best_parameters_2y_5y_10y_through_20251231_extended_stats.csv",
        instrument,
    )
    return {
        "Normal CDF scaled": {"lookback": int(scaled["signal_lookback_days"])},
        "Normal CDF unscaled": {"lookback": int(unscaled["signal_lookback_days"])},
        "EWMA": {"n": int(e["n"]), "m": int(e["m"])},
        "KAMA": {
            "fast": int(k["alpha_fast_n"]),
            "slow": int(k["alpha_slow_n"]),
            "er": int(k["er_lookback_days"]),
            "m": int(k["kama_extreme_window_m"]),
        },
    }


def parameter_labels(
    parameters: dict[str, dict[str, int]],
    scaled_position_rounding: float = CDF_POSITION_ROUNDING,
    unscaled_position_rounding: float = CDF_POSITION_ROUNDING,
    target_annualized_volatility: float = cdf.TARGET_ANNUALIZED_VOLATILITY,
) -> dict[str, str]:
    s, u, e, k = (parameters[name] for name in STRATEGIES)
    return {
        STRATEGIES[0]: (
            f"weekly T={s['lookback']}; vol target {target_annualized_volatility * 100:g}%; "
            f"position rounded {scaled_position_rounding:g}"
        ),
        STRATEGIES[1]: (
            f"weekly T={u['lookback']}; unscaled; "
            f"position rounded {unscaled_position_rounding:g}"
        ),
        STRATEGIES[2]: f"N={e['n']}, M={e['m']}",
        STRATEGIES[3]: f"fast={k['fast']}, slow={k['slow']}, ER={k['er']}, M={k['m']}",
    }


def select_rounded_scaled_lookback(prices: pd.Series) -> pd.DataFrame:
    rows = []
    for lookback in CDF_LOOKBACKS:
        frame = cdf.run_backtest(
            prices.loc[START:TRAIN_END],
            lookback_days=lookback,
            rebalance_frequency="weekly",
            position_change_threshold_fraction=CDF_REBALANCE_THRESHOLD,
            signal_cutoff=CDF_SIGNAL_CUTOFF,
            signal_mode="rounded",
            volatility_lookback_days=CDF_VOLATILITY_LOOKBACK,
            output_start_date=START,
            position_rounding_increment=CDF_POSITION_ROUNDING,
        )
        rows.append({"signal_lookback_days": lookback, **statistics(frame)})
    grid = pd.DataFrame(rows)
    if grid["sharpe_ratio"].isna().any():
        raise ValueError("A rounded scaled CDF candidate has an undefined Sharpe ratio.")
    return grid.sort_values("sharpe_ratio", ascending=False, kind="stable").reset_index(drop=True)


def report_scaled_lookback(instrument: str, grid: pd.DataFrame) -> int:
    if instrument in SCALED_CDF_LOOKBACK_OVERRIDES:
        return SCALED_CDF_LOOKBACK_OVERRIDES[instrument]
    return int(grid.iloc[0]["signal_lookback_days"])


def run_period(
    prices: pd.Series, parameters: dict[str, dict[str, int]], start: pd.Timestamp,
    scaled_position_rounding: float = CDF_POSITION_ROUNDING,
    unscaled_position_rounding: float = CDF_POSITION_ROUNDING,
    target_annualized_volatility: float = cdf.TARGET_ANNUALIZED_VOLATILITY,
) -> dict[str, pd.DataFrame]:
    prices = prices.loc[START:]
    output = {}
    for name in STRATEGIES[:2]:
        output[name] = cdf.run_backtest(
            prices,
            lookback_days=parameters[name]["lookback"],
            rebalance_frequency="weekly",
            position_change_threshold_fraction=CDF_REBALANCE_THRESHOLD,
            signal_cutoff=CDF_SIGNAL_CUTOFF,
            signal_mode="rounded",
            volatility_lookback_days=CDF_VOLATILITY_LOOKBACK,
            output_start_date=start,
            position_rounding_increment=(
                scaled_position_rounding if name == STRATEGIES[0]
                else unscaled_position_rounding
            ),
            use_volatility_scaling=name == STRATEGIES[0],
            target_annualized_volatility=target_annualized_volatility,
        )
    e, k = parameters["EWMA"], parameters["KAMA"]
    output["EWMA"] = ewma.run_backtest(
        prices, e["n"], e["m"], start_date=start
    ).loc[start:].copy()
    output["KAMA"] = kama.run_backtest(
        prices, k["fast"], k["slow"], k["er"], k["m"], start_date=start
    ).loc[start:].copy()
    # The reference computes period drawdowns from the period's own wealth.
    for frame in output.values():
        frame["drawdown"] = frame["wealth"] / frame["wealth"].cummax() - 1
    return output


def statistics(frame: pd.DataFrame) -> dict[str, object]:
    returns = frame["strategy_return"]
    position = frame["position"]
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    vol = returns.std(ddof=1) * np.sqrt(252)
    annual_return = returns.mean() * 252
    wins, losses = returns[returns > 0], returns[returns < 0]
    active = position.ne(0)
    longest = 0
    peak_date = frame.index[0]
    running_peak = wealth.iloc[0]
    underwater = False
    for date, value in wealth.items():
        if value >= running_peak:
            if underwater:
                longest = max(longest, (date - peak_date).days)
            running_peak = value
            peak_date = date
            underwater = False
        else:
            underwater = True
            longest = max(longest, (date - peak_date).days)
    max_drawdown = drawdown.min()
    return {
        "start_date": frame.index[0].date(),
        "end_date": frame.index[-1].date(),
        "observations": len(frame),
        "first_active_date": frame.index[active][0].date() if active.any() else None,
        "total_return_pct": (wealth.iloc[-1] - 1) * 100,
        "annualized_return_pct": annual_return * 100,
        "annualized_volatility_pct": vol * 100,
        "sharpe_ratio": annual_return / vol if vol > 0 else np.nan,
        "max_drawdown_pct": max_drawdown * 100,
        "calmar_ratio": annual_return / -max_drawdown if max_drawdown < 0 else np.nan,
        "max_drawdown_duration_calendar_days": longest,
        "win_ratio_pct": len(wins) / (len(wins) + len(losses)) * 100
        if len(wins) + len(losses) else np.nan,
        "risk_reward_ratio": wins.mean() / -losses.mean()
        if len(wins) and len(losses) else np.nan,
        "average_abs_position": position.abs().mean(),
        "maximum_abs_position": position.abs().max(),
        "long_days": int(position.gt(0).sum()),
        "flat_days": int(position.eq(0).sum()),
        "short_days": int(position.lt(0).sum()),
    }


def trading_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize trade days, position turnover, and uncompounded costs."""
    return {
        "trading_volume": int(frame["position_changed"].sum()),
        "position_turnover": float(frame["position_change"].sum()),
        "trading_cost_accumulated": float(frame["trading_cost"].sum()),
    }


def save_plots(
    frames: dict[str, pd.DataFrame], labels: dict[str, str],
    instrument: str, period: str, directory: Path,
) -> None:
    for column, filename, ylabel in (
        ("cumulative_pnl_pct", "cumulative_pnl", "Cumulative return (%)"),
        ("drawdown", "drawdown", "Drawdown from running peak (%)"),
    ):
        fig, ax = plt.subplots(figsize=(16, 8))
        for name, frame in frames.items():
            values = frame[column] * (100 if column == "drawdown" else 1)
            ax.plot(frame.index, values, label=f"{name}: {labels[name]}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=f"{instrument} best strategy comparison: {period}", xlabel="Date", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{period}_{filename}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    for ax, (name, frame) in zip(axes, frames.items()):
        sign = np.sign(frame["position"])
        groups = sign.ne(sign.shift()).cumsum()
        for _, group in frame.groupby(groups):
            position = group["position"].iloc[0]
            if position:
                ax.axvspan(group.index[0], group.index[-1] + pd.Timedelta(days=1),
                           color="green" if position > 0 else "red", alpha=0.2, linewidth=0)
        ax.plot(frame.index, frame["price"], color="#263445", linewidth=1)
        ax.set_title(f"{name}: {labels[name]}", loc="left", fontsize=11, weight="bold")
        ax.set_ylabel("Futures price")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{instrument} price and positions: {period}", fontsize=16, weight="bold")
    fig.legend(handles=[Patch(color="green", alpha=0.2, label="Long"),
                        Patch(color="red", alpha=0.2, label="Short"),
                        Patch(facecolor="white", edgecolor="gray", label="Flat")],
               loc="upper right", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(directory / f"{period}_price_positions.png", dpi=150)
    plt.close(fig)


def generate(instrument: str, prices: pd.Series) -> None:
    prices = prices.rename(instrument)
    parameters = selected_parameters(instrument)
    directory = RESULTS / f"{instrument.split()[0].lower()}_best_four_strategies_rounded_0p5_2016_2025_and_2026"
    directory.mkdir(parents=True, exist_ok=True)
    grid = select_rounded_scaled_lookback(prices)
    lookback = report_scaled_lookback(instrument, grid)
    grid["selected_for_report"] = grid["signal_lookback_days"].eq(lookback)
    grid.to_csv(directory / "rounded_scaled_cdf_training_grid.csv", index=False, float_format="%.8f")
    parameters["Normal CDF scaled"]["lookback"] = lookback
    save_report(instrument, prices, parameters, directory)


def save_report(
    instrument: str,
    prices: pd.Series,
    parameters: dict[str, dict[str, int]],
    directory: Path,
    scaled_position_rounding: float = CDF_POSITION_ROUNDING,
    unscaled_position_rounding: float = CDF_POSITION_ROUNDING,
    target_annualized_volatility: float = cdf.TARGET_ANNUALIZED_VOLATILITY,
) -> None:
    """Write reports; trading volume counts changes in the daily position.

    The first position is measured against flat at the period start. Annual
    counts use the previous trading day's position across year boundaries.
    Position turnover sums absolute changes; accumulated trading cost is the
    sum of daily return deductions, not a compounded dollar expense.
    """
    prices = prices.rename(instrument)
    directory.mkdir(parents=True, exist_ok=True)
    labels = parameter_labels(
        parameters, scaled_position_rounding, unscaled_position_rounding,
        target_annualized_volatility,
    )
    summary = []
    for period, start, end in (
        ("2016_2025", START, TRAIN_END),
        ("2026", TEST_START, TEST_END),
    ):
        frames = run_period(
            prices.loc[:end], parameters, start,
            scaled_position_rounding, unscaled_position_rounding,
            target_annualized_volatility,
        )
        annual = []
        for name, slug in zip(STRATEGIES, SLUGS):
            frame = frames[name]
            position_change = frame["position"].diff()
            position_change.iloc[0] = frame["position"].iloc[0]
            frame["position_changed"] = position_change.ne(0)
            frame.to_csv(directory / f"{period}_{slug}_daily.csv", index_label="date", float_format="%.10f")
            summary.append({"period": period, "instrument": instrument, "strategy": name,
                            "parameters": labels[name], **statistics(frame),
                            **trading_statistics(frame)})
            if period == "2016_2025":
                for year, yearly in frame.groupby(frame.index.year):
                    annual.append({"instrument": instrument, "strategy": name,
                                   "parameters": labels[name], "year": year,
                                   **statistics(yearly),
                                   **trading_statistics(yearly)})
        if annual:
            table = pd.DataFrame(annual)
            table.to_csv(directory / f"{period}_annual_stats.csv", index=False, float_format="%.8f")
            metrics = (*METRICS, "trading_volume", "position_turnover", "trading_cost_accumulated")
            pivot = table.pivot(index="year", columns="strategy", values=list(metrics))
            pivot = pivot.reindex(columns=pd.MultiIndex.from_product([metrics, STRATEGIES]))
            pivot.columns.names = ["metric", "strategy"]
            pivot.to_csv(directory / f"{period}_annual_stats_by_year_and_metric.csv", float_format="%.8f")
            pivot = pivot.swaplevel(axis=1).reindex(
                columns=pd.MultiIndex.from_product([STRATEGIES, metrics])
            )
            pivot.columns.names = ["strategy", "metric"]
            pivot.to_csv(directory / f"{period}_annual_stats_by_year_and_strategy.csv", float_format="%.8f")
        save_plots(frames, labels, instrument, period, directory)
    pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False, float_format="%.8f")
    print(f"{instrument}: {directory}")


def main() -> None:
    prices = cdf.load_single_future_prices(include_dbmf=False)
    for instrument in ("5Y future", "10Y future"):
        generate(instrument, prices[instrument])


if __name__ == "__main__":
    main()
