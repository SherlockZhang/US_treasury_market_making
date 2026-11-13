"""Search four capped 2Y CDF modes and compare their winners with EWMA and KAMA."""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_2y_cost_adjusted_best_strategies as cost_grid
import analyze_10y_six_best_models_normal_grid as charts
import analyze_best_four_futures_strategies as reference


INSTRUMENT = "2Y future"
ROUNDING = 0.5
POSITION_CAP = 3.0
TARGET = 0.015
FREQUENCIES: tuple[reference.cdf.RebalanceFrequency, ...] = ("daily", "weekly")
OUTPUT = (
    reference.RESULTS
    / "2y_six_best_models_by_cdf_mode_rounding_0p5_cap_3_target_vol_1p5pct_2016_2025_and_2026"
)


def backtest_cdf(
    prices: pd.Series,
    lookback: int,
    frequency: reference.cdf.RebalanceFrequency,
    scaled: bool,
    start: pd.Timestamp,
    end: pd.Timestamp,
    instrument: str = INSTRUMENT,
    target: float = TARGET,
    position_cap: float | None = POSITION_CAP,
) -> pd.DataFrame:
    return reference.cdf.run_backtest(
        prices.loc[reference.START:end],
        lookback_days=lookback,
        rebalance_frequency=frequency,
        position_change_threshold_fraction=0.0,
        signal_cutoff=0.0,
        signal_mode="rounded",
        volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
        output_start_date=start,
        position_rounding_increment=ROUNDING,
        use_volatility_scaling=scaled,
        target_annualized_volatility=target,
        trading_cost_ratio=reference.cdf.TRADING_COST_RATIOS[instrument],
        position_cap=position_cap,
    )


def search(prices: pd.Series) -> pd.DataFrame:
    rows = []
    for scaled in (True, False):
        for frequency in FREQUENCIES:
            for lookback in reference.CDF_LOOKBACKS:
                frame = backtest_cdf(
                    prices, lookback, frequency, scaled, reference.START, reference.TRAIN_END
                )
                stats = reference.statistics(frame)
                if not np.isfinite(stats["sharpe_ratio"]):
                    raise ValueError(
                        f"Undefined Sharpe for scaled={scaled}, {frequency}, T={lookback}."
                    )
                rows.append({
                    "instrument": INSTRUMENT,
                    "volatility_scaling": scaled,
                    "rebalance_frequency": frequency,
                    "signal_lookback_days": lookback,
                    "volatility_lookback_days": reference.CDF_VOLATILITY_LOOKBACK,
                    "target_annualized_volatility_pct": TARGET * 100 if scaled else np.nan,
                    "position_rounding_increment": ROUNDING,
                    "position_cap": POSITION_CAP,
                    "signal_cutoff": 0.0,
                    "rebalance_threshold_fraction": 0.0,
                    "trading_cost_ratio": reference.cdf.TRADING_COST_RATIOS[INSTRUMENT],
                    "days_position_capped": int(frame["uncapped_position"].abs().gt(POSITION_CAP).sum()),
                    "position_turnover": float(frame["position_change"].sum()),
                    "trading_cost_accumulated": float(frame["trading_cost"].sum()),
                    **stats,
                })
    grid = pd.DataFrame(rows).sort_values(
        ["volatility_scaling", "rebalance_frequency", "sharpe_ratio"],
        ascending=[False, True, False], kind="stable",
    ).reset_index(drop=True)
    grid.insert(0, "sharpe_rank", grid.groupby(
        ["volatility_scaling", "rebalance_frequency"], sort=False
    ).cumcount() + 1)
    return grid


def build_frames(
    prices: pd.Series, winners: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    instrument: str = INSTRUMENT, target: float = TARGET,
    position_cap: float | None = POSITION_CAP,
    ewma_winner: pd.Series | None = None,
    kama_winner: pd.Series | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames = {}
    labels = {}
    for _, row in winners.iterrows():
        name = charts.model_name(row)
        scaled = bool(row["volatility_scaling"])
        frame = backtest_cdf(
            prices, int(row["signal_lookback_days"]), row["rebalance_frequency"],
            scaled, start, end, instrument, target, position_cap,
        )
        if start == reference.START:
            actual = reference.statistics(frame)
            for metric in ("sharpe_ratio", "max_drawdown_pct", "calmar_ratio"):
                if not np.isclose(actual[metric], row[metric], rtol=0, atol=1e-8):
                    raise ValueError(f"{name} training {metric} does not match the grid.")
        frames[name] = frame
        labels[name] = (
            f"{row['rebalance_frequency']}; {'scaled' if scaled else 'unscaled'}; "
            f"T={int(row['signal_lookback_days'])}; rounding=0.5"
            + (f"; cap=+/-{position_cap:g}" if position_cap is not None else "")
            + (f"; target={target * 100:g}%" if scaled else "")
        )

    e = ewma_winner if ewma_winner is not None else cost_grid.selected_row(
        cost_grid.EWMA_BEST, instrument=instrument
    )
    k = kama_winner if kama_winner is not None else cost_grid.selected_row(
        cost_grid.KAMA_BEST, instrument=instrument
    )
    active_prices = prices.loc[reference.START:end]
    frames["EWMA"] = reference.ewma.run_backtest(
        active_prices, int(e["n"]), int(e["m"]), start_date=start
    ).loc[start:].copy()
    frames["KAMA"] = reference.kama.run_backtest(
        active_prices, int(k["alpha_fast_n"]), int(k["alpha_slow_n"]),
        int(k["er_lookback_days"]), int(k["kama_extreme_window_m"]), start_date=start,
    ).loc[start:].copy()
    labels["EWMA"] = f"N={int(e['n'])}, M={int(e['m'])}"
    labels["KAMA"] = (
        f"fast={int(k['alpha_fast_n'])}, slow={int(k['alpha_slow_n'])}, "
        f"ER={int(k['er_lookback_days'])}, M={int(k['kama_extreme_window_m'])}"
    )
    if start == reference.START:
        for name, row in (("EWMA", e), ("KAMA", k)):
            actual = reference.statistics(frames[name])
            for metric in ("sharpe_ratio", "max_drawdown_pct", "calmar_ratio"):
                if not np.isclose(actual[metric], row[metric], rtol=0, atol=1e-8):
                    raise ValueError(f"{name} training {metric} does not match its net-cost grid.")
    for frame in frames.values():
        frame["drawdown"] = frame["wealth"] / frame["wealth"].cummax() - 1
        position_change = frame["position"].diff()
        position_change.iloc[0] = frame["position"].iloc[0]
        frame["position_changed"] = position_change.ne(0)
    return frames, labels


def save_report(
    prices: pd.Series, winners: pd.DataFrame, output: Path,
    instrument: str = INSTRUMENT, target: float = TARGET,
    position_cap: float | None = POSITION_CAP,
    ewma_winner: pd.Series | None = None,
    kama_winner: pd.Series | None = None,
) -> None:
    modes = set(zip(winners["volatility_scaling"], winners["rebalance_frequency"]))
    expected_modes = {(scaled, frequency) for scaled in (True, False) for frequency in FREQUENCIES}
    if len(winners) != 4 or modes != expected_modes:
        raise ValueError("Expected one winner for each of the four CDF modes.")
    output.mkdir(parents=True, exist_ok=True)
    winners.to_csv(output / "cdf_best_parameters.csv", index=False, float_format="%.10f")
    summary = []
    for period, start, end in (
        ("2016_2025", reference.START, reference.TRAIN_END),
        ("2026", reference.TEST_START, reference.TEST_END),
    ):
        frames, labels = build_frames(
            prices, winners, start, end, instrument, target, position_cap,
            ewma_winner, kama_winner,
        )
        annual = []
        for name, frame in frames.items():
            if name in charts.MODEL_SLUGS:
                slug = charts.MODEL_SLUGS[name]
            else:
                slug = name.lower().replace(" ", "_").replace("=", "")
            frame.to_csv(
                output / f"{period}_{slug}_daily.csv",
                index_label="date", float_format="%.10f",
            )
            extra = {}
            if position_cap is not None and "uncapped_position" in frame:
                extra = {
                    "position_cap": position_cap,
                    "days_position_capped": int(frame["uncapped_position"].abs().gt(position_cap).sum()),
                }
            summary.append({
                "period": period, "instrument": instrument, "strategy": name,
                "parameters": labels[name], **extra,
                **reference.statistics(frame), **reference.trading_statistics(frame),
            })
            if period == "2016_2025":
                for year, yearly in frame.groupby(frame.index.year):
                    annual.append({
                        "instrument": instrument, "strategy": name, "parameters": labels[name],
                        "year": year, **reference.statistics(yearly),
                        **reference.trading_statistics(yearly),
                    })
        if annual:
            annual_table = pd.DataFrame(annual)
            annual_table.to_csv(output / "2016_2025_annual_stats.csv", index=False, float_format="%.8f")
            metrics = (*reference.METRICS, "trading_volume", "position_turnover", "trading_cost_accumulated")
            strategies = list(frames)
            pivot = annual_table.pivot(index="year", columns="strategy", values=list(metrics))
            pivot = pivot.reindex(columns=pd.MultiIndex.from_product([metrics, strategies]))
            pivot.columns.names = ["metric", "strategy"]
            pivot.to_csv(output / "2016_2025_annual_stats_by_year_and_metric.csv", float_format="%.8f")
            pivot = pivot.swaplevel(axis=1).reindex(
                columns=pd.MultiIndex.from_product([strategies, metrics])
            )
            pivot.columns.names = ["strategy", "metric"]
            pivot.to_csv(output / "2016_2025_annual_stats_by_year_and_strategy.csv", float_format="%.8f")
        charts.save_plots(frames, labels, period, output, instrument=instrument)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, float_format="%.8f")
    print(f"{instrument}: {output}")


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]
    grid = search(prices)
    winners = grid.loc[grid["sharpe_rank"].eq(1)].copy()
    save_report(prices, winners, OUTPUT)
    grid.to_csv(OUTPUT / "cdf_parameter_grid.csv", index=False, float_format="%.10f")


if __name__ == "__main__":
    main()
