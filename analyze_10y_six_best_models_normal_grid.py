"""Backtest each 10Y CDF scaling/frequency winner with EWMA and KAMA."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_10y_cdf_rounding_0p5_daily_weekly as grid_search


INSTRUMENT = "10Y future"
ROUNDING = 0.5
GRID_FILE = grid_search.OUTPUT_FILE
OUTPUT_DIR = (
    reference.RESULTS
    / "10y_six_best_models_by_cdf_mode_rounding_0p5_target_vol_5pct_2016_2025_and_2026"
)
MODEL_SLUGS = {
    "EWMA": "ewma",
    "KAMA": "kama",
}
MODEL_COLORS = {
    "Normal CDF scaled daily": "#69a9e3",
    "Normal CDF scaled weekly": "#124a87",
    "Normal CDF unscaled daily": "#f08080",
    "Normal CDF unscaled weekly": "#a51c30",
    "EWMA": "#79c879",
    "KAMA": "#176b37",
}


def selected_cdf_models() -> pd.DataFrame:
    grid = pd.read_csv(GRID_FILE)
    if not grid["instrument"].eq(INSTRUMENT).all():
        raise ValueError(f"{GRID_FILE} contains unexpected instruments.")
    if not np.isclose(grid["position_rounding_increment"], ROUNDING).all():
        raise ValueError(f"{GRID_FILE} does not use rounding {ROUNDING}.")
    scaled = grid["volatility_scaling"].eq(True)
    if not np.isclose(
        grid.loc[scaled, "target_annualized_volatility_pct"],
        grid_search.TARGET_ANNUALIZED_VOLATILITY * 100,
    ).all():
        raise ValueError(f"{GRID_FILE} does not use the requested volatility target.")
    expected = {
        (scaling, frequency, lookback)
        for scaling in (True, False)
        for frequency in grid_search.FREQUENCIES
        for lookback in reference.CDF_LOOKBACKS
    }
    actual = list(
        zip(
            grid["volatility_scaling"],
            grid["rebalance_frequency"],
            grid["signal_lookback_days"],
        )
    )
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"{GRID_FILE} does not contain the complete CDF grid.")
    if not np.isfinite(grid["sharpe_ratio"]).all():
        raise ValueError(f"{GRID_FILE} contains undefined Sharpe values.")
    selected = grid.sort_values(
        "sharpe_ratio", ascending=False, kind="stable"
    ).drop_duplicates(["volatility_scaling", "rebalance_frequency"]).copy()
    selected.insert(0, "winner_sharpe_rank", np.arange(1, len(selected) + 1))
    return selected.reset_index(drop=True)


def model_name(row: pd.Series) -> str:
    scaling = "scaled" if row["volatility_scaling"] else "unscaled"
    return (
        f"Normal CDF {scaling} {row['rebalance_frequency']} "
        f"T={int(row['signal_lookback_days'])}"
    )


def backtest_cdf(
    prices: pd.Series, row: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    target_pct = row["target_annualized_volatility_pct"]
    return reference.cdf.run_backtest(
        prices.loc[reference.START:end],
        lookback_days=int(row["signal_lookback_days"]),
        rebalance_frequency=row["rebalance_frequency"],
        position_change_threshold_fraction=float(row["rebalance_threshold_fraction"]),
        signal_cutoff=float(row["signal_cutoff"]),
        signal_mode="rounded",
        volatility_lookback_days=int(row["volatility_lookback_days"]),
        output_start_date=start,
        position_rounding_increment=float(row["position_rounding_increment"]),
        use_volatility_scaling=bool(row["volatility_scaling"]),
        target_annualized_volatility=(
            float(target_pct) / 100
            if pd.notna(target_pct)
            else grid_search.TARGET_ANNUALIZED_VOLATILITY
        ),
        trading_cost_ratio=reference.cdf.TRADING_COST_RATIOS[INSTRUMENT],
    )


def all_frames(
    prices: pd.Series,
    cdf_models: pd.DataFrame,
    ewma_parameters: dict[str, int],
    kama_parameters: dict[str, int],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    frames = {
        model_name(row): backtest_cdf(prices, row, start, end)
        for _, row in cdf_models.iterrows()
    }
    active_prices = prices.loc[reference.START:end]
    frames["EWMA"] = reference.ewma.run_backtest(
        active_prices,
        ewma_parameters["n"],
        ewma_parameters["m"],
        start_date=start,
    ).loc[start:].copy()
    frames["KAMA"] = reference.kama.run_backtest(
        active_prices,
        kama_parameters["fast"],
        kama_parameters["slow"],
        kama_parameters["er"],
        kama_parameters["m"],
        start_date=start,
    ).loc[start:].copy()
    for frame in frames.values():
        frame["drawdown"] = frame["wealth"] / frame["wealth"].cummax() - 1
        position_change = frame["position"].diff()
        position_change.iloc[0] = frame["position"].iloc[0]
        frame["position_changed"] = position_change.ne(0)
    return frames


def check_cdf_training_results(
    frames: dict[str, pd.DataFrame], cdf_models: pd.DataFrame
) -> None:
    for _, row in cdf_models.iterrows():
        name = model_name(row)
        actual = reference.statistics(frames[name])["sharpe_ratio"]
        expected = float(row["sharpe_ratio"])
        if not np.isclose(actual, expected, rtol=0, atol=1e-8):
            raise ValueError(
                f"{name} training Sharpe {actual} does not match grid value {expected}."
            )


def model_color(name: str) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    for mode, color in MODEL_COLORS.items():
        if name.startswith(f"{mode} T="):
            return color
    raise ValueError(f"No chart color configured for {name}.")


def plot_price_and_position(ax, frame: pd.DataFrame) -> None:
    position = frame["position"]
    if position.empty or not np.isfinite(position).all():
        raise ValueError("Position chart requires non-empty, finite position values.")
    dates = pd.DatetimeIndex(frame.index)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Position chart requires strictly increasing dates.")
    widths = (dates[1:] - dates[:-1]).total_seconds() / 86400
    widths = np.append(widths, 1.0)
    maximum = float(position.abs().max())
    position_limit = maximum if maximum > 0 else 1.0
    colors = np.where(position.gt(0), "green", "red")
    ax.bar(dates, position, width=widths, align="edge", color=colors, alpha=0.25, linewidth=0)
    ax.set_ylim(-position_limit, position_limit)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Position (units)")
    ax.grid(axis="y", alpha=0.2)
    price_axis = ax.twinx()
    price_axis.plot(dates, frame["price"], color="#263445", linewidth=1)
    price_axis.set_ylabel("Futures price")


def save_plots(
    frames: dict[str, pd.DataFrame],
    labels: dict[str, str],
    period: str,
    directory: Path,
    instrument: str = INSTRUMENT,
) -> None:
    for column, filename, ylabel in (
        ("cumulative_pnl_pct", "cumulative_pnl", "Cumulative return (%)"),
        ("drawdown", "drawdown", "Drawdown from running peak (%)"),
    ):
        fig, ax = plt.subplots(figsize=(16, 8))
        for name, frame in frames.items():
            values = frame[column] * (100 if column == "drawdown" else 1)
            ax.plot(
                frame.index, values, color=model_color(name),
                label=f"{name}: {labels[name]}",
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            title=f"{instrument} six best models: {period}",
            xlabel="Date",
            ylabel=ylabel,
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{period}_{filename}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(18, 19), sharex=True)
    for ax, (name, frame) in zip(axes, frames.items()):
        plot_price_and_position(ax, frame)
        ax.set_title(f"{name}: {labels[name]}", loc="left", fontsize=10, weight="bold")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{instrument} price and positions: {period}", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(directory / f"{period}_price_positions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cdf_models = selected_cdf_models()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cdf_models.to_csv(
        OUTPUT_DIR / "selected_normal_cdf_grid_models.csv",
        index=False,
        float_format="%.8f",
    )
    ewma_row = reference.selected_row(
        "ewma_best_parameters_2y_5y_10y_through_20251231_extended_stats.csv",
        INSTRUMENT,
    )
    kama_row = reference.selected_row(
        "kama_best_parameters_2y_5y_10y_through_20251231_extended_stats.csv",
        INSTRUMENT,
    )
    ewma_parameters = {"n": int(ewma_row["n"]), "m": int(ewma_row["m"])}
    kama_parameters = {
        "fast": int(kama_row["alpha_fast_n"]),
        "slow": int(kama_row["alpha_slow_n"]),
        "er": int(kama_row["er_lookback_days"]),
        "m": int(kama_row["kama_extreme_window_m"]),
    }
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[INSTRUMENT]

    labels = {
        model_name(row): (
            f"{row['rebalance_frequency']}; "
            f"{'scaled' if row['volatility_scaling'] else 'unscaled'}; "
            f"T={int(row['signal_lookback_days'])}; rounding={ROUNDING:g}"
            + (
                f"; target={float(row['target_annualized_volatility_pct']):g}%"
                if pd.notna(row["target_annualized_volatility_pct"])
                else ""
            )
        )
        for _, row in cdf_models.iterrows()
    }
    labels["EWMA"] = f"N={ewma_parameters['n']}, M={ewma_parameters['m']}"
    labels["KAMA"] = (
        f"fast={kama_parameters['fast']}, slow={kama_parameters['slow']}, "
        f"ER={kama_parameters['er']}, M={kama_parameters['m']}"
    )
    slug_by_name = {
        **{
            model_name(row): (
                f"normal_cdf_{'scaled' if row['volatility_scaling'] else 'unscaled'}_"
                f"{row['rebalance_frequency']}_t{int(row['signal_lookback_days'])}"
            )
            for _, row in cdf_models.iterrows()
        },
        **MODEL_SLUGS,
    }

    summary = []
    for period, start, end in (
        ("2016_2025", reference.START, reference.TRAIN_END),
        ("2026", reference.TEST_START, reference.TEST_END),
    ):
        frames = all_frames(
            prices, cdf_models, ewma_parameters, kama_parameters, start, end
        )
        if period == "2016_2025":
            check_cdf_training_results(frames, cdf_models)
        annual = []
        for name, frame in frames.items():
            frame.to_csv(
                OUTPUT_DIR / f"{period}_{slug_by_name[name]}_daily.csv",
                index_label="date",
                float_format="%.10f",
            )
            summary.append(
                {
                    "period": period,
                    "instrument": INSTRUMENT,
                    "strategy": name,
                    "parameters": labels[name],
                    **reference.statistics(frame),
                    **reference.trading_statistics(frame),
                }
            )
            if period == "2016_2025":
                for year, yearly in frame.groupby(frame.index.year):
                    annual.append(
                        {
                            "instrument": INSTRUMENT,
                            "strategy": name,
                            "parameters": labels[name],
                            "year": year,
                            **reference.statistics(yearly),
                            **reference.trading_statistics(yearly),
                        }
                    )
        if annual:
            annual_table = pd.DataFrame(annual)
            annual_table.to_csv(
                OUTPUT_DIR / "2016_2025_annual_stats.csv",
                index=False,
                float_format="%.8f",
            )
            metrics = (
                *reference.METRICS,
                "trading_volume",
                "position_turnover",
                "trading_cost_accumulated",
            )
            strategies = list(frames)
            pivot = annual_table.pivot(
                index="year", columns="strategy", values=list(metrics)
            )
            pivot = pivot.reindex(
                columns=pd.MultiIndex.from_product([metrics, strategies])
            )
            pivot.columns.names = ["metric", "strategy"]
            pivot.to_csv(
                OUTPUT_DIR / "2016_2025_annual_stats_by_year_and_metric.csv",
                float_format="%.8f",
            )
            pivot = pivot.swaplevel(axis=1).reindex(
                columns=pd.MultiIndex.from_product([strategies, metrics])
            )
            pivot.columns.names = ["strategy", "metric"]
            pivot.to_csv(
                OUTPUT_DIR / "2016_2025_annual_stats_by_year_and_strategy.csv",
                float_format="%.8f",
            )
        save_plots(frames, labels, period, OUTPUT_DIR)

    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "summary.csv", index=False, float_format="%.8f"
    )
    print(f"{INSTRUMENT}: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
