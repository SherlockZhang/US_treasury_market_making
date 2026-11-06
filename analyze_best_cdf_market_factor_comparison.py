"""Compare 2026 rounded scaled CDF models with and without market factors."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import analyze_best_four_futures_strategies as analysis
import backtest_normal_cdf_signal as cdf


OUTPUT_DIR = analysis.RESULTS / "cdf_four_strategy_model_2026_market_factor_comparison"
INSTRUMENTS = ("2Y future", "5Y future", "10Y future")


def backtest_pair(
    instrument: str, prices: pd.Series, lookback: int
) -> dict[str, pd.DataFrame]:
    prices = prices.loc[analysis.START:analysis.TEST_END]
    specs = cdf.INSTRUMENT_MARKET_FACTOR_PROFILES[instrument]
    factors = tuple(
        (spec, cdf.calculate_instrument_market_factor(spec, prices.index))
        for spec in specs
    )
    results = {}
    for mode, factor_data in (("none", ()), ("instrument", factors)):
        results[mode] = cdf.run_backtest(
            prices,
            lookback_days=lookback,
            rebalance_frequency="weekly",
            position_change_threshold_fraction=analysis.CDF_REBALANCE_THRESHOLD,
            signal_cutoff=analysis.CDF_SIGNAL_CUTOFF,
            signal_mode="rounded",
            volatility_lookback_days=analysis.CDF_VOLATILITY_LOOKBACK,
            output_start_date=analysis.TEST_START,
            position_rounding_increment=analysis.CDF_POSITION_ROUNDING,
            instrument_factor_data=factor_data,
        )
    return results


def profile_description(instrument: str) -> str:
    return "; ".join(
        f"{spec.ticker}:p={spec.probability:g}:"
        f"same_sign_{'increase' if spec.same_sign_increases else 'decrease'}:"
        f"{spec.measure}"
        for spec in cdf.INSTRUMENT_MARKET_FACTOR_PROFILES[instrument]
    )


def summarize(
    instrument: str, mode: str, lookback: int, frame: pd.DataFrame
) -> dict[str, object]:
    stats = analysis.statistics(frame)
    stats["position_change_count"] = int(frame["position"].diff().fillna(0).ne(0).sum())
    return {
        "instrument": instrument,
        "market_factor_mode": mode,
        "market_factor_profile": profile_description(instrument) if mode == "instrument" else "",
        "strategy": "Normal CDF scaled",
        "signal_mode": "rounded",
        "volatility_scaling": True,
        "lookback_selection": (
            "user-specified" if instrument in analysis.SCALED_CDF_LOOKBACK_OVERRIDES
            else "2016-2025 rounded training Sharpe"
        ),
        "start_date": stats.pop("start_date"),
        "end_date": stats.pop("end_date"),
        "observations": stats.pop("observations"),
        "first_active_date": stats.pop("first_active_date"),
        "rebalance_frequency": "weekly",
        "signal_lookback_days": lookback,
        "volatility_lookback_days": analysis.CDF_VOLATILITY_LOOKBACK,
        "target_annualized_volatility_pct": cdf.TARGET_ANNUALIZED_VOLATILITY * 100,
        "position_rounding_increment": analysis.CDF_POSITION_ROUNDING,
        "signal_cutoff": analysis.CDF_SIGNAL_CUTOFF,
        "rebalance_threshold_fraction": analysis.CDF_REBALANCE_THRESHOLD,
        **stats,
    }


def save_plots(instrument: str, frames: dict[str, pd.DataFrame], lookback: int) -> None:
    stem = instrument.split()[0].lower()
    for column, scale, label, slug in (
        ("cumulative_pnl_pct", 1, "Cumulative return (%)", "cumulative_pnl"),
        ("drawdown", 100, "Drawdown (%)", "drawdown"),
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        for mode, name in (("none", "Without market factors"), ("instrument", "With market factors")):
            frame = frames[mode]
            ax.plot(frame.index, frame[column] * scale, label=name)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=f"{instrument}: market factor comparison (T={lookback}, 2026)",
               xlabel="Date", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{stem}_{slug}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    prices_by_instrument = cdf.load_single_future_prices(include_dbmf=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    grids = []
    for instrument in INSTRUMENTS:
        prices = prices_by_instrument[instrument]
        grid = analysis.select_rounded_scaled_lookback(prices)
        lookback = analysis.report_scaled_lookback(instrument, grid)
        grid["selected_for_report"] = grid["signal_lookback_days"].eq(lookback)
        grid.insert(0, "instrument", instrument)
        grids.append(grid)
        frames = backtest_pair(instrument, prices, lookback)
        stem = instrument.split()[0].lower()
        for mode, suffix in (("none", "without"), ("instrument", "with")):
            frame = frames[mode]
            frame.to_csv(
                OUTPUT_DIR / f"{stem}_{suffix}_market_factors_daily.csv",
                index_label="date", float_format="%.10f",
            )
            summaries.append(summarize(instrument, mode, lookback, frame))
        save_plots(instrument, frames, lookback)
    pd.concat(grids, ignore_index=True).to_csv(
        OUTPUT_DIR / "rounded_scaled_cdf_training_grid.csv", index=False, float_format="%.8f"
    )
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "summary.csv", index=False, float_format="%.8f"
    )
    print(f"Market-factor comparison: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
