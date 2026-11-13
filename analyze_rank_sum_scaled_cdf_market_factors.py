"""Compare the latest rank-sum scaled CDF winners with and without market factors."""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_2y_all_rank_sum_six_models as two
import analyze_2y_four_cdf_modes_cap3 as report
import analyze_5y_10y_all_rank_sum_six_models as longer
import analyze_best_cdf_market_factor_comparison as comparison
import analyze_best_four_futures_strategies as reference


OUTPUT = reference.RESULTS / "scaled_cdf_rank_sum_daily_weekly_2026_market_factor_comparison"
REPORTS = {"2Y future": two.OUTPUT, **longer.OUTPUTS}
SETTINGS = {
    "2Y future": (0.015, 3.0),
    "5Y future": (0.03, None),
    "10Y future": (0.05, None),
}
FREQUENCIES = ("daily", "weekly")


def selected_row(instrument: str, frequency: str) -> pd.Series:
    table = pd.read_csv(REPORTS[instrument] / "cdf_best_parameters.csv")
    rows = table.loc[
        table["volatility_scaling"].eq(True)
        & table["rebalance_frequency"].eq(frequency)
        & table["selected_best"].eq(True)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one scaled {frequency} winner for {instrument}.")
    row = rows.iloc[0]
    target, cap = SETTINGS[instrument]
    valid_cap = (
        "position_cap" not in row.index or pd.isna(row["position_cap"])
        if cap is None
        else "position_cap" in row.index and row["position_cap"] == cap
    )
    if (
        row["instrument"] != instrument
        or row["end_date"] != "2025-12-31"
        or row["position_rounding_increment"] != report.ROUNDING
        or row["volatility_lookback_days"] != reference.CDF_VOLATILITY_LOOKBACK
        or row["signal_cutoff"] != 0
        or row["rebalance_threshold_fraction"] != 0
        or not np.isclose(row["target_annualized_volatility_pct"], target * 100)
        or not np.isclose(
            row["trading_cost_ratio"], reference.cdf.TRADING_COST_RATIOS[instrument],
            rtol=0, atol=1e-12,
        )
        or not valid_cap
        or row["rank_sum"] != row["sharpe_rank"] + row["max_drawdown_rank"] + row["calmar_rank"]
    ):
        raise ValueError(f"Unexpected settings for {instrument} scaled {frequency} winner.")
    return row


def backtest_pair(
    instrument: str, frequency: reference.cdf.RebalanceFrequency, prices: pd.Series,
    row: pd.Series,
    factor_data: tuple[tuple[reference.cdf.MarketFactorSpec, pd.DataFrame], ...],
) -> dict[str, pd.DataFrame]:
    target, cap = SETTINGS[instrument]
    prices = prices.loc[reference.START:reference.TEST_END]
    frames = {}
    for mode, factors in (("none", ()), ("instrument", factor_data)):
        frames[mode] = reference.cdf.run_backtest(
            prices,
            lookback_days=int(row["signal_lookback_days"]),
            rebalance_frequency=frequency,
            position_change_threshold_fraction=0.0,
            signal_cutoff=0.0,
            signal_mode="rounded",
            volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
            output_start_date=reference.TEST_START,
            position_rounding_increment=report.ROUNDING,
            instrument_factor_data=factors,
            use_volatility_scaling=True,
            target_annualized_volatility=target,
            trading_cost_ratio=reference.cdf.TRADING_COST_RATIOS[instrument],
            position_cap=cap,
        )
    return frames


def verify_baseline(instrument: str, frequency: str, lookback: int, frame: pd.DataFrame) -> None:
    path = REPORTS[instrument] / f"2026_normal_cdf_scaled_{frequency}_t{lookback}_daily.csv"
    saved = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if not frame.index.equals(saved.index):
        raise ValueError(f"{instrument} {frequency} baseline dates differ from {path}.")
    for column in ("position", "strategy_return", "gross_strategy_return", "trading_cost"):
        if not np.allclose(frame[column], saved[column], rtol=0, atol=1e-9):
            raise ValueError(f"{instrument} {frequency} baseline {column} differs from {path}.")


def save_report(directory: Path = OUTPUT) -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    coverage = []
    winners = []
    for instrument in comparison.INSTRUMENTS:
        instrument_prices = prices[instrument].loc[reference.START:reference.TEST_END]
        factor_data = tuple(
            (spec, reference.cdf.calculate_instrument_market_factor(spec, instrument_prices.index))
            for spec in reference.cdf.INSTRUMENT_MARKET_FACTOR_PROFILES[instrument]
        )
        end = instrument_prices.index[-1]
        for spec, data in factor_data:
            series = reference.cdf.load_market_data_series(spec.ticker).loc[:end]
            if series.empty:
                raise ValueError(f"No {spec.ticker} data available by {end.date()}.")
            latest = series.index[-1]
            period = data.loc[reference.TEST_START:end]
            coverage.append({
                "instrument": instrument,
                "ticker": spec.ticker,
                "first_observation": series.index[0].date(),
                "last_observation": latest.date(),
                "backtest_end": end.date(),
                "calendar_days_since_last_observation": (end - latest).days,
                "2026_observations_with_factor_value": int(period["value"].notna().sum()),
                "2026_observations_outside_one_std": int(period["outside_one_std"].fillna(False).sum()),
            })
        for frequency in FREQUENCIES:
            row = selected_row(instrument, frequency)
            winners.append(row)
            frames = backtest_pair(instrument, frequency, instrument_prices, row, factor_data)
            verify_baseline(instrument, frequency, int(row["signal_lookback_days"]), frames["none"])
            output = directory / frequency
            output.mkdir(parents=True, exist_ok=True)
            stem = instrument.split()[0].lower()
            target, cap = SETTINGS[instrument]
            for mode, suffix in (("none", "without"), ("instrument", "with")):
                frame = frames[mode]
                frame["position_changed"] = frame["position_change"].ne(0)
                frame.to_csv(
                    output / f"{stem}_{suffix}_market_factors_daily.csv",
                    index_label="date", float_format="%.10f",
                )
                summary = comparison.summarize(
                    instrument, mode, int(row["signal_lookback_days"]), frame,
                    target, report.ROUNDING, "2016-2025 rank-sum winner",
                )
                summary.update(reference.trading_statistics(frame))
                summary["rebalance_frequency"] = frequency
                summary["position_cap"] = cap
                summary["trading_cost_ratio"] = reference.cdf.TRADING_COST_RATIOS[instrument]
                summaries.append(summary)
            comparison.save_plots(
                instrument, frames, int(row["signal_lookback_days"]), output, frequency,
            )
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False, float_format="%.10f")
    pd.DataFrame(coverage).to_csv(directory / "market_factor_data_coverage.csv", index=False)
    pd.DataFrame(winners).to_csv(directory / "cdf_best_parameters.csv", index=False, float_format="%.10f")
    print(f"Market-factor comparison: {directory}")


if __name__ == "__main__":
    save_report()
