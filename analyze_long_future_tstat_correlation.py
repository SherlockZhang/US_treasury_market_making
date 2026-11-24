"""Compare a one-contract long future with the weekly scaled T-stat strategy."""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as cdf_report
import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as two_year
import search_5y_10y_cdf_rank_sum as other_years
import search_cdf_custom_vol_daily_weekly as configuration


OUTPUT = reference.RESULTS / "long_one_future_vs_weekly_scaled_tstat"
WINNER_DIRECTORIES = {"2Y future": two_year.OUTPUT, **other_years.OUTPUTS}
PERIODS = (
    ("2016_2025", reference.START, reference.TRAIN_END),
    ("2026", reference.TEST_START, reference.TEST_END),
)


def long_one_backtest(prices: pd.Series, instrument: str) -> pd.DataFrame:
    """Buy one unit at the first close and hold it through the period."""
    if prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must be nonempty, ordered, and have unique dates.")
    if prices.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    results = prices.rename("price").to_frame()
    results["daily_return"] = results["price"].pct_change(fill_method=None).fillna(0.0)
    results["position"] = 1.0
    reference.cdf.apply_trading_cost(
        results, instrument, reference.cdf.TRADING_COST_RATIOS[instrument]
    )
    results["wealth"] = (1 + results["strategy_return"]).cumprod()
    return results


def compare(frame: pd.DataFrame, instrument: str, period: str, lookback: int) -> tuple[dict, pd.DataFrame]:
    long = long_one_backtest(frame["price"], instrument)
    daily = pd.DataFrame({
        "price": frame["price"],
        "long_position": long["position"],
        "long_trading_cost": long["trading_cost"],
        "long_net_return": long["strategy_return"],
        "long_wealth": long["wealth"],
        "tstat_position": frame["position"],
        "tstat_trading_cost": frame["trading_cost"],
        "tstat_net_return": frame["strategy_return"],
    })
    correlation = daily["long_net_return"].corr(daily["tstat_net_return"])
    if not np.isfinite(correlation):
        raise ValueError(f"{instrument}, {period}: daily return correlation is undefined.")
    summary = {
        "instrument": instrument,
        "period": period,
        "start_date": daily.index[0].date(),
        "end_date": daily.index[-1].date(),
        "observations": len(daily),
        "signal_lookback_days": lookback,
        "daily_net_return_correlation": correlation,
    }
    return summary, daily


def main(output: Path = OUTPUT) -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for instrument, directory in WINNER_DIRECTORIES.items():
        winners = pd.read_csv(directory / "cdf_best_parameters.csv")
        selected = winners.loc[
            winners["selected_best"].eq(True)
            & winners["volatility_scaling"].eq(True)
            & winners["rebalance_frequency"].eq("weekly")
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one weekly scaled CDF winner for {instrument}.")
        lookback = int(selected.iloc[0]["signal_lookback_days"])
        target = configuration.CONFIGURATIONS[instrument][0]
        cap = cdf_report.POSITION_CAP if instrument == "2Y future" else None
        for period, start, end in PERIODS:
            frame = cdf_report.backtest_cdf(
                prices[instrument], lookback, "weekly", True, start, end,
                instrument, target, cap,
            )
            summary, daily = compare(frame, instrument, period, lookback)
            summaries.append(summary)
            daily.to_csv(
                output / f"{instrument.split()[0].lower()}_{period}_daily.csv",
                index_label="date", float_format="%.12f",
            )
    table = pd.DataFrame(summaries)
    table.to_csv(output / "correlations.csv", index=False, float_format="%.10f")
    print(table.to_string(index=False))
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
