"""Compare long futures with trained EWMA, KAMA, and weekly scaled T-stat."""

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_2y_all_rank_sum_six_models as selection
import analyze_2y_four_cdf_modes_cap3 as report
import analyze_best_four_futures_strategies as reference
import analyze_long_future_tstat_correlation as long_comparison


OUTPUT = reference.RESULTS / "ewma_kama_and_long_tstat_correlations"


def compare(
    frames: dict[str, pd.DataFrame], long_daily: pd.DataFrame,
    instrument: str, period: str,
) -> tuple[dict, pd.DataFrame]:
    ewma, kama = frames["EWMA"], frames["KAMA"]
    if (
        ewma.empty or ewma.index.has_duplicates or
        not ewma.index.equals(kama.index) or not ewma.index.equals(long_daily.index)
    ):
        raise ValueError(f"{instrument}, {period}: backtest dates do not match.")
    if not (
        np.allclose(ewma["price"], kama["price"], rtol=0, atol=1e-9)
        and np.allclose(ewma["price"], long_daily["price"], rtol=0, atol=1e-9)
    ):
        raise ValueError(f"{instrument}, {period}: backtest prices do not match.")
    daily = pd.DataFrame({
        "price": ewma["price"],
        "ewma_position": ewma["position"],
        "ewma_net_return": ewma["strategy_return"],
        "kama_position": kama["position"],
        "kama_net_return": kama["strategy_return"],
        "long_net_return": long_daily["long_net_return"],
        "tstat_net_return": long_daily["tstat_net_return"],
    })
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError(f"{instrument}, {period}: daily data contains nonfinite values.")
    trend_corr = daily["ewma_net_return"].corr(daily["kama_net_return"])
    long_corr = daily["long_net_return"].corr(daily["tstat_net_return"])
    long_ewma_corr = daily["long_net_return"].corr(daily["ewma_net_return"])
    long_kama_corr = daily["long_net_return"].corr(daily["kama_net_return"])
    if not np.isfinite([trend_corr, long_corr, long_ewma_corr, long_kama_corr]).all():
        raise ValueError(f"{instrument}, {period}: correlation is undefined.")
    return {
        "instrument": instrument,
        "period": period,
        "start_date": daily.index[0].date(),
        "end_date": daily.index[-1].date(),
        "observations": len(daily),
        "long_tstat_correlation": long_corr,
        "long_ewma_correlation": long_ewma_corr,
        "long_kama_correlation": long_kama_corr,
        "ewma_kama_correlation": trend_corr,
    }, daily


def main(output: Path = OUTPUT) -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for instrument in long_comparison.WINNER_DIRECTORIES:
        ewma = selection.selected_trend_winner("EWMA", instrument)
        kama = selection.selected_trend_winner("KAMA", instrument)
        for period, start, end in long_comparison.PERIODS:
            frames, _ = report.build_single_trend_frames(
                prices[instrument], start, end, instrument, ewma, kama, 1
            )
            filename = f"{instrument.split()[0].lower()}_{period}_daily.csv"
            long_daily = pd.read_csv(
                long_comparison.OUTPUT / filename, parse_dates=["date"]
            ).set_index("date")
            summary, daily = compare(frames, long_daily, instrument, period)
            summaries.append(summary)
            daily.to_csv(output / filename, index_label="date", float_format="%.12f")
    table = pd.DataFrame(summaries)
    table.to_csv(output / "correlations.csv", index=False, float_format="%.10f")
    print(table.to_string(index=False))
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
