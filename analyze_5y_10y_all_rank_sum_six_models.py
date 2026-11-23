"""Compare 5Y and 10Y CDF winners with trained or user-specified trend models."""

import pandas as pd

import analyze_2y_all_rank_sum_six_models as trend_selection
import analyze_2y_four_cdf_modes_cap3 as report
import analyze_best_four_futures_strategies as reference
import averaged_trend_signals as averaged_trends
import search_2y_cdf_rank_sum as ranking
import search_5y_10y_cdf_rank_sum as search
import search_cdf_custom_vol_daily_weekly as common


OUTPUTS = {
    "5Y future": reference.RESULTS / "5y_six_models_all_rank_sum_trained_trends_unit_position_rounding_0p5_target_vol_3pct_2016_2025_and_2026",
    "10Y future": reference.RESULTS / "10y_six_models_all_rank_sum_trained_trends_unit_position_rounding_0p5_target_vol_5pct_2016_2025_and_2026",
}
AVERAGED_OUTPUTS = {
    "5Y future": reference.RESULTS / "5y_six_models_all_rank_sum_averaged_trends_unit_position_rounding_0p5_target_vol_3pct_2016_2025_and_2026",
    "10Y future": reference.RESULTS / "10y_six_models_all_rank_sum_averaged_trends_unit_position_rounding_0p5_target_vol_5pct_2016_2025_and_2026",
}


def main(trend_model: str = "trained") -> None:
    if trend_model not in ("trained", "averaged"):
        raise ValueError(f"Unknown trend model {trend_model!r}.")
    all_prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    prices = {instrument: all_prices[instrument] for instrument in search.INSTRUMENTS}
    grids = search.search(prices)
    keys = [*ranking.GROUPS, "signal_lookback_days", "rank_sum"]
    for instrument, grid in grids.items():
        winners = grid.loc[grid["selected_best"]].copy()
        saved = pd.read_csv(search.OUTPUTS[instrument] / "cdf_best_parameters.csv")
        expected = saved[keys].sort_values(keys).reset_index(drop=True)
        actual = winners[keys].sort_values(keys).reset_index(drop=True)
        if not expected.equals(actual):
            raise ValueError(f"{instrument}: current CDF winners differ from the saved rank-sum grid.")
        if len(winners) != 4:
            raise ValueError(f"{instrument}: expected one CDF winner per mode.")
        if trend_model == "trained":
            output = OUTPUTS[instrument]
            ewma = trend_selection.selected_trend_winner("EWMA", instrument)
            kama = trend_selection.selected_trend_winner("KAMA", instrument)
            report.save_report(
                prices[instrument], winners, output, instrument=instrument,
                target=common.CONFIGURATIONS[instrument][0], position_cap=None,
                ewma_winner=ewma, kama_winner=kama, trend_position_size=1,
            )
            parameters = {"EWMA": pd.DataFrame([ewma]), "KAMA": pd.DataFrame([kama])}
        else:
            output = AVERAGED_OUTPUTS[instrument]
            report.save_report(
                prices[instrument], winners, output, instrument=instrument,
                target=common.CONFIGURATIONS[instrument][0], position_cap=None,
                trend_position_size=1, trend_prices=all_prices,
            )
            parameters = {
                model: averaged_trends.parameter_rows(instrument, model, 1)
                for model in ("EWMA", "KAMA")
            }
        grid.to_csv(output / "cdf_parameter_grid.csv", index=False, float_format="%.10f")
        for model, table in parameters.items():
            table.to_csv(output / f"{model.lower()}_best_parameters.csv", index=False, float_format="%.10f")


if __name__ == "__main__":
    main(trend_selection.parse_args(__doc__).trend_model)
