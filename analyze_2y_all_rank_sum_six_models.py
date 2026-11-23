"""Compare 2Y CDF winners with trained or user-specified trend models."""

import argparse
import numpy as np
import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as report
import analyze_best_four_futures_strategies as reference
import averaged_trend_signals as averaged_trends
import search_2y_cdf_rank_sum as cdf_rank
import search_ewma_kama_rank_sum as trend_rank


OUTPUT = (
    reference.RESULTS
    / "2y_six_models_all_rank_sum_trained_trends_unit_position_rounding_0p5_cap_3_target_vol_1p5pct_2016_2025_and_2026"
)
AVERAGED_OUTPUT = (
    reference.RESULTS
    / "2y_six_models_all_rank_sum_averaged_trends_unit_position_rounding_0p5_cap_3_target_vol_1p5pct_2016_2025_and_2026"
)


def parse_args(description: str | None = __doc__) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--trend-model", choices=("trained", "averaged"), default="trained",
        help="Use rank-sum trend winners (default) or the fixed averaged signals.",
    )
    return parser.parse_args()


def selected_trend_winner(model: str, instrument: str = report.INSTRUMENT) -> pd.Series:
    path = trend_rank.OUTPUT_DIR / f"{model.lower()}_best_parameters.csv"
    table = pd.read_csv(path)
    rows = table.loc[table["instrument"].eq(instrument) & table["selected_best"].eq(True)]
    if rows.empty:
        raise ValueError(f"Expected a rank-sum {model} winner for {instrument}.")
    if len(rows) > 1:
        parameter_columns = list(trend_rank.previous.PARAMETERS[model])
        rows = rows.sort_values(
            ["sharpe_ratio", *parameter_columns],
            ascending=[False, *([True] * len(parameter_columns))],
            kind="stable",
        )
    row = rows.iloc[0]
    if (
        row["end_date"] != "2025-12-31"
        or not np.isclose(
            row["trading_cost_ratio"], reference.cdf.TRADING_COST_RATIOS[instrument],
            rtol=0, atol=1e-12,
        )
        or row["rank_sum"] != row["sharpe_rank"] + row["max_drawdown_rank"] + row["calmar_rank"]
    ):
        raise ValueError(f"{model} rank-sum winner has unexpected settings.")
    return row


def main(trend_model: str = "trained") -> None:
    if trend_model not in ("trained", "averaged"):
        raise ValueError(f"Unknown trend model {trend_model!r}.")
    all_prices = reference.cdf.load_single_future_prices(include_dbmf=False)
    prices = all_prices[report.INSTRUMENT]
    saved = pd.read_csv(cdf_rank.OUTPUT / "cdf_best_parameters.csv")
    grid = cdf_rank.rank_grid(report.search(prices))
    winners = grid.loc[grid["selected_best"]].copy()
    keys = [*cdf_rank.GROUPS, "signal_lookback_days", "rank_sum"]
    if not saved[keys].sort_values(keys).reset_index(drop=True).equals(
        winners[keys].sort_values(keys).reset_index(drop=True)
    ):
        raise ValueError("Current CDF winners do not match the saved rank-sum grid.")
    if trend_model == "trained":
        output = OUTPUT
        ewma = selected_trend_winner("EWMA")
        kama = selected_trend_winner("KAMA")
        report.save_report(
            prices, winners, output, ewma_winner=ewma, kama_winner=kama,
            trend_position_size=1,
        )
        parameters = {"EWMA": pd.DataFrame([ewma]), "KAMA": pd.DataFrame([kama])}
    else:
        output = AVERAGED_OUTPUT
        report.save_report(
            prices, winners, output, trend_position_size=1, trend_prices=all_prices,
        )
        parameters = {
            model: averaged_trends.parameter_rows(report.INSTRUMENT, model, 1)
            for model in ("EWMA", "KAMA")
        }
    grid.to_csv(output / "cdf_parameter_grid.csv", index=False, float_format="%.10f")
    for model, table in parameters.items():
        table.to_csv(output / f"{model.lower()}_best_parameters.csv", index=False, float_format="%.10f")


if __name__ == "__main__":
    main(parse_args().trend_model)
