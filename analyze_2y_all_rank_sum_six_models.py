"""Compare 2Y CDF, EWMA, and KAMA rank-sum grid winners."""

import numpy as np
import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as report
import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as cdf_rank
import search_ewma_kama_rank_sum as trend_rank


OUTPUT = (
    reference.RESULTS
    / "2y_six_models_all_rank_sum_rounding_0p5_cap_3_target_vol_1p5pct_2016_2025_and_2026"
)


def selected_trend_winner(model: str, instrument: str = report.INSTRUMENT) -> pd.Series:
    path = trend_rank.OUTPUT_DIR / f"{model.lower()}_best_parameters.csv"
    table = pd.read_csv(path)
    rows = table.loc[table["instrument"].eq(instrument) & table["selected_best"].eq(True)]
    if len(rows) != 1:
        raise ValueError(f"Expected one rank-sum {model} winner for {instrument}.")
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


def main() -> None:
    prices = reference.cdf.load_single_future_prices(include_dbmf=False)[report.INSTRUMENT]
    saved = pd.read_csv(cdf_rank.OUTPUT / "cdf_best_parameters.csv")
    grid = cdf_rank.rank_grid(report.search(prices))
    winners = grid.loc[grid["selected_best"]].copy()
    keys = [*cdf_rank.GROUPS, "signal_lookback_days", "rank_sum"]
    if not saved[keys].sort_values(keys).reset_index(drop=True).equals(
        winners[keys].sort_values(keys).reset_index(drop=True)
    ):
        raise ValueError("Current CDF winners do not match the saved rank-sum grid.")
    ewma = selected_trend_winner("EWMA")
    kama = selected_trend_winner("KAMA")
    report.save_report(prices, winners, OUTPUT, ewma_winner=ewma, kama_winner=kama)
    grid.to_csv(OUTPUT / "cdf_parameter_grid.csv", index=False, float_format="%.10f")
    pd.DataFrame([ewma]).to_csv(
        OUTPUT / "ewma_best_parameters.csv", index=False, float_format="%.10f"
    )
    pd.DataFrame([kama]).to_csv(
        OUTPUT / "kama_best_parameters.csv", index=False, float_format="%.10f"
    )


if __name__ == "__main__":
    main()
