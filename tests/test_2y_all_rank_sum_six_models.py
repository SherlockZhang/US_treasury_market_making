import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import analyze_2y_all_rank_sum_six_models as comparison
import analyze_2y_four_cdf_modes_cap3 as report
import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as ranking


class TwoYearAllRankSumComparisonTests(unittest.TestCase):
    def test_selected_parameters_and_report_metrics(self):
        prices = reference.cdf.load_single_future_prices(include_dbmf=False)[report.INSTRUMENT]
        grid = ranking.rank_grid(report.search(prices))
        winners = grid.loc[grid["selected_best"]]
        ewma = comparison.selected_trend_winner("EWMA")
        kama = comparison.selected_trend_winner("KAMA")
        self.assertEqual((ewma["n"], ewma["m"]), (126, 5))
        self.assertEqual(
            (kama["alpha_fast_n"], kama["alpha_slow_n"], kama["er_lookback_days"], kama["kama_extreme_window_m"]),
            (5, 504, 20, 3),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(report.charts, "save_plots") as plot:
                report.save_report(
                    prices, winners, output, ewma_winner=ewma, kama_winner=kama,
                    trend_position_size=3,
                )
            self.assertEqual(plot.call_count, 2)
            summary = pd.read_csv(output / "summary.csv")
            self.assertEqual(len(summary), 12)
            self.assertEqual(len(list(output.glob("*_daily.csv"))), 12)
            for model in ("ewma", "kama"):
                daily = pd.read_csv(output / f"2016_2025_{model}_daily.csv")
                self.assertEqual(set(daily["position"].unique()), {-3, 0, 3})
            for name in ("EWMA", "KAMA"):
                selected = summary.loc[
                    summary["period"].eq("2016_2025") & summary["strategy"].eq(name)
                ].iloc[0]
                self.assertIn("position=+/-3", selected["parameters"])
            self.assertIn("N=126, M=5", summary.loc[summary["strategy"].eq("EWMA"), "parameters"].iloc[0])
            self.assertIn("ER=20", summary.loc[summary["strategy"].eq("KAMA"), "parameters"].iloc[0])


if __name__ == "__main__":
    unittest.main()
