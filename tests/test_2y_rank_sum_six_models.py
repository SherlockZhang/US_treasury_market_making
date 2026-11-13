import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as report
import analyze_2y_rank_sum_six_models as comparison
import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as ranking


class RankSumComparisonTests(unittest.TestCase):
    def test_six_models_use_four_rank_sum_winners(self):
        prices = reference.cdf.load_single_future_prices(include_dbmf=False)[report.INSTRUMENT]
        grid = ranking.rank_grid(report.search(prices))
        winners = grid.loc[grid["selected_best"]].copy()
        self.assertEqual(len(winners), 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(report.charts, "save_plots") as plot:
                report.save_report(prices, winners, output)
            self.assertEqual(plot.call_count, 2)
            summary = pd.read_csv(output / "summary.csv")
            saved_winners = pd.read_csv(output / "cdf_best_parameters.csv")
            self.assertEqual(len(summary), 12)
            self.assertEqual(len(list(output.glob("*_daily.csv"))), 12)
            weekly = saved_winners.loc[
                saved_winners["volatility_scaling"]
                & saved_winners["rebalance_frequency"].eq("weekly")
            ]
            self.assertEqual(weekly.iloc[0]["signal_lookback_days"], 64)
            self.assertTrue(summary["strategy"].str.contains("scaled weekly T=64").any())
            self.assertTrue(summary["strategy"].eq("EWMA").any())
            self.assertTrue(summary["strategy"].eq("KAMA").any())
            self.assertEqual(len(pd.read_csv(output / "2016_2025_annual_stats.csv")), 60)

    def test_report_rejects_missing_or_duplicate_mode(self):
        winners = pd.read_csv(ranking.OUTPUT / "cdf_best_parameters.csv")
        winners.loc[0, "rebalance_frequency"] = "weekly"
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            report.save_report(pd.Series(dtype=float), winners, Path(directory))

    def test_output_is_separate_from_previous_report(self):
        self.assertNotEqual(comparison.OUTPUT, report.OUTPUT)


if __name__ == "__main__":
    unittest.main()
