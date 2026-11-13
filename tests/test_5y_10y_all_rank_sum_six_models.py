import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import analyze_2y_all_rank_sum_six_models as trends
import analyze_2y_four_cdf_modes_cap3 as report
import analyze_5y_10y_all_rank_sum_six_models as comparison
import analyze_5y_10y_rank_sum_six_models as previous
import analyze_best_four_futures_strategies as reference
import search_5y_10y_cdf_rank_sum as search
import search_cdf_custom_vol_daily_weekly as common


class FiveAndTenYearAllRankSumTests(unittest.TestCase):
    def test_selected_models_and_reports(self):
        source = reference.cdf.load_single_future_prices(include_dbmf=False)
        prices = {instrument: source[instrument] for instrument in search.INSTRUMENTS}
        grids = search.search(prices)
        expected = {
            "5Y future": ((20, 20), (5, 20, 15, 3)),
            "10Y future": ((35, 15), (5, 25, 20, 3)),
        }
        for instrument, grid in grids.items():
            with self.subTest(instrument=instrument), tempfile.TemporaryDirectory() as directory:
                ewma = trends.selected_trend_winner("EWMA", instrument)
                kama = trends.selected_trend_winner("KAMA", instrument)
                self.assertEqual((ewma["n"], ewma["m"]), expected[instrument][0])
                self.assertEqual(
                    (kama["alpha_fast_n"], kama["alpha_slow_n"], kama["er_lookback_days"], kama["kama_extreme_window_m"]),
                    expected[instrument][1],
                )
                winners = grid.loc[grid["selected_best"]]
                output = Path(directory)
                with patch.object(report.charts, "save_plots") as plot:
                    report.save_report(
                        prices[instrument], winners, output, instrument=instrument,
                        target=common.CONFIGURATIONS[instrument][0], position_cap=None,
                        ewma_winner=ewma, kama_winner=kama,
                    )
                self.assertEqual(plot.call_count, 2)
                summary = pd.read_csv(output / "summary.csv")
                self.assertEqual(len(summary), 12)
                self.assertEqual(len(list(output.glob("*_daily.csv"))), 12)
                self.assertEqual(len(pd.read_csv(output / "2016_2025_annual_stats.csv")), 60)
                for name, winner in (("EWMA", ewma), ("KAMA", kama)):
                    actual = summary.loc[
                        summary["period"].eq("2016_2025") & summary["strategy"].eq(name)
                    ].iloc[0]
                    for metric in ("sharpe_ratio", "max_drawdown_pct", "calmar_ratio"):
                        self.assertAlmostEqual(actual[metric], winner[metric], places=7)
                for _, winner in winners.iterrows():
                    name = report.charts.model_name(winner)
                    actual = summary.loc[
                        summary["period"].eq("2016_2025") & summary["strategy"].eq(name)
                    ].iloc[0]
                    self.assertAlmostEqual(actual["sharpe_ratio"], winner["sharpe_ratio"], places=7)
                self.assertNotEqual(comparison.OUTPUTS[instrument], previous.OUTPUTS[instrument])


if __name__ == "__main__":
    unittest.main()
