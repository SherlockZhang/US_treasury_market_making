import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as report
import analyze_5y_10y_rank_sum_six_models as comparison
import analyze_best_four_futures_strategies as reference
import search_5y_10y_cdf_rank_sum as search
import search_cdf_custom_vol_daily_weekly as common


class FiveAndTenYearComparisonTests(unittest.TestCase):
    def test_six_models_per_instrument(self):
        source = reference.cdf.load_single_future_prices(include_dbmf=False)
        prices = {instrument: source[instrument] for instrument in search.INSTRUMENTS}
        grids = search.search(prices)
        for instrument, grid in grids.items():
            with self.subTest(instrument=instrument), tempfile.TemporaryDirectory() as directory:
                winners = grid.loc[grid["selected_best"]]
                output = Path(directory)
                with patch.object(report.charts, "save_plots") as plot:
                    report.save_report(
                        prices[instrument], winners, output, instrument=instrument,
                        target=common.CONFIGURATIONS[instrument][0], position_cap=None,
                    )
                self.assertEqual(plot.call_count, 2)
                summary = pd.read_csv(output / "summary.csv")
                self.assertEqual(len(summary), 12)
                self.assertTrue(summary["instrument"].eq(instrument).all())
                self.assertEqual(len(list(output.glob("*_daily.csv"))), 12)
                self.assertEqual(len(pd.read_csv(output / "2016_2025_annual_stats.csv")), 60)
                self.assertTrue(summary["strategy"].eq("EWMA").any())
                self.assertTrue(summary["strategy"].eq("KAMA").any())
                self.assertNotIn("position_cap", summary.columns)
                for _, winner in winners.iterrows():
                    name = report.charts.model_name(winner)
                    training = summary.loc[
                        summary["period"].eq("2016_2025") & summary["strategy"].eq(name)
                    ]
                    self.assertEqual(len(training), 1)
                    for metric in ("sharpe_ratio", "max_drawdown_pct", "calmar_ratio"):
                        self.assertAlmostEqual(training.iloc[0][metric], winner[metric], places=7)
                if instrument == "10Y future":
                    self.assertTrue(summary["strategy"].eq("Normal CDF unscaled daily T=32").any())
                self.assertNotEqual(comparison.OUTPUTS[instrument], search.OUTPUTS[instrument])


if __name__ == "__main__":
    unittest.main()
