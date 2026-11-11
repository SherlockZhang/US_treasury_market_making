import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_2y_best_gross_win_filtered_model as report
import analyze_best_four_futures_strategies as reference


class BestGrossWinFilteredModelTests(unittest.TestCase):
    def test_selected_model_and_saved_report(self):
        winner = report.select_best()
        self.assertEqual(winner["model"], "cdf")
        self.assertTrue(winner["volatility_scaling"])
        self.assertEqual(winner["rebalance_frequency"], "weekly")
        self.assertEqual(winner["signal_lookback_days"], 64)
        self.assertEqual(winner["position_rounding_increment"], 1.5)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            report.save_report(directory)
            summary = pd.read_csv(directory / "summary.csv").set_index("period")
            annual = pd.read_csv(directory / "2016_2025_annual_stats.csv")
            for period in ("2016_2025", "2026"):
                frame = pd.read_csv(directory / f"{period}_daily.csv")
                gross = frame["gross_strategy_return"]
                net = frame["strategy_return"]
                win_ratio = gross.gt(0).sum() / (gross.gt(0).sum() + gross.lt(0).sum()) * 100
                self.assertAlmostEqual(summary.loc[period, "win_ratio_pct"], win_ratio, places=7)
                self.assertAlmostEqual(
                    summary.loc[period, "sharpe_ratio"],
                    net.mean() / net.std(ddof=1) * np.sqrt(252), places=6,
                )
                self.assertAlmostEqual(
                    summary.loc[period, "total_return_pct"],
                    ((1 + net).prod() - 1) * 100, places=5,
                )
                np.testing.assert_allclose(
                    frame["trading_cost"],
                    frame["position_change"] * reference.cdf.TRADING_COST_RATIOS[report.INSTRUMENT],
                    atol=1e-10,
                )
                self.assertAlmostEqual(
                    summary.loc[period, "trading_cost_accumulated"],
                    frame["trading_cost"].sum(), places=7,
                )
            self.assertAlmostEqual(summary.loc["2016_2025", "sharpe_ratio"], winner["sharpe_ratio"], places=8)
            self.assertAlmostEqual(
                annual["trading_cost_accumulated"].sum(),
                summary.loc["2016_2025", "trading_cost_accumulated"], places=8,
            )


if __name__ == "__main__":
    unittest.main()
