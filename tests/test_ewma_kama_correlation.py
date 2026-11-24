import unittest

import numpy as np
import pandas as pd

import analyze_ewma_kama_correlation as analysis


class TrendCorrelationTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2026-01-01", periods=4)
        self.ewma = pd.DataFrame({
            "price": [100.0, 101.0, 102.0, 103.0],
            "position": [0.0, 1.0, 0.0, -1.0],
            "strategy_return": [0.0, 0.02, -0.01, -0.03],
        }, index=index)
        self.kama = self.ewma.copy()
        self.kama["strategy_return"] = [0.0, -0.01, 0.03, -0.02]
        self.long_daily = pd.DataFrame({
            "price": self.ewma["price"],
            "long_net_return": [0.0, 0.1, -0.1, 0.05],
            "tstat_net_return": [0.0, -0.1, 0.1, -0.05],
        }, index=index)

    def test_paired_daily_returns_include_flat_days_and_costs(self):
        summary, daily = analysis.compare(
            {"EWMA": self.ewma, "KAMA": self.kama}, self.long_daily, "2Y future", "test"
        )
        self.assertAlmostEqual(
            summary["ewma_kama_correlation"],
            np.corrcoef([0, 0.02, -0.01, -0.03], [0, -0.01, 0.03, -0.02])[0, 1],
        )
        self.assertAlmostEqual(summary["long_tstat_correlation"], -1.0)
        self.assertAlmostEqual(
            summary["long_ewma_correlation"],
            np.corrcoef([0, 0.1, -0.1, 0.05], [0, 0.02, -0.01, -0.03])[0, 1],
        )
        self.assertAlmostEqual(
            summary["long_kama_correlation"],
            np.corrcoef([0, 0.1, -0.1, 0.05], [0, -0.01, 0.03, -0.02])[0, 1],
        )
        self.assertEqual(len(daily), 4)

    def test_misaligned_dates_are_rejected(self):
        self.kama.index = self.kama.index + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "dates do not match"):
            analysis.compare(
                {"EWMA": self.ewma, "KAMA": self.kama}, self.long_daily, "2Y future", "test"
            )

    def test_undefined_correlation_is_rejected(self):
        self.kama["strategy_return"] = 0.0
        with self.assertRaisesRegex(ValueError, "correlation is undefined"):
            analysis.compare(
                {"EWMA": self.ewma, "KAMA": self.kama}, self.long_daily, "2Y future", "test"
            )


if __name__ == "__main__":
    unittest.main()
