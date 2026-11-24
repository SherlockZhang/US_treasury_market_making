import unittest

import numpy as np
import pandas as pd

import analyze_long_future_tstat_correlation as analysis


class LongFutureCorrelationTests(unittest.TestCase):
    def test_long_enters_at_first_close_and_pays_one_entry_cost(self):
        prices = pd.Series(
            [100.0, 110.0, 99.0],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        frame = analysis.long_one_backtest(prices, "2Y future")
        np.testing.assert_allclose(frame["position"], [1, 1, 1])
        np.testing.assert_allclose(frame["gross_strategy_return"], [0, 0.1, -0.1])
        np.testing.assert_allclose(frame["trading_cost"], [0.00195 / 100, 0, 0])
        np.testing.assert_allclose(frame["strategy_return"], [-0.00195 / 100, 0.1, -0.1])

    def test_compare_uses_paired_daily_net_returns_including_flat_days(self):
        index = pd.date_range("2026-01-01", periods=4)
        frame = pd.DataFrame({
            "price": [100.0, 110.0, 99.0, 108.9],
            "position": [0.0, 1.0, 0.0, -1.0],
            "trading_cost": [0.0, 0.0, 0.01, 0.0],
            "strategy_return": [0.0, 0.1, -0.01, -0.1],
        }, index=index)
        summary, daily = analysis.compare(frame, "2Y future", "test", 64)
        expected = np.corrcoef([-0.00195 / 100, 0.1, -0.1, 0.1], [0, 0.1, -0.01, -0.1])[0, 1]
        self.assertAlmostEqual(summary["daily_net_return_correlation"], expected)
        self.assertEqual(summary["observations"], 4)
        self.assertTrue(daily.index.equals(index))

    def test_undefined_correlation_is_reported(self):
        index = pd.date_range("2026-01-01", periods=3)
        frame = pd.DataFrame({
            "price": [100.0, 101.0, 102.0],
            "position": [0.0, 0.0, 0.0],
            "trading_cost": [0.0, 0.0, 0.0],
            "strategy_return": [0.0, 0.0, 0.0],
        }, index=index)
        with self.assertRaisesRegex(ValueError, "correlation is undefined"):
            analysis.compare(frame, "2Y future", "test", 64)


if __name__ == "__main__":
    unittest.main()
