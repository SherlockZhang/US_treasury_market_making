import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference


class TradingVolumeReportTests(unittest.TestCase):
    def test_counts_boundaries_reversals_and_annual_pivots(self):
        dates = pd.to_datetime([
            "2016-12-29", "2016-12-30", "2017-01-03", "2017-01-04", "2017-01-05"
        ])
        position = pd.Series([1.0, 1.0, 1.0, -2.0, 0.0], index=dates)
        frame = pd.DataFrame({
            "position": position,
            "strategy_return": [0.01, 0.0, 0.01, -0.01, 0.0],
        }, index=dates)
        parameters = {
            "Normal CDF scaled": {"lookback": 64},
            "Normal CDF unscaled": {"lookback": 252},
            "EWMA": {"n": 5, "m": 40},
            "KAMA": {"fast": 5, "slow": 25, "er": 25, "m": 3},
        }
        frames = {name: frame.copy() for name in reference.STRATEGIES}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(reference, "run_period", side_effect=lambda *args: {
                name: value.copy() for name, value in frames.items()
            }), patch.object(reference, "save_plots"):
                reference.save_report("2Y future", pd.Series(dtype=float), parameters, directory)
            summary = pd.read_csv(directory / "summary.csv")
            self.assertTrue(summary["trading_volume"].eq(3).all())
            annual = pd.read_csv(directory / "2016_2025_annual_stats.csv")
            self.assertTrue(annual.loc[annual["year"].eq(2016), "trading_volume"].eq(1).all())
            self.assertTrue(annual.loc[annual["year"].eq(2017), "trading_volume"].eq(2).all())
            pivot = pd.read_csv(
                directory / "2016_2025_annual_stats_by_year_and_metric.csv",
                header=[0, 1], index_col=0,
            )
            for name in reference.STRATEGIES:
                self.assertEqual(pivot.loc[2016, ("trading_volume", name)], 1)
                self.assertEqual(pivot.loc[2017, ("trading_volume", name)], 2)
            daily = pd.read_csv(directory / "2016_2025_cdf_scaled_daily.csv")
            np.testing.assert_array_equal(
                daily["position_changed"], [True, False, False, True, True]
            )


if __name__ == "__main__":
    unittest.main()
