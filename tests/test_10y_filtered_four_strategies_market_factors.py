import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_10y_filtered_four_strategies_market_factors as report
import analyze_10y_four_weekly_t64_rounding_0p5 as previous
import analyze_best_four_futures_strategies as reference
import analyze_selected_cdf_market_factor_comparison as factor_comparison


class FilteredTenYearFactorTests(unittest.TestCase):
    def test_scaled_factor_and_other_strategies(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(reference, "save_plots"):
                report.save_report(output)
            summary = pd.read_csv(output / "summary.csv").set_index("strategy")
            self.assertEqual(len(summary), 4)
            for name, slug in zip(reference.STRATEGIES, reference.SLUGS):
                frame = pd.read_csv(output / f"2026_{slug}_daily.csv")
                if name == reference.STRATEGIES[0]:
                    prior = pd.read_csv(factor_comparison.OUTPUT / "10y_with_market_factors_daily.csv")
                    self.assertEqual(summary.loc[name, "market_factor_mode"], "instrument")
                    self.assertTrue(frame["market_factor_signal_multiplier"].ne(1).any())
                    np.testing.assert_allclose(frame["signal"], frame["raw_signal"] * frame["market_factor_signal_multiplier"],
                                               equal_nan=True, atol=1e-9)
                else:
                    prior = pd.read_csv(previous.OUTPUT / f"2026_{slug}_daily.csv")
                    self.assertEqual(summary.loc[name, "market_factor_mode"], "none")
                for column in ("date", "position", "strategy_return", "gross_strategy_return", "trading_cost"):
                    pd.testing.assert_series_equal(frame[column], prior[column])
                gross, net = frame["gross_strategy_return"], frame["strategy_return"]
                ratio = gross.gt(0).sum() / (gross.gt(0).sum() + gross.lt(0).sum()) * 100
                self.assertAlmostEqual(summary.loc[name, "win_ratio_pct"], ratio, places=7)
                self.assertAlmostEqual(
                    summary.loc[name, "sharpe_ratio"], net.mean() / net.std(ddof=1) * np.sqrt(252), places=6,
                )
                self.assertAlmostEqual(
                    summary.loc[name, "total_return_pct"], ((1 + net).prod() - 1) * 100, delta=0.00001,
                )
                np.testing.assert_allclose(
                    frame["trading_cost"], frame["position_change"] * reference.cdf.TRADING_COST_RATIOS[report.INSTRUMENT],
                    atol=1e-10,
                )
                self.assertAlmostEqual(summary.loc[name, "trading_cost_accumulated"], frame["trading_cost"].sum(), places=7)


if __name__ == "__main__":
    unittest.main()
