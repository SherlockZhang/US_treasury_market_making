import unittest

import numpy as np
import pandas as pd

import analyze_best_cdf_market_factor_comparison as comparison
import analyze_best_four_futures_strategies as analysis


class BestCdfMarketFactorComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = analysis.cdf.load_single_future_prices(include_dbmf=False)

    def test_selected_training_lookbacks_ignore_2026(self):
        for instrument, expected in (("2Y future", 252), ("5Y future", 252), ("10Y future", 64)):
            with self.subTest(instrument=instrument):
                prices = self.prices[instrument]
                grid = analysis.select_rounded_scaled_lookback(prices)
                self.assertEqual(int(grid.iloc[0]["signal_lookback_days"]), expected)
                altered = prices.copy()
                altered.loc[analysis.TEST_START:] *= 2
                pd.testing.assert_frame_equal(grid, analysis.select_rounded_scaled_lookback(altered))

    def test_10y_64_day_comparison_matches_original(self):
        instrument = "10Y future"
        original_dir = analysis.RESULTS / "cdf_rounded_vol_scaled_2026_market_factor_comparison"
        original_summary = pd.read_csv(original_dir / "summary.csv")
        frames = comparison.backtest_pair(instrument, self.prices[instrument], 64)
        for mode, suffix in (("none", "without"), ("instrument", "with")):
            with self.subTest(mode=mode):
                reference = pd.read_csv(
                    original_dir / f"10y_{suffix}_market_factors_daily.csv",
                    parse_dates=["date"],
                ).set_index("date")
                frame = frames[mode]
                self.assertEqual(list(frame.index), list(reference.index))
                np.testing.assert_allclose(frame["position"], reference["position"], atol=1e-9)
                np.testing.assert_allclose(frame["strategy_return"], reference["strategy_return"], atol=1e-9)
                expected = original_summary.loc[
                    original_summary.instrument.eq(instrument) & original_summary.market_factor_mode.eq(mode)
                ].iloc[0]
                summary = comparison.summarize(instrument, mode, 64, frame)
                for metric in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "position_change_count"):
                    self.assertAlmostEqual(summary[metric], expected[metric], delta=0.000001)

    def test_comparison_uses_four_strategy_report_parameters(self):
        for instrument in ("2Y future", "5Y future", "10Y future"):
            with self.subTest(instrument=instrument):
                prices = self.prices[instrument]
                grid = analysis.select_rounded_scaled_lookback(prices)
                lookback = analysis.report_scaled_lookback(instrument, grid)
                self.assertEqual(lookback, 64)
                frames = comparison.backtest_pair(instrument, prices, lookback)
                stem = instrument.split()[0].lower()
                root = analysis.RESULTS / f"{stem}_best_four_strategies_rounded_0p5_2016_2025_and_2026"
                reference = pd.read_csv(
                    root / "2026_cdf_scaled_daily.csv", parse_dates=["date"]
                ).set_index("date")
                self.assertEqual(list(reference.index), list(frames["none"].index))
                np.testing.assert_allclose(reference["position"], frames["none"]["position"], atol=1e-9)
                np.testing.assert_allclose(reference["strategy_return"], frames["none"]["strategy_return"], atol=1e-9)
                for mode, frame in frames.items():
                    summary = comparison.summarize(instrument, mode, lookback, frame)
                    self.assertEqual(summary["signal_lookback_days"], 64)
                    self.assertEqual(summary["volatility_lookback_days"], 64)
                    self.assertEqual(summary["position_rounding_increment"], 0.5)
                    self.assertEqual(summary["target_annualized_volatility_pct"], 1)
                    self.assertEqual(summary["signal_mode"], "rounded")
                    self.assertEqual(summary["rebalance_frequency"], "weekly")
                    self.assertEqual(
                        summary["lookback_selection"],
                        "user-specified" if instrument in ("2Y future", "5Y future")
                        else "2016-2025 rounded training Sharpe",
                    )


if __name__ == "__main__":
    unittest.main()
