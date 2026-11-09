import unittest

import numpy as np
import pandas as pd

import analyze_10y_four_strategies_market_factors as analysis
import analyze_best_four_futures_strategies as reference
import analyze_custom_vol_cdf_market_factor_comparison as factor_comparison


class TenYearFourStrategyFactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prices = reference.cdf.load_single_future_prices(include_dbmf=False)[analysis.INSTRUMENT]
        cls.frames = analysis.backtest(prices)

    def test_scaled_factor_result_matches_previous_comparison(self):
        previous = pd.read_csv(
            factor_comparison.OUTPUT_DIR / "10y_with_market_factors_daily.csv",
            parse_dates=["date"],
        ).set_index("date")
        frame = self.frames[reference.STRATEGIES[0]]
        self.assertEqual(list(frame.index), list(previous.index))
        for column in ("signal", "position", "strategy_return", "market_factor_signal_multiplier"):
            np.testing.assert_allclose(frame[column], previous[column], atol=1e-9)
        self.assertTrue(frame["market_factor_signal_multiplier"].ne(1).any())

    def test_other_strategies_match_four_strategy_reference(self):
        for name, slug in zip(reference.STRATEGIES[1:], reference.SLUGS[1:]):
            with self.subTest(strategy=name):
                previous = pd.read_csv(
                    reference.RESULTS / analysis.PRIOR_FOLDER / f"2026_{slug}_daily.csv",
                    parse_dates=["date"],
                ).set_index("date")
                frame = self.frames[name]
                self.assertEqual(list(frame.index), list(previous.index))
                for column in ("position", "strategy_return"):
                    np.testing.assert_allclose(frame[column], previous[column], atol=1e-9)
                self.assertGreaterEqual(frame.index.min(), reference.TEST_START)
                self.assertLessEqual(frame.index.max(), reference.TEST_END)


if __name__ == "__main__":
    unittest.main()
