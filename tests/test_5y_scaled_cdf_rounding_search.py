import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_5y_scaled_cdf_rounding as search


class ScaledCdfRoundingSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)["5Y future"]
        cls.grid = search.search(cls.prices)

    def test_grid_and_reference_results(self):
        expected = {
            (lookback, increment)
            for lookback in reference.CDF_LOOKBACKS
            for increment in search.ROUNDING_INCREMENTS
        }
        self.assertEqual(
            set(zip(self.grid["signal_lookback_days"], self.grid["position_rounding_increment"])),
            expected,
        )
        self.assertEqual(len(self.grid), len(expected))
        self.assertTrue(self.grid["sharpe_ratio"].is_monotonic_decreasing)
        previous = reference.select_rounded_scaled_lookback(self.prices).set_index(
            "signal_lookback_days"
        )
        for _, row in self.grid.loc[self.grid["position_rounding_increment"].eq(0.5)].iterrows():
            self.assertAlmostEqual(
                row["sharpe_ratio"], previous.loc[row["signal_lookback_days"], "sharpe_ratio"]
            )

    def test_ignores_out_of_sample_prices(self):
        prices = self.prices.copy()
        prices.loc[reference.TEST_START:] *= 2
        pd.testing.assert_frame_equal(self.grid, search.search(prices))

    def test_new_increments_round_scaled_positions(self):
        for increment in (0.3, 0.4):
            with self.subTest(increment=increment):
                frame = reference.cdf.run_backtest(
                    self.prices.loc[reference.START:reference.TRAIN_END],
                    lookback_days=64,
                    signal_cutoff=0,
                    position_change_threshold_fraction=0,
                    signal_mode="rounded",
                    position_rounding_increment=increment,
                )
                eligible = frame["signal"].notna() & frame["realized_volatility"].gt(0)
                scaled = (
                    frame.loc[eligible, "signal"]
                    * reference.cdf.TARGET_ANNUALIZED_VOLATILITY
                    / frame.loc[eligible, "realized_volatility"]
                )
                expected = (scaled / increment).round().mul(increment).round(1)
                np.testing.assert_allclose(frame.loc[eligible, "desired_position"], expected)


if __name__ == "__main__":
    unittest.main()
