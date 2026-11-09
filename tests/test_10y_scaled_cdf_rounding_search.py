import unittest

import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_5y_scaled_cdf_rounding as common
import search_10y_scaled_cdf_rounding as search


class TenYearScaledCdfSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[search.INSTRUMENT]
        cls.grid = common.search(cls.prices, search.INSTRUMENT, search.ROUNDING_INCREMENTS)

    def test_grid_and_metrics(self):
        expected = {
            (lookback, increment)
            for lookback in reference.CDF_LOOKBACKS
            for increment in search.ROUNDING_INCREMENTS
        }
        self.assertEqual(len(self.grid), len(expected))
        self.assertEqual(
            set(zip(self.grid["signal_lookback_days"], self.grid["position_rounding_increment"])),
            expected,
        )
        self.assertTrue(self.grid["sharpe_ratio"].is_monotonic_decreasing)
        self.assertTrue(self.grid["instrument"].eq(search.INSTRUMENT).all())
        for _, row in self.grid.iterrows():
            frame = reference.cdf.run_backtest(
                self.prices.loc[reference.START:reference.TRAIN_END],
                lookback_days=int(row["signal_lookback_days"]),
                rebalance_frequency="weekly",
                position_change_threshold_fraction=0,
                signal_cutoff=0,
                signal_mode="rounded",
                volatility_lookback_days=64,
                output_start_date=reference.START,
                position_rounding_increment=row["position_rounding_increment"],
            )
            self.assertAlmostEqual(row["sharpe_ratio"], reference.statistics(frame)["sharpe_ratio"])
            self.assertLessEqual(frame.index.max(), reference.TRAIN_END)

    def test_ignores_out_of_sample_prices(self):
        prices = self.prices.copy()
        prices.loc[reference.TEST_START:] *= 2
        pd.testing.assert_frame_equal(
            self.grid, common.search(prices, search.INSTRUMENT, search.ROUNDING_INCREMENTS)
        )


if __name__ == "__main__":
    unittest.main()
