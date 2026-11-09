import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_5y_cdf_rounding_0p2 as search
import search_5y_scaled_cdf_rounding as scaled_search


class CdfRoundingComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[search.INSTRUMENT]
        cls.grid = search.search(cls.prices)

    def test_grid_and_existing_scaled_search(self):
        self.assertEqual(len(self.grid), 10)
        for scaled in (True, False):
            part = self.grid.loc[self.grid["volatility_scaling"].eq(scaled)]
            self.assertEqual(set(part["signal_lookback_days"]), set(reference.CDF_LOOKBACKS))
            self.assertEqual(list(part["sharpe_rank"]), [1, 2, 3, 4, 5])
            self.assertTrue(part["sharpe_ratio"].is_monotonic_decreasing)
            self.assertTrue(part["position_rounding_increment"].eq(0.2).all())
        expected = scaled_search.search(self.prices, rounding_increments=(0.2,))
        pd.testing.assert_frame_equal(
            self.grid.loc[self.grid["volatility_scaling"]].reset_index(drop=True), expected
        )

    def test_unscaled_positions_and_statistics(self):
        for lookback in reference.CDF_LOOKBACKS:
            frame = reference.cdf.run_backtest(
                self.prices.loc[reference.START:reference.TRAIN_END],
                lookback_days=lookback,
                rebalance_frequency="weekly",
                position_change_threshold_fraction=0,
                signal_cutoff=0,
                signal_mode="rounded",
                volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
                output_start_date=reference.START,
                position_rounding_increment=0.2,
                use_volatility_scaling=False,
            )
            signal = frame["signal"].dropna()
            np.testing.assert_allclose(
                frame.loc[signal.index, "desired_position"],
                (signal / 0.2).round().mul(0.2).round(1),
            )
            row = self.grid.loc[
                ~self.grid["volatility_scaling"] & self.grid["signal_lookback_days"].eq(lookback)
            ].iloc[0]
            self.assertAlmostEqual(row["sharpe_ratio"], reference.statistics(frame)["sharpe_ratio"])
            self.assertTrue(pd.isna(row["target_annualized_volatility_pct"]))

    def test_ignores_out_of_sample_prices(self):
        prices = self.prices.copy()
        prices.loc[reference.TEST_START:] *= 2
        pd.testing.assert_frame_equal(self.grid, search.search(prices))


if __name__ == "__main__":
    unittest.main()
