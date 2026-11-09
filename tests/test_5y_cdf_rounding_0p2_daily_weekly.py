import unittest

import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_5y_cdf_rounding_0p2 as previous
import search_5y_cdf_rounding_0p2_daily_weekly as search


class CdfDailyWeeklyGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[previous.INSTRUMENT]
        cls.grid = search.search(cls.prices)

    def test_grid_and_weekly_reference(self):
        self.assertEqual(len(self.grid), 20)
        expected = {
            (scaled, frequency, lookback)
            for scaled in (True, False)
            for frequency in ("daily", "weekly")
            for lookback in reference.CDF_LOOKBACKS
        }
        self.assertEqual(
            set(zip(self.grid["volatility_scaling"], self.grid["rebalance_frequency"],
                    self.grid["signal_lookback_days"])),
            expected,
        )
        for scaled in (True, False):
            part = self.grid.loc[self.grid["volatility_scaling"].eq(scaled)]
            self.assertTrue(part["sharpe_ratio"].is_monotonic_decreasing)
            self.assertEqual(list(part["sharpe_rank"]), list(range(1, 11)))
        old = previous.search(self.prices).drop(columns="sharpe_rank")
        weekly = self.grid.loc[self.grid["rebalance_frequency"].eq("weekly")].drop(columns="sharpe_rank")
        keys = ["volatility_scaling", "signal_lookback_days"]
        pd.testing.assert_frame_equal(
            weekly.sort_values(keys).reset_index(drop=True),
            old.sort_values(keys).reset_index(drop=True),
        )

    def test_daily_results_and_out_of_sample_isolation(self):
        for _, row in self.grid.loc[self.grid["rebalance_frequency"].eq("daily")].iterrows():
            frame = reference.cdf.run_backtest(
                self.prices.loc[reference.START:reference.TRAIN_END],
                lookback_days=int(row["signal_lookback_days"]),
                rebalance_frequency="daily",
                position_change_threshold_fraction=0,
                signal_cutoff=0,
                signal_mode="rounded",
                volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
                output_start_date=reference.START,
                position_rounding_increment=0.2,
                use_volatility_scaling=bool(row["volatility_scaling"]),
            )
            self.assertAlmostEqual(row["sharpe_ratio"], reference.statistics(frame)["sharpe_ratio"])
        changed = self.prices.copy()
        changed.loc[reference.TEST_START:] *= 2
        pd.testing.assert_frame_equal(self.grid, search.search(changed))


if __name__ == "__main__":
    unittest.main()
