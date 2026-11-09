import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_10y_cdf_rounding_0p1_daily_weekly as search
import search_10y_scaled_cdf_rounding as prior
import search_5y_scaled_cdf_rounding as scaled_search


class TenYearCdfDailyWeeklyGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[search.INSTRUMENT]
        cls.grid = search.search(cls.prices)

    def test_grid_and_reference(self):
        combinations = {
            (scaled, frequency, lookback)
            for scaled in (True, False)
            for frequency in ("daily", "weekly")
            for lookback in reference.CDF_LOOKBACKS
        }
        self.assertEqual(len(self.grid), 20)
        self.assertEqual(
            set(zip(self.grid["volatility_scaling"], self.grid["rebalance_frequency"],
                    self.grid["signal_lookback_days"])), combinations
        )
        self.assertTrue(self.grid["position_rounding_increment"].eq(0.1).all())
        self.assertTrue(self.grid["instrument"].eq(search.INSTRUMENT).all())
        for scaled in (True, False):
            part = self.grid.loc[self.grid["volatility_scaling"].eq(scaled)]
            self.assertTrue(part["sharpe_ratio"].is_monotonic_decreasing)
            self.assertEqual(list(part["sharpe_rank"]), list(range(1, 11)))
        old = scaled_search.search(
            self.prices, prior.INSTRUMENT, prior.ROUNDING_INCREMENTS
        )
        old = old.loc[old["position_rounding_increment"].eq(0.1)].set_index("signal_lookback_days")
        weekly = self.grid.loc[
            self.grid["volatility_scaling"] & self.grid["rebalance_frequency"].eq("weekly")
        ].set_index("signal_lookback_days")
        np.testing.assert_allclose(
            weekly.sort_index()["sharpe_ratio"], old.sort_index()["sharpe_ratio"]
        )

    def test_backtests_and_out_of_sample_isolation(self):
        for _, row in self.grid.iterrows():
            frequency: reference.cdf.RebalanceFrequency = (
                "daily" if row["rebalance_frequency"] == "daily" else "weekly"
            )
            frame = reference.cdf.run_backtest(
                self.prices.loc[reference.START:reference.TRAIN_END],
                lookback_days=int(row["signal_lookback_days"]),
                rebalance_frequency=frequency,
                position_change_threshold_fraction=0,
                signal_cutoff=0,
                signal_mode="rounded",
                volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
                output_start_date=reference.START,
                position_rounding_increment=0.1,
                use_volatility_scaling=bool(row["volatility_scaling"]),
            )
            self.assertAlmostEqual(row["sharpe_ratio"], reference.statistics(frame)["sharpe_ratio"])
            if not row["volatility_scaling"]:
                signal = frame["signal"].dropna()
                np.testing.assert_allclose(
                    frame.loc[signal.index, "desired_position"],
                    (signal / 0.1).round().mul(0.1).round(1),
                )
        changed = self.prices.copy()
        changed.loc[reference.TEST_START:] *= 2
        pd.testing.assert_frame_equal(self.grid, search.search(changed))


if __name__ == "__main__":
    unittest.main()
