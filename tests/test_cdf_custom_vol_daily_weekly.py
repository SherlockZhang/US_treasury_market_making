import unittest
from itertools import product

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_cdf_custom_vol_daily_weekly as search


class CdfCustomVolatilityGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)
        cls.grid = search.search({key: cls.prices[key] for key in search.CONFIGURATIONS})

    def test_grid_and_net_metrics(self):
        self.assertEqual(len(self.grid), 60)
        for instrument, (target, rounding) in search.CONFIGURATIONS.items():
            for scaled in (True, False):
                with self.subTest(instrument=instrument, scaled=scaled):
                    grid = self.grid.loc[
                        self.grid["instrument"].eq(instrument)
                        & self.grid["volatility_scaling"].eq(scaled)
                    ]
                    self.assertEqual(
                        set(zip(grid["rebalance_frequency"], grid["signal_lookback_days"])),
                        set(product(search.FREQUENCIES, reference.CDF_LOOKBACKS)),
                    )
                    self.assertEqual(grid["sharpe_rank"].tolist(), list(range(1, 11)))
                    self.assertTrue(grid["sharpe_ratio"].is_monotonic_decreasing)
                    self.assertEqual(grid["selected_best"].sum(), 1)
                    self.assertTrue(grid["position_rounding_increment"].eq(rounding).all())
                    self.assertTrue(grid["signal_cutoff"].eq(0).all())
                    self.assertTrue(grid["rebalance_threshold_fraction"].eq(0).all())
                    if scaled:
                        np.testing.assert_allclose(grid["target_annualized_volatility_pct"], target * 100)
                    else:
                        self.assertTrue(grid["target_annualized_volatility_pct"].isna().all())
                    np.testing.assert_allclose(
                        grid["trading_cost_accumulated"],
                        grid["position_turnover"] * reference.cdf.TRADING_COST_RATIOS[instrument],
                    )
                    for _, row in grid.iterrows():
                        prices = self.prices[instrument].loc[reference.START:reference.TRAIN_END]
                        frame = reference.cdf.run_backtest(
                            prices, lookback_days=int(row["signal_lookback_days"]),
                            rebalance_frequency=row["rebalance_frequency"],
                            position_change_threshold_fraction=0, signal_cutoff=0,
                            signal_mode="rounded", volatility_lookback_days=64,
                            output_start_date=reference.START,
                            position_rounding_increment=rounding,
                            use_volatility_scaling=scaled, target_annualized_volatility=target,
                        )
                        self.assertAlmostEqual(row["sharpe_ratio"], reference.statistics(frame)["sharpe_ratio"])
                        self.assertLessEqual(frame.index.max(), reference.TRAIN_END)
                        eligible = frame["signal"].notna()
                        if scaled:
                            eligible &= frame["realized_volatility"].gt(0)
                            desired = frame.loc[eligible, "signal"] * target / frame.loc[eligible, "realized_volatility"]
                        else:
                            desired = frame.loc[eligible, "signal"]
                        expected = (desired / rounding).round().mul(rounding).round(1)
                        np.testing.assert_allclose(frame.loc[eligible, "desired_position"], expected)

    def test_out_of_sample_prices_do_not_affect_search(self):
        altered = self.prices["2Y future"].copy()
        altered.loc[reference.TEST_START:] *= 2
        expected = self.grid.loc[self.grid["instrument"].eq("2Y future")].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, search.search({"2Y future": altered}))


if __name__ == "__main__":
    unittest.main()
