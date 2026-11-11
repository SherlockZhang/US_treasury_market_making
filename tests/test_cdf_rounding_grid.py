import unittest
from itertools import product

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_cdf_custom_vol_daily_weekly as common
import search_cdf_rounding_grid as search


class CdfRoundingGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)
        cls.grid = common.search(
            {name: cls.prices[name] for name in common.CONFIGURATIONS},
            search.ROUNDING_INCREMENTS,
        )

    def test_grid_rankings_and_zero_volatility(self):
        self.assertEqual(len(self.grid), 300)
        combinations = set(product(common.FREQUENCIES, reference.CDF_LOOKBACKS, search.ROUNDING_INCREMENTS))
        for instrument, (target, _) in common.CONFIGURATIONS.items():
            for scaled in (True, False):
                with self.subTest(instrument=instrument, scaled=scaled):
                    grid = self.grid.loc[
                        self.grid["instrument"].eq(instrument)
                        & self.grid["volatility_scaling"].eq(scaled)
                    ]
                    self.assertEqual(len(grid), 50)
                    self.assertEqual(
                        set(zip(grid["rebalance_frequency"], grid["signal_lookback_days"], grid["position_rounding_increment"])),
                        combinations,
                    )
                    valid = grid.loc[grid["sharpe_status"].eq("ok")]
                    undefined = grid.loc[grid["sharpe_status"].eq("undefined_zero_volatility")]
                    self.assertTrue(valid["sharpe_ratio"].is_monotonic_decreasing)
                    self.assertEqual(valid["sharpe_rank"].tolist(), list(range(1, len(valid) + 1)))
                    self.assertEqual(grid["selected_best"].sum(), 1)
                    self.assertTrue(undefined["sharpe_ratio"].isna().all())
                    self.assertTrue(undefined["sharpe_rank"].isna().all())
                    self.assertFalse(undefined["selected_best"].any())
                    self.assertTrue(undefined["annualized_volatility_pct"].eq(0).all())
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

    def test_each_winner_and_coarse_unscaled_candidate(self):
        for instrument, (target, _) in common.CONFIGURATIONS.items():
            prices = self.prices[instrument].loc[reference.START:reference.TRAIN_END]
            for scaled in (True, False):
                grid = self.grid.loc[
                    self.grid["instrument"].eq(instrument)
                    & self.grid["volatility_scaling"].eq(scaled)
                ]
                rows = [grid.loc[grid["selected_best"]].iloc[0]]
                if not scaled:
                    rows.append(grid.loc[
                        grid["position_rounding_increment"].eq(2.5)
                        & grid["rebalance_frequency"].eq("daily")
                    ].iloc[0])
                for row in rows:
                    frame = reference.cdf.run_backtest(
                        prices, lookback_days=int(row["signal_lookback_days"]),
                        rebalance_frequency=row["rebalance_frequency"],
                        position_change_threshold_fraction=0, signal_cutoff=0,
                        signal_mode="rounded", volatility_lookback_days=64,
                        output_start_date=reference.START,
                        position_rounding_increment=row["position_rounding_increment"],
                        use_volatility_scaling=scaled, target_annualized_volatility=target,
                    )
                    actual = reference.statistics(frame)["sharpe_ratio"]
                    if row["sharpe_status"] == "ok":
                        self.assertAlmostEqual(row["sharpe_ratio"], actual)
                    else:
                        self.assertTrue(np.isnan(actual))
                        self.assertTrue(frame["position"].eq(0).all())
                    self.assertLessEqual(frame.index.max(), reference.TRAIN_END)

    def test_training_isolated_from_out_of_sample_prices(self):
        changed = self.prices["2Y future"].copy()
        changed.loc[reference.TEST_START:] *= 2
        expected = self.grid.loc[self.grid["instrument"].eq("2Y future")].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            expected, common.search({"2Y future": changed}, search.ROUNDING_INCREMENTS),
        )


if __name__ == "__main__":
    unittest.main()
