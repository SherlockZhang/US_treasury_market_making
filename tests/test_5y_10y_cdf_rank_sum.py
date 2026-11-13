import unittest
from itertools import product

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as ranking
import search_5y_10y_cdf_rank_sum as search
import search_cdf_custom_vol_daily_weekly as common


class FiveAndTenYearRankSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prices = reference.cdf.load_single_future_prices(include_dbmf=False)
        cls.prices = {instrument: prices[instrument] for instrument in search.INSTRUMENTS}
        cls.grids = search.search(cls.prices)

    def test_grid_settings_and_mode_ranks(self):
        for instrument, grid in self.grids.items():
            self.assertEqual(len(grid), 20)
            self.assertTrue(grid["instrument"].eq(instrument).all())
            self.assertTrue(grid["position_rounding_increment"].eq(0.5).all())
            self.assertTrue(grid["signal_cutoff"].eq(0).all())
            self.assertTrue(grid["rebalance_threshold_fraction"].eq(0).all())
            target = common.CONFIGURATIONS[instrument][0] * 100
            for scaled, frequency in product((True, False), common.FREQUENCIES):
                mode = grid.loc[
                    grid["volatility_scaling"].eq(scaled)
                    & grid["rebalance_frequency"].eq(frequency)
                ]
                self.assertEqual(set(mode["signal_lookback_days"]), set(reference.CDF_LOOKBACKS))
                if scaled:
                    np.testing.assert_allclose(mode["target_annualized_volatility_pct"], target)
                else:
                    self.assertTrue(mode["target_annualized_volatility_pct"].isna().all())
                for metric, rank_name in ranking.METRICS.items():
                    expected = mode[metric].rank(ascending=False, method="min").astype(int)
                    pd.testing.assert_series_equal(mode[rank_name], expected, check_names=False)
                expected = mode[list(ranking.METRICS.values())].sum(axis=1)
                pd.testing.assert_series_equal(mode["rank_sum"], expected, check_names=False)
                self.assertTrue(mode.loc[mode["selected_best"], "rank_sum"].eq(mode["rank_sum"].min()).all())
                self.assertGreaterEqual(mode["selected_best"].sum(), 1)

    def test_winners_reproduce_net_metrics(self):
        for instrument, grid in self.grids.items():
            target = common.CONFIGURATIONS[instrument][0]
            for _, row in grid.loc[grid["selected_best"]].iterrows():
                frame = reference.cdf.run_backtest(
                    self.prices[instrument].loc[reference.START:reference.TRAIN_END],
                    lookback_days=int(row["signal_lookback_days"]),
                    rebalance_frequency=row["rebalance_frequency"],
                    position_change_threshold_fraction=0.0, signal_cutoff=0.0,
                    signal_mode="rounded", volatility_lookback_days=reference.CDF_VOLATILITY_LOOKBACK,
                    output_start_date=reference.START, position_rounding_increment=0.5,
                    use_volatility_scaling=bool(row["volatility_scaling"]),
                    target_annualized_volatility=target,
                    trading_cost_ratio=reference.cdf.TRADING_COST_RATIOS[instrument],
                )
                stats = reference.statistics(frame)
                for metric in ranking.METRICS:
                    self.assertAlmostEqual(stats[metric], row[metric])
                self.assertEqual(frame.index.max(), reference.TRAIN_END)

    def test_out_of_sample_prices_do_not_affect_grid(self):
        altered = {instrument: prices.copy() for instrument, prices in self.prices.items()}
        for prices in altered.values():
            prices.loc[reference.TEST_START:] *= 2
        grids = search.search(altered)
        for instrument in search.INSTRUMENTS:
            pd.testing.assert_frame_equal(self.grids[instrument], grids[instrument])


if __name__ == "__main__":
    unittest.main()
