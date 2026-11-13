import unittest
from functools import partial
from itertools import product

import numpy as np
import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as report
import analyze_best_four_futures_strategies as reference


class TwoYearCappedGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[
            report.INSTRUMENT
        ]
        cls.grid = report.search(cls.prices)

    def test_grid_winners_and_parameters(self):
        self.assertEqual(len(self.grid), 20)
        self.assertTrue(self.grid["position_rounding_increment"].eq(0.5).all())
        self.assertTrue(self.grid["position_cap"].eq(3.0).all())
        self.assertTrue(self.grid.loc[
            self.grid["volatility_scaling"], "target_annualized_volatility_pct"
        ].eq(1.5).all())
        winners = self.grid.loc[self.grid["sharpe_rank"].eq(1)]
        self.assertEqual(len(winners), 4)
        self.assertEqual(
            set(zip(winners["volatility_scaling"], winners["rebalance_frequency"])),
            set(product((True, False), report.FREQUENCIES)),
        )
        for scaled, frequency in product((True, False), report.FREQUENCIES):
            mode = self.grid.loc[
                self.grid["volatility_scaling"].eq(scaled)
                & self.grid["rebalance_frequency"].eq(frequency)
            ]
            self.assertEqual(set(mode["signal_lookback_days"]), set(reference.CDF_LOOKBACKS))
            self.assertEqual(mode["sharpe_rank"].tolist(), list(range(1, 6)))
            self.assertTrue(mode["sharpe_ratio"].is_monotonic_decreasing)
            row = mode.iloc[0]
            frame = report.backtest_cdf(
                self.prices, int(row["signal_lookback_days"]), frequency, scaled,
                reference.START, reference.TRAIN_END,
            )
            self.assertAlmostEqual(reference.statistics(frame)["sharpe_ratio"], row["sharpe_ratio"])
            self.assertLessEqual(frame["position"].abs().max(), 3)

    def test_actual_cap_costs_and_uncapped_compatibility(self):
        prices = self.prices.loc[reference.START:reference.TRAIN_END]
        backtest = partial(
            reference.cdf.run_backtest, prices,
            lookback_days=64, rebalance_frequency="daily", signal_mode="rounded",
            signal_cutoff=0, position_change_threshold_fraction=0,
            volatility_lookback_days=64, output_start_date=reference.START,
            position_rounding_increment=0.5, target_annualized_volatility=0.015,
        )
        original = backtest()
        explicit_none = backtest(position_cap=None)
        pd.testing.assert_frame_equal(original, explicit_none)
        capped = backtest(position_cap=3)
        self.assertGreater(int(original["position"].abs().gt(3).sum()), 0)
        np.testing.assert_allclose(capped["uncapped_position"], original["position"])
        np.testing.assert_allclose(capped["position"], original["position"].clip(-3, 3))
        expected_change = capped["position"].diff().fillna(capped["position"]).abs()
        previous_price = capped["price"].shift(1).fillna(capped["price"])
        expected_cost = expected_change * reference.cdf.TRADING_COST_RATIOS[report.INSTRUMENT] / previous_price
        np.testing.assert_allclose(capped["trading_cost"], expected_cost)
        np.testing.assert_allclose(
            capped["strategy_return"], capped["position"] * capped["daily_return"].fillna(0) - expected_cost
        )
        np.testing.assert_allclose(capped["wealth"], (1 + capped["strategy_return"]).cumprod())
        for value in (0, -1, np.inf, np.nan):
            with self.subTest(cap=value), self.assertRaises(ValueError):
                backtest(position_cap=value)

    def test_out_of_sample_prices_do_not_affect_selection(self):
        altered = self.prices.copy()
        altered.loc[reference.TEST_START:] *= 2
        pd.testing.assert_frame_equal(self.grid, report.search(altered))

    def test_six_training_models_match_grids(self):
        winners = self.grid.loc[self.grid["sharpe_rank"].eq(1)]
        frames, labels = report.build_frames(
            self.prices, winners, reference.START, reference.TRAIN_END
        )
        self.assertEqual(len(frames), 6)
        self.assertEqual(set(frames), set(labels))


if __name__ == "__main__":
    unittest.main()
