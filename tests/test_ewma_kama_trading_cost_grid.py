import unittest
from itertools import product

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import backtest_ewma_extreme_strategy as ewma
import backtest_kama_extreme_strategy as kama
import search_ewma_kama_trading_cost_grid as search


class CostAdjustedGridTests(unittest.TestCase):
    def test_parameter_space_matches_reference_grids(self):
        expected = {
            "EWMA": {
                "n": (5, 10, 20, 32, 64, 126, 252, 504),
                "m": (5, 10, 20, 32, 64, 126, 252, 504),
            },
            "KAMA": {
                "alpha_fast_n": (2, 3, 5, 10),
                "alpha_slow_n": (15, 20, 25, 64, 126, 252, 504),
                "er_lookback_days": (5, 10, 15, 20, 25),
                "kama_extreme_window_m": (3, 32, 64, 252),
            },
        }
        self.assertEqual(search.PARAMETERS, expected)
        expected_counts = {"EWMA": 64, "KAMA": 560}
        for model, parameters in expected.items():
            self.assertEqual(
                len(list(product(*parameters.values()))), expected_counts[model]
            )
        space = search.parameter_space()
        for model, parameters in expected.items():
            for parameter, values in parameters.items():
                listed = space.loc[
                    space["model"].eq(model) & space["parameter"].eq(parameter),
                    "value",
                ]
                self.assertEqual(list(listed), list(values))

    def test_gross_matches_historical_and_net_uses_costs(self):
        prices = reference.cdf.load_single_future_prices(False)["2Y future"].loc[
            reference.START:reference.TRAIN_END
        ]
        for run, args, expected in (
            (ewma.run_backtest, (5, 40), 0.90187368),
            (kama.run_backtest, (5, 25, 25, 3), 0.77252346),
        ):
            with self.subTest(model=run.__module__):
                gross = run(prices, *args, trading_cost_ratio=0)
                net = run(prices, *args)
                self.assertAlmostEqual(reference.statistics(gross)["sharpe_ratio"], expected, places=7)
                np.testing.assert_allclose(net["position"], gross["position"])
                expected_cost = (
                    net["position_change"] * 0.0039
                    / net["price"].shift(1).fillna(net["price"])
                )
                np.testing.assert_allclose(
                    net["strategy_return"], gross["strategy_return"] - expected_cost,
                )
                self.assertEqual(net["trading_cost"].dtype, np.dtype("float64"))
                self.assertAlmostEqual(
                    net["trading_cost"].sum(), expected_cost.sum(), places=14,
                )

    def test_position_size_scales_ewma_and_kama_exposure_and_costs(self):
        prices = reference.cdf.load_single_future_prices(False)["2Y future"].loc[
            reference.START:reference.TRAIN_END
        ]
        for run, args in (
            (ewma.run_backtest, (5, 40)),
            (kama.run_backtest, (5, 25, 25, 3)),
        ):
            with self.subTest(model=run.__module__):
                unit = run(prices, *args)
                triple = run(prices, *args, position_size=3)
                np.testing.assert_array_equal(
                    triple["position"], unit["position"] * 3,
                )
                np.testing.assert_allclose(
                    triple["position_change"], unit["position_change"] * 3,
                )
                np.testing.assert_allclose(
                    triple["trading_cost"], unit["trading_cost"] * 3,
                )
                np.testing.assert_allclose(
                    triple["strategy_return"], unit["strategy_return"] * 3,
                )
                self.assertEqual(set(triple["position"].unique()), {-3, 0, 3})

    def test_search_ranks_net_sharpe_and_ignores_out_of_sample_prices(self):
        dates = pd.bdate_range("2015-12-01", "2026-03-01")
        prices = pd.Series(
            100 + np.arange(len(dates)) * 0.02 + np.sin(np.arange(len(dates)) / 5),
            index=dates, name="wrong name",
        )
        original = search.PARAMETERS
        try:
            search.PARAMETERS = {
                "EWMA": {"n": (5, 10), "m": (5,)},
                "KAMA": {
                    "alpha_fast_n": (2,), "alpha_slow_n": (10,),
                    "er_lookback_days": (5,), "kama_extreme_window_m": (3,),
                },
            }
            results = search.search({"2Y future": prices})
            changed = prices.copy()
            changed.loc[reference.TEST_START:] *= 10
            repeated = search.search({"2Y future": changed})
        finally:
            search.PARAMETERS = original
        for model, grid in results.items():
            pd.testing.assert_frame_equal(grid, repeated[model])
            self.assertTrue(grid["sharpe_ratio"].is_monotonic_decreasing)
            self.assertEqual(grid["selected_best"].sum(), 1)
            row = grid.iloc[0]
            training = prices.loc[reference.START:reference.TRAIN_END].rename("2Y future")
            if model == "EWMA":
                frame = ewma.run_backtest(training, int(row["n"]), int(row["m"]))
            else:
                frame = kama.run_backtest(training, 2, 10, 5, 3)
            self.assertAlmostEqual(row["sharpe_ratio"], reference.statistics(frame)["sharpe_ratio"])
            self.assertAlmostEqual(
                row["trading_cost_accumulated"], frame["trading_cost"].sum(),
            )
            self.assertEqual(row["training_start_date"], training.index[0].date())
            self.assertEqual(row["training_end_date"], training.index[-1].date())


if __name__ == "__main__":
    unittest.main()
