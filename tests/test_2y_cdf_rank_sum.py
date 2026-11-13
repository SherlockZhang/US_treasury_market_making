import unittest
from itertools import product

import numpy as np
import pandas as pd

import analyze_2y_four_cdf_modes_cap3 as capped
import analyze_best_four_futures_strategies as reference
import search_2y_cdf_rank_sum as search


class RankSumTests(unittest.TestCase):
    def test_equal_metric_ranks_and_winner_ties(self):
        grid = pd.DataFrame({
            "volatility_scaling": [True, True, True],
            "rebalance_frequency": ["daily"] * 3,
            "signal_lookback_days": [32, 64, 126],
            "sharpe_ratio": [2.0, 2.0, 1.0],
            "max_drawdown_pct": [-1.0, -1.0, -2.0],
            "calmar_ratio": [2.0, 2.0, 1.0],
        })
        ranked = search.rank_grid(grid)
        self.assertEqual(ranked["sharpe_rank"].tolist(), [1, 1, 3])
        self.assertEqual(ranked["rank_sum"].tolist(), [3, 3, 9])
        self.assertEqual(ranked["selected_best"].tolist(), [True, True, False])
        grid.loc[0, "calmar_ratio"] = np.nan
        with self.assertRaises(ValueError):
            search.rank_grid(grid)

    def test_actual_grid_selects_lowest_independent_rank_sum(self):
        prices = reference.cdf.load_single_future_prices(include_dbmf=False)[capped.INSTRUMENT]
        grid = search.rank_grid(capped.search(prices))
        self.assertEqual(len(grid), 20)
        for scaled, frequency in product((True, False), capped.FREQUENCIES):
            mode = grid.loc[
                grid["volatility_scaling"].eq(scaled)
                & grid["rebalance_frequency"].eq(frequency)
            ]
            self.assertEqual(len(mode), 5)
            for metric, rank_name in search.METRICS.items():
                expected = mode[metric].rank(ascending=False, method="min").astype(int)
                pd.testing.assert_series_equal(mode[rank_name], expected, check_names=False)
            expected_sum = mode[list(search.METRICS.values())].sum(axis=1)
            pd.testing.assert_series_equal(mode["rank_sum"], expected_sum, check_names=False)
            winners = mode.loc[mode["selected_best"]]
            self.assertEqual(len(winners), 1)
            self.assertEqual(winners.iloc[0]["rank_sum"], mode["rank_sum"].min())


if __name__ == "__main__":
    unittest.main()
