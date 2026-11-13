import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_ewma_kama_rank_sum as search
import search_ewma_kama_trading_cost_grid as previous
from rank_search_metrics import METRICS, rank_metrics


class EwmaKamaRankSumTests(unittest.TestCase):
    def test_metric_ranks_ties_and_rejects_nonfinite(self):
        frame = pd.DataFrame({
            "instrument": ["2Y future"] * 3 + ["5Y future"] * 2,
            "sharpe_ratio": [3., 2., 2., 4., 1.],
            "max_drawdown_pct": [-3., -1., -1., -1., -2.],
            "calmar_ratio": [1., 3., 3., 4., 1.],
        })
        ranked = rank_metrics(frame, ["instrument"])
        self.assertEqual(ranked["rank_sum"].tolist(), [7, 4, 4, 3, 6])
        self.assertEqual(ranked["selected_best"].tolist(), [False, True, True, True, False])
        frame.loc[0, "max_drawdown_pct"] = np.nan
        with self.assertRaises(ValueError):
            rank_metrics(frame, ["instrument"])

    def test_actual_search_ranks_each_model_and_instrument(self):
        prices = reference.cdf.load_single_future_prices(include_dbmf=False)
        subset = {name: prices[name] for name in ("2Y future", "5Y future")}
        parameters = {
            "EWMA": {"n": (5, 10), "m": (5, 10)},
            "KAMA": {
                "alpha_fast_n": (2, 5), "alpha_slow_n": (10,),
                "er_lookback_days": (5, 10), "kama_extreme_window_m": (3,),
            },
        }
        with patch.object(previous, "PARAMETERS", parameters):
            grids = search.search(subset)
            changed = {name: series.copy() for name, series in subset.items()}
            for series in changed.values():
                series.loc[reference.TEST_START:] *= 2
            repeated = search.search(changed)
        for model, grid in grids.items():
            pd.testing.assert_frame_equal(grid, repeated[model])
            self.assertEqual(len(grid), 8)
            for instrument in subset:
                mode = grid.loc[grid["instrument"].eq(instrument)]
                for metric, rank_name in METRICS.items():
                    expected = mode[metric].rank(ascending=False, method="min").astype(int)
                    pd.testing.assert_series_equal(mode[rank_name], expected, check_names=False)
                expected_sum = mode[list(METRICS.values())].sum(axis=1)
                pd.testing.assert_series_equal(mode["rank_sum"], expected_sum, check_names=False)
                winners = mode.loc[mode["selected_best"]]
                self.assertGreaterEqual(len(winners), 1)
                self.assertTrue(winners["rank_sum"].eq(mode["rank_sum"].min()).all())
                self.assertTrue(mode["trading_cost_ratio"].eq(
                    reference.cdf.TRADING_COST_RATIOS[instrument]
                ).all())


if __name__ == "__main__":
    unittest.main()
