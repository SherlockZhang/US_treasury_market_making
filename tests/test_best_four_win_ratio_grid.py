import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import search_best_four_win_ratio_grid as search


class WinRatioSelectionTests(unittest.TestCase):
    def test_win_ratio_uses_gross_while_sharpe_uses_net(self):
        frame = pd.DataFrame({
            "position": [1.0, 1.0, 1.0, 0.0],
            "gross_strategy_return": [0.01, 0.001, -0.005, 0.0],
            "strategy_return": [0.01, -0.002, -0.005, -0.003],
        }, index=pd.date_range("2020-01-01", periods=4))
        stats = reference.statistics(frame)
        self.assertAlmostEqual(stats["win_ratio_pct"], 200 / 3)
        expected = frame["strategy_return"].mean() / frame["strategy_return"].std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(stats["sharpe_ratio"], expected)
        self.assertAlmostEqual(stats["risk_reward_ratio"], 0.01 / ((0.002 + 0.005 + 0.003) / 3))

    def test_gross_zero_days_do_not_count(self):
        frame = pd.DataFrame({
            "position": [1.0, 0.0, 0.0],
            "gross_strategy_return": [0.01, 0.0, 0.0],
            "strategy_return": [0.01, -0.001, 0.0],
        }, index=pd.date_range("2020-01-01", periods=3))
        self.assertEqual(reference.statistics(frame)["win_ratio_pct"], 100)
        frame["gross_strategy_return"] = 0.0
        self.assertTrue(np.isnan(reference.statistics(frame)["win_ratio_pct"]))

    def test_strict_filter_ranking_and_missing_values(self):
        frame = pd.DataFrame({
            "instrument": ["2Y future"] * 6 + ["5Y future"],
            "sharpe_rank": [1, 2, 3, 4, 5, 6, 1],
            "sharpe_ratio": [3.0, 2.0, 1.5, 1.0, np.nan, 0.9, 1.0],
            "win_ratio_pct": [50.0, 50.00001, 75.0, np.nan, np.nan, 49.0, 49.0],
            "selected_best": [True, False, False, False, False, False, True],
        })
        result = search.rank_eligible(frame, ["instrument"])
        two_year = result.loc[result["instrument"].eq("2Y future")]
        self.assertEqual(two_year.loc[two_year["eligible"], "sharpe_rank"].tolist(), [1, 2])
        winner = two_year.loc[two_year["selected_best"]].iloc[0]
        self.assertEqual(winner["sharpe_ratio"], 2.0)
        exactly_50 = two_year.loc[two_year["win_ratio_pct"].eq(50)].iloc[0]
        self.assertFalse(exactly_50["eligible"])
        self.assertEqual(exactly_50["selection_status"], "win_ratio_not_above_50pct")
        self.assertTrue(pd.isna(exactly_50["sharpe_rank"]))
        self.assertEqual(
            two_year.loc[two_year["sharpe_ratio"].eq(1.0), "selection_status"].iloc[0],
            "undefined_win_ratio",
        )
        self.assertEqual(
            two_year.loc[two_year["sharpe_ratio"].isna(), "selection_status"].iloc[0],
            "undefined_sharpe",
        )
        self.assertFalse(result.loc[result["instrument"].eq("5Y future"), "selected_best"].any())

    def test_rank_separately_by_instrument_and_cdf_scaling(self):
        frame = pd.DataFrame({
            "instrument": ["2Y future"] * 4,
            "volatility_scaling": [True, True, False, False],
            "sharpe_rank": [1, 2, 1, 2],
            "sharpe_ratio": [2.0, 1.0, 4.0, 3.0],
            "win_ratio_pct": [49.0, 51.0, 55.0, 51.0],
            "selected_best": [True, False, True, False],
        })
        result = search.rank_eligible(frame, ["instrument", "volatility_scaling"])
        winners = result.loc[result["selected_best"]]
        self.assertEqual(len(winners), 2)
        self.assertEqual(set(winners["sharpe_ratio"]), {1.0, 4.0})


if __name__ == "__main__":
    unittest.main()
