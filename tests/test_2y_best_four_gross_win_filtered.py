import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_2y_best_four_gross_win_filtered as report
import analyze_best_four_futures_strategies as reference


class GrossFilteredFourStrategyTests(unittest.TestCase):
    def test_selected_winners_and_report(self):
        grids = [
            pd.read_csv(report.search.OUTPUT_DIR / f"{model}_best_parameters.csv")
            for model in ("cdf", "ewma", "kama")
        ]
        cdf = grids[0].loc[grids[0]["instrument"].eq("2Y future")]
        winners = [
            cdf.loc[cdf["volatility_scaling"].eq(True)].iloc[0],
            cdf.loc[cdf["volatility_scaling"].eq(False)].iloc[0],
            *[grid.loc[grid["instrument"].eq("2Y future")].iloc[0] for grid in grids[1:]],
        ]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(reference, "save_plots"):
                report.save_report(directory)
            summary = pd.read_csv(directory / "summary.csv").set_index(["period", "strategy"])
            annual = pd.read_csv(directory / "2016_2025_annual_stats.csv")
            self.assertEqual(len(summary), 8)
            for name, slug, winner in zip(reference.STRATEGIES, reference.SLUGS, winners):
                training = summary.loc[("2016_2025", name)]
                self.assertAlmostEqual(training["sharpe_ratio"], winner["sharpe_ratio"], places=7)
                self.assertAlmostEqual(training["win_ratio_pct"], winner["win_ratio_pct"], places=7)
                self.assertGreater(training["win_ratio_pct"], 50)
                for period in ("2016_2025", "2026"):
                    frame = pd.read_csv(directory / f"{period}_{slug}_daily.csv")
                    row = summary.loc[(period, name)]
                    gross, net = frame["gross_strategy_return"], frame["strategy_return"]
                    ratio = gross.gt(0).sum() / (gross.gt(0).sum() + gross.lt(0).sum()) * 100
                    self.assertAlmostEqual(row["win_ratio_pct"], ratio, places=7)
                    self.assertAlmostEqual(row["sharpe_ratio"], net.mean() / net.std(ddof=1) * np.sqrt(252), places=6)
                    np.testing.assert_allclose(
                        frame["trading_cost"],
                        frame["position_change"] * reference.cdf.TRADING_COST_RATIOS["2Y future"],
                        atol=1e-10,
                    )
                    self.assertAlmostEqual(row["trading_cost_accumulated"], frame["trading_cost"].sum(), places=7)
                    self.assertAlmostEqual(row["total_return_pct"], ((1 + net).prod() - 1) * 100, delta=0.00001)
                self.assertAlmostEqual(
                    annual.loc[annual["strategy"].eq(name), "trading_cost_accumulated"].sum(),
                    training["trading_cost_accumulated"], places=7,
                )


if __name__ == "__main__":
    unittest.main()
