import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_2y_cost_adjusted_best_strategies as analysis
import analyze_best_four_futures_strategies as reference


class BestTwoYearStrategiesTests(unittest.TestCase):
    def test_selected_training_sharpes_and_period_outputs(self):
        parameters, frequencies, sharpes = analysis.selected_parameters()
        self.assertEqual(frequencies, ("weekly", "daily"))
        prices = reference.cdf.load_single_future_prices(False)[analysis.INSTRUMENT]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(reference, "save_plots"):
                reference.save_report(
                    analysis.INSTRUMENT, prices, parameters, directory,
                    analysis.ROUNDING, analysis.ROUNDING, analysis.TARGET, frequencies,
                )
            summary = pd.read_csv(directory / "summary.csv")
            for name, slug in zip(reference.STRATEGIES, reference.SLUGS):
                row = summary.loc[
                    summary["period"].eq("2016_2025") & summary["strategy"].eq(name)
                ].iloc[0]
                self.assertAlmostEqual(row["sharpe_ratio"], sharpes[name], places=7)
                for period in ("2016_2025", "2026"):
                    frame = pd.read_csv(directory / f"{period}_{slug}_daily.csv")
                    row = summary.loc[
                        summary["period"].eq(period) & summary["strategy"].eq(name)
                    ].iloc[0]
                    np.testing.assert_allclose(
                        frame["trading_cost"],
                        frame["position_change"] * reference.cdf.TRADING_COST_RATIOS[analysis.INSTRUMENT],
                        atol=1e-10,
                    )
                    self.assertAlmostEqual(row["trading_cost_accumulated"], frame["trading_cost"].sum(), places=7)
            labels = summary.set_index(["period", "strategy"])["parameters"]
            self.assertTrue(labels.loc[("2016_2025", "Normal CDF unscaled")].startswith("daily "))
            self.assertTrue(labels.loc[("2016_2025", "Normal CDF scaled")].startswith("weekly "))


if __name__ == "__main__":
    unittest.main()
