import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_2y_cost_adjusted_best_strategies as previous
import analyze_2y_expanded_cdf_grid_winners as analysis
import analyze_best_four_futures_strategies as reference


class ExpandedCdfWinnersTests(unittest.TestCase):
    def test_selected_models_and_reports(self):
        parameters, frequencies, rounding, sharpes = analysis.selected_parameters()
        self.assertEqual(parameters["Normal CDF scaled"]["lookback"], 64)
        self.assertEqual(parameters["Normal CDF unscaled"]["lookback"], 252)
        self.assertEqual(frequencies, ("weekly", "daily"))
        self.assertEqual(rounding, (1.5, 1.0))
        prices = reference.cdf.load_single_future_prices(False)[previous.INSTRUMENT]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(reference, "save_plots"):
                reference.save_report(
                    previous.INSTRUMENT, prices, parameters, directory,
                    rounding[0], rounding[1], previous.TARGET, frequencies,
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
                        frame["position_change"] * reference.cdf.TRADING_COST_RATIOS[previous.INSTRUMENT],
                        atol=1e-10,
                    )
                    self.assertAlmostEqual(row["trading_cost_accumulated"], frame["trading_cost"].sum(), places=7)
                    self.assertAlmostEqual(
                        row["total_return_pct"], ((1 + frame["strategy_return"]).prod() - 1) * 100,
                        delta=0.00001,
                    )
            labels = summary.set_index(["period", "strategy"])["parameters"]
            self.assertIn("weekly T=64", labels.loc[("2016_2025", "Normal CDF scaled")])
            self.assertIn("position rounded 1.5", labels.loc[("2016_2025", "Normal CDF scaled")])
            self.assertIn("daily T=252", labels.loc[("2016_2025", "Normal CDF unscaled")])
            self.assertIn("position rounded 1", labels.loc[("2016_2025", "Normal CDF unscaled")])


if __name__ == "__main__":
    unittest.main()
