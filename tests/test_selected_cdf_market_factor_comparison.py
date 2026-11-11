import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_selected_cdf_market_factor_comparison as report
import analyze_best_four_futures_strategies as reference


class SelectedFactorComparisonTests(unittest.TestCase):
    def test_comparison_and_prior_unenhanced_reports(self):
        prior = {
            "2Y future": "2y_best_four_gross_win_filtered_2016_2025_and_2026",
            "5Y future": "5y_best_four_gross_win_filtered_2016_2025_and_2026",
            "10Y future": "10y_four_gross_filtered_weekly_t64_rounding_0p5_2016_2025_and_2026",
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(report.comparison, "save_plots"):
                report.save_report(directory)
            summary = pd.read_csv(directory / "summary.csv")
            coverage = pd.read_csv(directory / "market_factor_data_coverage.csv")
            self.assertEqual(len(summary), 6)
            self.assertEqual(len(coverage), sum(
                len(reference.cdf.INSTRUMENT_MARKET_FACTOR_PROFILES[i]) for i in prior
            ))
            for instrument, folder in prior.items():
                with self.subTest(instrument=instrument):
                    stem = instrument.split()[0].lower()
                    winner = report.selected_row(instrument)
                    self.assertEqual(winner["sharpe_rank"], 3 if stem == "10y" else 1)
                    baseline = pd.read_csv(reference.RESULTS / folder / "2026_cdf_scaled_daily.csv")
                    for mode, suffix in (("none", "without"), ("instrument", "with")):
                        frame = pd.read_csv(directory / f"{stem}_{suffix}_market_factors_daily.csv")
                        row = summary.loc[
                            summary["instrument"].eq(instrument) & summary["market_factor_mode"].eq(mode)
                        ].iloc[0]
                        self.assertEqual(row["selected_grid_sharpe_rank"], winner["sharpe_rank"])
                        gross, net = frame["gross_strategy_return"], frame["strategy_return"]
                        win_ratio = gross.gt(0).sum() / (gross.gt(0).sum() + gross.lt(0).sum()) * 100
                        self.assertAlmostEqual(row["win_ratio_pct"], win_ratio, places=7)
                        self.assertAlmostEqual(row["sharpe_ratio"], net.mean() / net.std(ddof=1) * np.sqrt(252), places=6)
                        self.assertAlmostEqual(row["total_return_pct"], ((1 + net).prod() - 1) * 100, delta=0.00001)
                        np.testing.assert_allclose(frame["trading_cost"],
                            frame["position_change"] * reference.cdf.TRADING_COST_RATIOS[instrument], atol=1e-10)
                        self.assertAlmostEqual(row["trading_cost_accumulated"], frame["trading_cost"].sum(), places=7)
                        if mode == "none":
                            for column in ("date", "position", "strategy_return"):
                                pd.testing.assert_series_equal(frame[column], baseline[column])
                        else:
                            np.testing.assert_allclose(
                                frame["signal"], frame["raw_signal"] * frame["market_factor_signal_multiplier"],
                                atol=1e-9, equal_nan=True,
                            )
                            self.assertTrue(frame["market_factor_signal_multiplier"].ne(1).any())


if __name__ == "__main__":
    unittest.main()
