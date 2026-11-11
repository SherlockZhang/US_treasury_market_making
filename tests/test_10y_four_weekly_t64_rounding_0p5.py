import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_10y_four_weekly_t64_rounding_0p5 as report
import analyze_2y_best_four_gross_win_filtered as baseline
import analyze_best_four_futures_strategies as reference


class ThirdScaledCandidateReportTests(unittest.TestCase):
    def test_candidate_and_reports(self):
        candidate = report.selected_candidate()
        self.assertEqual(candidate["sharpe_rank"], 3)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "candidate"
            original = Path(temp) / "original"
            with patch.object(reference, "save_plots"):
                report.save_report(directory)
                baseline.save_report(original, report.INSTRUMENT)
            summary = pd.read_csv(directory / "summary.csv").set_index(["period", "strategy"])
            selected = pd.read_csv(directory / "scaled_cdf_selected_grid_row.csv").iloc[0]
            self.assertEqual(selected["sharpe_rank"], 3)
            scaled = summary.loc[("2016_2025", "Normal CDF scaled")]
            self.assertIn("weekly T=64", scaled["parameters"])
            self.assertIn("position rounded 0.5", scaled["parameters"])
            self.assertAlmostEqual(scaled["sharpe_ratio"], candidate["sharpe_ratio"], places=7)
            self.assertAlmostEqual(scaled["win_ratio_pct"], candidate["win_ratio_pct"], places=7)
            for name, slug in zip(reference.STRATEGIES, reference.SLUGS):
                for period in ("2016_2025", "2026"):
                    frame = pd.read_csv(directory / f"{period}_{slug}_daily.csv")
                    row = summary.loc[(period, name)]
                    gross, net = frame["gross_strategy_return"], frame["strategy_return"]
                    ratio = gross.gt(0).sum() / (gross.gt(0).sum() + gross.lt(0).sum()) * 100
                    self.assertAlmostEqual(row["win_ratio_pct"], ratio, places=7)
                    self.assertAlmostEqual(row["sharpe_ratio"], net.mean() / net.std(ddof=1) * np.sqrt(252), places=6)
                    self.assertAlmostEqual(row["total_return_pct"], ((1 + net).prod() - 1) * 100, delta=0.00001)
                    np.testing.assert_allclose(
                        frame["trading_cost"],
                        frame["position_change"] * reference.cdf.TRADING_COST_RATIOS[report.INSTRUMENT],
                        atol=1e-10,
                    )
                    self.assertAlmostEqual(row["trading_cost_accumulated"], frame["trading_cost"].sum(), places=7)
                    if name != "Normal CDF scaled":
                        original_frame = pd.read_csv(original / f"{period}_{slug}_daily.csv")
                        pd.testing.assert_frame_equal(frame, original_frame)


if __name__ == "__main__":
    unittest.main()
