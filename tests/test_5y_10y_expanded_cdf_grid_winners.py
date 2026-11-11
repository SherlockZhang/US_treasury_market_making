import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_2y_expanded_cdf_grid_winners as winners
import analyze_5y_10y_expanded_cdf_grid_winners as analysis
import analyze_best_four_futures_strategies as reference
import search_cdf_custom_vol_daily_weekly as cdf_grid


class FiveAndTenYearWinnersTests(unittest.TestCase):
    def test_selected_models_and_reports(self):
        prices = reference.cdf.load_single_future_prices(False)
        for instrument in analysis.INSTRUMENTS:
            parameters, frequencies, rounding, sharpes = winners.selected_parameters(instrument)
            target, _ = cdf_grid.CONFIGURATIONS[instrument]
            with self.subTest(instrument=instrument), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                with patch.object(reference, "save_plots"):
                    reference.save_report(
                        instrument, prices[instrument], parameters, directory,
                        rounding[0], rounding[1], target, frequencies,
                    )
                summary = pd.read_csv(directory / "summary.csv")
                annual = pd.read_csv(directory / "2016_2025_annual_stats.csv")
                for name, slug in zip(reference.STRATEGIES, reference.SLUGS):
                    training = summary.loc[
                        summary["period"].eq("2016_2025") & summary["strategy"].eq(name)
                    ].iloc[0]
                    self.assertAlmostEqual(training["sharpe_ratio"], sharpes[name], places=7)
                    for period in ("2016_2025", "2026"):
                        frame = pd.read_csv(
                            directory / f"{period}_{slug}_daily.csv", parse_dates=["date"]
                        )
                        row = summary.loc[
                            summary["period"].eq(period) & summary["strategy"].eq(name)
                        ].iloc[0]
                        np.testing.assert_allclose(
                            frame["trading_cost"],
                            frame["position_change"] * reference.cdf.TRADING_COST_RATIOS[instrument],
                            atol=1e-10,
                        )
                        np.testing.assert_allclose(
                            frame["strategy_return"],
                            frame["gross_strategy_return"] - frame["trading_cost"], atol=2e-10,
                        )
                        self.assertAlmostEqual(row["trading_cost_accumulated"], frame["trading_cost"].sum(), places=7)
                        self.assertAlmostEqual(
                            row["total_return_pct"],
                            (float(np.prod(1 + frame["strategy_return"].to_numpy(dtype=float))) - 1) * 100,
                            delta=0.00001,
                        )
                        if period == "2016_2025":
                            for year, yearly in frame.groupby(frame["date"].dt.year):
                                annual_row = annual.loc[
                                    annual["strategy"].eq(name) & annual["year"].eq(year)
                                ].iloc[0]
                                self.assertAlmostEqual(
                                    annual_row["trading_cost_accumulated"], yearly["trading_cost"].sum(),
                                    delta=1e-8,
                                )
                for index, name in enumerate(reference.STRATEGIES[:2]):
                    label = summary.loc[
                        summary["period"].eq("2016_2025") & summary["strategy"].eq(name),
                        "parameters",
                    ].iloc[0]
                    self.assertIn(frequencies[index], label)
                    self.assertIn(f"position rounded {rounding[index]:g}", label)


if __name__ == "__main__":
    unittest.main()
