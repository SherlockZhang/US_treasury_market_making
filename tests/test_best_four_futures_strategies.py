import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as analysis


class BestFourFuturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = analysis.cdf.load_single_future_prices(include_dbmf=False)

    def test_reference_2y_daily_results_and_summary(self):
        root = analysis.RESULTS / "2y_best_four_strategies_rounded_0p5_2016_2025_and_2026"
        summary = pd.read_csv(root / "summary.csv")
        parameters = analysis.selected_parameters("2Y future")
        for period, start, end in (
            ("2016_2025", analysis.START, analysis.TRAIN_END),
            ("2026", analysis.TEST_START, analysis.TEST_END),
        ):
            frames = analysis.run_period(self.prices["2Y future"].loc[:end], parameters, start)
            for name, slug in zip(analysis.STRATEGIES, analysis.SLUGS):
                with self.subTest(period=period, strategy=name):
                    reference = pd.read_csv(
                        root / f"{period}_{slug}_daily.csv", parse_dates=["date"]
                    ).set_index("date")
                    actual = frames[name]
                    self.assertEqual(list(actual.index), list(reference.index))
                    np.testing.assert_allclose(actual["position"], reference["position"], atol=1e-9)
                    np.testing.assert_allclose(actual["strategy_return"], reference["strategy_return"], atol=1e-9)
                    expected = summary.loc[
                        summary["period"].eq(period) & summary["strategy"].eq(name)
                    ].iloc[0]
                    stats = analysis.statistics(actual)
                    for metric in (
                        "total_return_pct", "annualized_return_pct", "annualized_volatility_pct",
                        "sharpe_ratio", "max_drawdown_pct", "calmar_ratio",
                        "max_drawdown_duration_calendar_days", "win_ratio_pct", "risk_reward_ratio",
                    ):
                        self.assertAlmostEqual(stats[metric], expected[metric], places=7, msg=metric)

    def test_rounded_scaled_training_selection(self):
        for instrument, best in (("5Y future", 252), ("10Y future", 64)):
            with self.subTest(instrument=instrument):
                prices = self.prices[instrument]
                grid = analysis.select_rounded_scaled_lookback(prices)
                self.assertEqual(int(grid.iloc[0]["signal_lookback_days"]), best)
                self.assertEqual(set(grid["signal_lookback_days"]), set(analysis.CDF_LOOKBACKS))
                changed_test_prices = prices.copy()
                changed_test_prices.loc[analysis.TEST_START:] *= 2
                changed_grid = analysis.select_rounded_scaled_lookback(changed_test_prices)
                pd.testing.assert_frame_equal(grid, changed_grid)

    def test_reference_annual_metrics(self):
        root = analysis.RESULTS / "2y_best_four_strategies_rounded_0p5_2016_2025_and_2026"
        reference = pd.read_csv(root / "2016_2025_annual_stats.csv")
        frames = analysis.run_period(
            self.prices["2Y future"].loc[:analysis.TRAIN_END],
            analysis.selected_parameters("2Y future"),
            analysis.START,
        )
        for name, frame in frames.items():
            for year, yearly in frame.groupby(frame.index.year):
                expected = reference.loc[
                    reference["strategy"].eq(name) & reference["year"].eq(year)
                ].iloc[0]
                actual = analysis.statistics(yearly)
                for metric in (
                    "total_return_pct", "annualized_return_pct", "annualized_volatility_pct",
                    "sharpe_ratio", "max_drawdown_pct", "calmar_ratio",
                    "max_drawdown_duration_calendar_days", "win_ratio_pct", "risk_reward_ratio",
                ):
                    if pd.isna(expected[metric]):
                        self.assertTrue(pd.isna(actual[metric]))
                    else:
                        self.assertAlmostEqual(actual[metric], expected[metric], delta=0.00001)


if __name__ == "__main__":
    unittest.main()
