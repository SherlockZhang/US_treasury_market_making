import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as analysis


class BestScaledCdfReportsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)

    def test_scaled_matches_grid_and_other_strategies_match_reference(self):
        for instrument, config in analysis.REQUESTED_CONFIGURATIONS.items():
            stem = instrument.split()[0].lower()
            grid = pd.read_csv(
                reference.RESULTS / f"{stem}_scaled_cdf_rounding_grid_2016_2025.csv"
            )
            matching = grid.loc[
                grid["signal_lookback_days"].eq(config["parameters"]["Normal CDF scaled"]["lookback"])
                & grid["position_rounding_increment"].eq(config["rounding"])
            ]
            self.assertEqual(len(matching), 1)
            selected = matching.iloc[0]
            old_dir = reference.RESULTS / (
                f"{stem}_best_four_strategies_rounded_0p5_2016_2025_and_2026"
            )
            for period, start, end in (
                ("2016_2025", reference.START, reference.TRAIN_END),
                ("2026", reference.TEST_START, reference.TEST_END),
            ):
                with self.subTest(instrument=instrument, period=period):
                    frames = reference.run_period(
                        self.prices[instrument].loc[:end],
                        config["parameters"],
                        start,
                        config["rounding"],
                    )
                    scaled = frames["Normal CDF scaled"]
                    self.assertLessEqual(scaled.index.max(), end)
                    if period == "2016_2025":
                        stats = reference.statistics(scaled)
                        for metric in ("sharpe_ratio", "total_return_pct",
                                       "annualized_return_pct", "annualized_volatility_pct",
                                       "max_drawdown_pct"):
                            self.assertAlmostEqual(stats[metric], selected[metric], places=7)
                    for name, slug in zip(reference.STRATEGIES[1:], reference.SLUGS[1:]):
                        original = pd.read_csv(
                            old_dir / f"{period}_{slug}_daily.csv", parse_dates=["date"]
                        ).set_index("date")
                        actual = frames[name]
                        self.assertEqual(list(actual.index), list(original.index))
                        np.testing.assert_allclose(actual["position"], original["position"], atol=1e-9)
                        np.testing.assert_allclose(
                            actual["strategy_return"], original["strategy_return"], atol=1e-9
                        )


if __name__ == "__main__":
    unittest.main()
