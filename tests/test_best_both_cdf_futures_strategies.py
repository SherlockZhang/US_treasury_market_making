import unittest

import numpy as np
import pandas as pd

import analyze_best_both_cdf_futures_strategies as analysis
import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as previous


class BothCdfComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)

    def test_selected_models_and_unchanged_strategies(self):
        for instrument, config in previous.REQUESTED_CONFIGURATIONS.items():
            grid = analysis.selected_grid(instrument, config)
            selected = grid.loc[grid["selected_for_report"]]
            self.assertEqual(len(selected), 2)
            stem = instrument.split()[0].lower()
            old_dir = reference.RESULTS / f"{stem}_best_four_strategies_rounded_0p5_2016_2025_and_2026"
            for period, start, end in (
                ("2016_2025", reference.START, reference.TRAIN_END),
                ("2026", reference.TEST_START, reference.TEST_END),
            ):
                with self.subTest(instrument=instrument, period=period):
                    frames = reference.run_period(
                        self.prices[instrument].loc[:end], config["parameters"], start,
                        config["rounding"], config["rounding"],
                    )
                    for scaled, name in ((True, reference.STRATEGIES[0]), (False, reference.STRATEGIES[1])):
                        frame = frames[name]
                        expected = selected.loc[selected["volatility_scaling"].eq(scaled)].iloc[0]
                        self.assertLessEqual(frame.index.max(), end)
                        if period == "2016_2025":
                            gross = frame.assign(strategy_return=frame["gross_strategy_return"])
                            stats = reference.statistics(gross)
                            for metric in ("sharpe_ratio", "total_return_pct",
                                           "annualized_return_pct", "annualized_volatility_pct",
                                           "max_drawdown_pct"):
                                self.assertAlmostEqual(stats[metric], expected[metric], places=7)
                        values = frame["desired_position"].dropna()
                        np.testing.assert_allclose(
                            values / config["rounding"],
                            np.round(values / config["rounding"]), atol=1e-10,
                        )
                    for name, slug in zip(reference.STRATEGIES[2:], reference.SLUGS[2:]):
                        original = pd.read_csv(
                            old_dir / f"{period}_{slug}_daily.csv", parse_dates=["date"]
                        ).set_index("date")
                        self.assertEqual(list(frames[name].index), list(original.index))
                        np.testing.assert_allclose(frames[name]["position"], original["position"], atol=1e-9)
                        np.testing.assert_allclose(
                            frames[name]["gross_strategy_return"], original["strategy_return"], atol=1e-9
                        )
                    labels = reference.parameter_labels(
                        config["parameters"], config["rounding"], config["rounding"]
                    )
                    for name in reference.STRATEGIES[:2]:
                        self.assertIn(f"position rounded {config['rounding']:g}", labels[name])


if __name__ == "__main__":
    unittest.main()
