import unittest

import numpy as np
import pandas as pd

import analyze_2y_four_strategies_target_volatility as analysis
import analyze_best_four_futures_strategies as reference


class TwoYearTargetVolatilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[analysis.INSTRUMENT]

    def test_reference_models_and_target(self):
        analysis.validate_reference()
        for period, start, end in (
            ("2016_2025", reference.START, reference.TRAIN_END),
            ("2026", reference.TEST_START, reference.TEST_END),
        ):
            with self.subTest(period=period):
                frames = reference.run_period(
                    self.prices.loc[:end], analysis.PARAMETERS, start,
                    analysis.ROUNDING, analysis.ROUNDING, analysis.TARGET,
                )
                scaled = frames[reference.STRATEGIES[0]]
                eligible = scaled["signal"].notna() & scaled["realized_volatility"].gt(0)
                desired = (
                    scaled.loc[eligible, "signal"] * analysis.TARGET
                    / scaled.loc[eligible, "realized_volatility"]
                )
                np.testing.assert_allclose(
                    scaled.loc[eligible, "desired_position"],
                    (desired / analysis.ROUNDING).round().mul(analysis.ROUNDING).round(1),
                )
                for name, slug in zip(reference.STRATEGIES[1:], reference.SLUGS[1:]):
                    old = pd.read_csv(
                        analysis.SOURCE / f"{period}_{slug}_daily.csv", parse_dates=["date"]
                    ).set_index("date")
                    self.assertEqual(list(frames[name].index), list(old.index))
                    np.testing.assert_allclose(frames[name]["position"], old["position"], atol=1e-9)
                    np.testing.assert_allclose(
                        frames[name]["strategy_return"], old["strategy_return"], atol=1e-9
                    )
                labels = reference.parameter_labels(
                    analysis.PARAMETERS, analysis.ROUNDING, analysis.ROUNDING, analysis.TARGET
                )
                self.assertIn("vol target 1.5%", labels[reference.STRATEGIES[0]])


if __name__ == "__main__":
    unittest.main()
