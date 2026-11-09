import unittest

import numpy as np
import pandas as pd

import analyze_2y_four_strategies_target_volatility as analysis
import analyze_best_four_futures_strategies as reference


class TwoYearIntegerRoundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[analysis.INSTRUMENT]

    def test_rounding_and_other_strategies(self):
        for period, start, end in (
            ("2016_2025", reference.START, reference.TRAIN_END),
            ("2026", reference.TEST_START, reference.TEST_END),
        ):
            with self.subTest(period=period):
                frames = reference.run_period(
                    self.prices.loc[:end], analysis.PARAMETERS, start, 1.0, 1.0, analysis.TARGET
                )
                for scaled, name in ((True, reference.STRATEGIES[0]), (False, reference.STRATEGIES[1])):
                    frame = frames[name]
                    eligible = frame["signal"].notna()
                    if scaled:
                        eligible &= frame["realized_volatility"].gt(0)
                        desired = (
                            frame.loc[eligible, "signal"] * analysis.TARGET
                            / frame.loc[eligible, "realized_volatility"]
                        )
                    else:
                        desired = frame.loc[eligible, "signal"]
                    np.testing.assert_allclose(frame.loc[eligible, "desired_position"], desired.round())
                for name, slug in zip(reference.STRATEGIES[2:], reference.SLUGS[2:]):
                    old = pd.read_csv(
                        analysis.OUTPUT / f"{period}_{slug}_daily.csv", parse_dates=["date"]
                    ).set_index("date")
                    self.assertEqual(list(frames[name].index), list(old.index))
                    np.testing.assert_allclose(frames[name]["position"], old["position"], atol=1e-9)
                    np.testing.assert_allclose(
                        frames[name]["strategy_return"], old["strategy_return"], atol=1e-9
                    )
                labels = reference.parameter_labels(analysis.PARAMETERS, 1.0, 1.0, analysis.TARGET)
                for name in reference.STRATEGIES[:2]:
                    self.assertIn("position rounded 1", labels[name])
                self.assertIn("vol target 1.5%", labels[reference.STRATEGIES[0]])


if __name__ == "__main__":
    unittest.main()
