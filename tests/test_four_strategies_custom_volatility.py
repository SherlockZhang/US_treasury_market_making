import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as configs
import analyze_four_strategies_custom_volatility as analysis


class CustomVolatilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)

    def test_target_scaling_and_unchanged_strategies(self):
        for instrument, target in analysis.TARGETS.items():
            config = configs.REQUESTED_CONFIGURATIONS[instrument]
            rounding = config["rounding"]
            for start, end in ((reference.START, reference.TRAIN_END),
                               (reference.TEST_START, reference.TEST_END)):
                with self.subTest(instrument=instrument, start=start):
                    prices = self.prices[instrument].loc[:end]
                    frames = reference.run_period(
                        prices, config["parameters"], start, rounding, rounding, target
                    )
                    baseline = reference.run_period(
                        prices, config["parameters"], start, rounding, rounding
                    )
                    for name in reference.STRATEGIES[1:]:
                        pd.testing.assert_frame_equal(frames[name], baseline[name])
                    frame = frames[reference.STRATEGIES[0]]
                    eligible = frame["signal"].notna() & frame["realized_volatility"].gt(0)
                    expected = (
                        (frame.loc[eligible, "signal"] * target
                         / frame.loc[eligible, "realized_volatility"] / rounding)
                        .round().mul(rounding).round(1)
                    )
                    np.testing.assert_allclose(frame.loc[eligible, "desired_position"], expected)
                    self.assertLessEqual(frame.index.max(), end)
                    labels = reference.parameter_labels(config["parameters"], rounding, rounding, target)
                    self.assertIn(f"vol target {target * 100:g}%", labels[reference.STRATEGIES[0]])

    def test_threshold_uses_requested_target_and_invalid_target_rejected(self):
        prices = self.prices["5Y future"].loc[reference.START:reference.TRAIN_END]
        frame = reference.cdf.run_backtest(
            prices, lookback_days=64, position_change_threshold_fraction=0.25,
            target_annualized_volatility=0.03,
        )
        eligible = frame["position_change_threshold"].notna()
        np.testing.assert_allclose(
            frame.loc[eligible, "position_change_threshold"],
            0.25 * 0.03 / frame.loc[eligible, "realized_volatility"],
        )
        for target in (0, -0.01, np.nan, np.inf):
            with self.subTest(target=target), self.assertRaises(ValueError):
                reference.cdf.run_backtest(prices, target_annualized_volatility=target)


if __name__ == "__main__":
    unittest.main()
