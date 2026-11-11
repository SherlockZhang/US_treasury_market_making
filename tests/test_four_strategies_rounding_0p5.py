import unittest

import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference
import analyze_best_scaled_cdf_futures_strategies as configs
import analyze_four_strategies_custom_volatility as analysis


class CdfHalfRoundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)

    def test_rounding_both_models_and_unchanged_ewma_kama(self):
        for instrument, target in analysis.TARGETS.items():
            config = configs.REQUESTED_CONFIGURATIONS[instrument]
            for period, start, end in (
                ("2016_2025", reference.START, reference.TRAIN_END),
                ("2026", reference.TEST_START, reference.TEST_END),
            ):
                with self.subTest(instrument=instrument, period=period):
                    frames = reference.run_period(
                        self.prices[instrument].loc[:end], config["parameters"],
                        start, 0.5, 0.5, target,
                    )
                    for scaled, name in ((True, reference.STRATEGIES[0]),
                                         (False, reference.STRATEGIES[1])):
                        frame = frames[name]
                        eligible = frame["signal"].notna()
                        if scaled:
                            eligible &= frame["realized_volatility"].gt(0)
                            desired = (
                                frame.loc[eligible, "signal"] * target
                                / frame.loc[eligible, "realized_volatility"]
                            )
                        else:
                            desired = frame.loc[eligible, "signal"]
                        np.testing.assert_allclose(
                            frame.loc[eligible, "desired_position"],
                            (desired / 0.5).round().mul(0.5).round(1),
                        )
                    stem = instrument.split()[0].lower()
                    tag = str(config["rounding"]).replace(".", "p")
                    root = reference.RESULTS / (
                        f"{stem}_four_strategies_cdf_rounded_{tag}_target_vol_"
                        f"{round(target * 100)}pct_2016_2025_and_2026"
                    )
                    for name, slug in zip(reference.STRATEGIES[2:], reference.SLUGS[2:]):
                        original = pd.read_csv(
                            root / f"{period}_{slug}_daily.csv", parse_dates=["date"]
                        ).set_index("date")
                        self.assertEqual(list(frames[name].index), list(original.index))
                        np.testing.assert_allclose(frames[name]["position"], original["position"], atol=1e-9)
                        np.testing.assert_allclose(
                            frames[name]["gross_strategy_return"], original["strategy_return"], atol=1e-9
                        )
                    labels = reference.parameter_labels(config["parameters"], 0.5, 0.5, target)
                    self.assertIn(f"vol target {target * 100:g}%", labels[reference.STRATEGIES[0]])
                    self.assertTrue(all("position rounded 0.5" in labels[n] for n in reference.STRATEGIES[:2]))


if __name__ == "__main__":
    unittest.main()
