import unittest

import numpy as np
import pandas as pd

import analyze_best_cdf_market_factor_comparison as comparison
import analyze_best_four_futures_strategies as reference
import analyze_custom_vol_cdf_market_factor_comparison as custom


class CustomVolMarketFactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)

    def test_reference_and_factor_sizing(self):
        for instrument, (lookback, target, rounding, prior_folder) in custom.SETTINGS.items():
            with self.subTest(instrument=instrument):
                frames = comparison.backtest_pair(
                    instrument, self.prices[instrument], lookback, target, rounding
                )
                previous = pd.read_csv(
                    reference.RESULTS / prior_folder / "2026_cdf_scaled_daily.csv",
                    parse_dates=["date"],
                ).set_index("date")
                self.assertEqual(list(frames["none"].index), list(previous.index))
                for column in ("position", "strategy_return"):
                    actual = "gross_strategy_return" if column == "strategy_return" else column
                    np.testing.assert_allclose(frames["none"][actual], previous[column], atol=1e-9)
                frame = frames["instrument"]
                eligible = frame["signal"].notna() & frame["realized_volatility"].gt(0)
                expected = (
                    frame.loc[eligible, "signal"] * target
                    / frame.loc[eligible, "realized_volatility"] / rounding
                ).round().mul(rounding).round(1)
                np.testing.assert_allclose(frame.loc[eligible, "desired_position"], expected)
                np.testing.assert_allclose(
                    frame.loc[eligible, "signal"],
                    frame.loc[eligible, "raw_signal"]
                    * frame.loc[eligible, "market_factor_signal_multiplier"],
                )
                self.assertTrue(frame["market_factor_signal_multiplier"].ne(1).any())
                summary = comparison.summarize(
                    instrument, "instrument", lookback, frame, target, rounding, "retained"
                )
                self.assertAlmostEqual(summary["target_annualized_volatility_pct"], target * 100)
                self.assertEqual(summary["position_rounding_increment"], rounding)
                self.assertEqual(summary["lookback_selection"], "retained")


if __name__ == "__main__":
    unittest.main()
