import unittest

import numpy as np
import pandas as pd

import backtest_normal_cdf_signal as cdf


class InstrumentMarketProfileTests(unittest.TestCase):
    def test_profile_definitions(self):
        spec = cdf.MarketFactorSpec
        expected = {
            "2Y future": (
                spec("tff2glfn", "TFF2GLFN Index", 0.70, True, "change"),
                spec("cta", "CTA US Equity", 0.60, True, "return"),
                spec("spx", "SPX", 0.59, False, "return"),
                spec("tff2ilfn", "TFF2ILFN Index", 0.58, False, "change"),
                spec("cl1", "CL1 COMB Comdty", 0.56, False, "return"),
            ),
            "5Y future": (
                spec("tff2glfn", "TFF2GLFN Index", 0.71, True, "change"),
                spec("cta", "CTA US Equity", 0.57, True, "return"),
                spec("ukx", "UKX", 0.57, True, "return"),
                spec("cl1", "CL1 COMB Comdty", 0.54, False, "return"),
                spec("spx", "SPX", 0.56, False, "return"),
                spec("tff2ilfn", "TFF2ILFN Index", 0.58, False, "change"),
                spec("sofrate", "SOFRRATE Index", 0.68, False, "change"),
            ),
            "10Y future": (
                spec("tff2glfn", "TFF2GLFN Index", 0.71, True, "change"),
                spec("ukx", "UKX", 0.60, True, "return"),
                spec("cta", "CTA US Equity", 0.57, True, "return"),
                spec("sofrate", "SOFRRATE Index", 0.58, False, "change"),
                spec("tff2hlfn", "TFF2HLFN Index", 0.61, False, "change"),
            ),
        }
        self.assertEqual(cdf.INSTRUMENT_MARKET_FACTOR_PROFILES, expected)

    def test_profiles_apply_expected_multipliers(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        returns = np.where(np.arange(len(dates)) % 3 == 0, -0.001, 0.002)
        prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        baseline = cdf.run_backtest(
            prices, lookback_days=5, signal_cutoff=0,
            position_change_threshold_fraction=0, use_volatility_scaling=False,
        )
        self.assertTrue(baseline["signal"].dropna().gt(0).all())
        factor_data = pd.DataFrame(
            {
                "value": np.where(np.arange(len(dates)) % 2, 1.0, -1.0),
                "mean": 0.0,
                "std": 0.5,
                "outside_one_std": np.arange(len(dates)) % 3 != 0,
            },
            index=dates,
        )
        for instrument, specs in cdf.INSTRUMENT_MARKET_FACTOR_PROFILES.items():
            with self.subTest(instrument=instrument):
                result = cdf.run_backtest(
                    prices, lookback_days=5, signal_cutoff=0,
                    position_change_threshold_fraction=0, use_volatility_scaling=False,
                    instrument_factor_data=tuple((spec, factor_data) for spec in specs),
                )
                expected_combined = pd.Series(1.0, index=dates)
                for spec in specs:
                    same = baseline["signal"].mul(factor_data["value"]).gt(0)
                    adjustment = 2 * spec.probability - 1
                    same_multiplier = 1 + adjustment if spec.same_sign_increases else 1 - adjustment
                    different_multiplier = 1 - adjustment if spec.same_sign_increases else 1 + adjustment
                    expected = np.select(
                        [factor_data["outside_one_std"] & same,
                         factor_data["outside_one_std"] & ~same],
                        [same_multiplier, different_multiplier],
                        default=1.0,
                    )
                    np.testing.assert_allclose(result[f"{spec.name}_signal_multiplier"], expected)
                    expected_combined *= expected
                np.testing.assert_allclose(result["market_factor_signal_multiplier"], expected_combined)
                np.testing.assert_allclose(
                    result["signal"], baseline["signal"] * expected_combined,
                    equal_nan=True,
                )


if __name__ == "__main__":
    unittest.main()
