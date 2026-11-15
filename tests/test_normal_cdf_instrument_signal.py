import unittest
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import norm

import backtest_normal_cdf_signal as cdf


class InstrumentSignalTests(unittest.TestCase):
    def test_formula_and_cutoff(self):
        returns = pd.Series([0.01, 0.02, 0.015, -0.003, -0.01, -0.02, -0.015])
        for instrument in ("2Y future", "GC", "5Y future", "10Y future", None):
            with self.subTest(instrument=instrument):
                result = cdf.calculate_normal_cdf_signal(
                    returns, 3, 0.5, instrument=instrument
                )
                probability = norm.cdf(result["t_stat"])
                if instrument == "2Y future":
                    expected = (8 / 3) * probability - 4 / 3
                elif instrument == "GC":
                    expected = 2 * probability - 1
                else:
                    expected = 4 * probability - 2
                np.testing.assert_allclose(
                    result["unfiltered_signal"], expected, equal_nan=True
                )
                filtered = np.where(np.abs(expected) > 0.5, expected, 0.0)
                filtered[np.isnan(expected)] = np.nan
                np.testing.assert_allclose(result["signal"], filtered, equal_nan=True)
                if instrument == "GC":
                    self.assertLessEqual(result["unfiltered_signal"].abs().max(), 1)
                else:
                    self.assertGreater(result["unfiltered_signal"].max(), 1)
                    self.assertLess(result["unfiltered_signal"].min(), -1)

    def test_constant_returns_remain_undefined(self):
        result = cdf.calculate_normal_cdf_signal(
            pd.Series([0.0] * 8), 3, instrument="2Y future"
        )
        self.assertTrue(result["unfiltered_signal"].isna().all())
        self.assertTrue(result["signal"].isna().all())

    def test_backtest_single_rounded_and_average_modes(self):
        dates = pd.bdate_range("2018-01-01", periods=520)
        returns = 0.0002 + 0.001 * np.sin(np.arange(len(dates)) / 5)
        prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        for mode in ("single", "rounded", "average"):
            for instrument in ("2Y future", "GC", "5Y future", "10Y future", None):
                with self.subTest(mode=mode, instrument=instrument):
                    result = cdf.run_backtest(
                        prices.rename(instrument),
                        lookback_days=32,
                        signal_cutoff=0,
                        signal_mode=mode,
                        position_change_threshold_fraction=0,
                        use_volatility_scaling=False,
                        trading_cost_ratio=0,
                    )
                    expected = cdf.calculate_normal_cdf_signal(
                        result["daily_return"], 32, 0, instrument=instrument
                    )
                    np.testing.assert_allclose(
                        result["unfiltered_signal"],
                        expected["unfiltered_signal"],
                        equal_nan=True,
                    )
                    if mode == "average":
                        components = []
                        for lookback in cdf.AVERAGED_SIGNAL_LOOKBACK_DAYS:
                            component = cdf.calculate_normal_cdf_signal(
                                result["daily_return"], lookback, 0,
                                instrument=instrument,
                            )["signal"]
                            np.testing.assert_allclose(
                                result[f"signal_t{lookback}"], component, equal_nan=True
                            )
                            components.append(component)
                        expected_signal = pd.concat(components, axis=1).mean(
                            axis=1, skipna=False
                        )
                    else:
                        expected_signal = expected["signal"]
                    np.testing.assert_allclose(
                        result["signal"], expected_signal, equal_nan=True
                    )
                    if mode == "rounded":
                        desired = (
                            (expected_signal / 0.1).round().mul(0.1).round(1)
                        )
                        np.testing.assert_allclose(
                            result["desired_position"], desired, equal_nan=True
                        )

    def test_rounded_average_is_opt_in_for_scaled_and_unscaled(self):
        dates = pd.bdate_range("2018-01-01", periods=600)
        returns = 0.0002 + 0.001 * np.sin(np.arange(len(dates)) / 5)
        prices = pd.Series(100 * np.cumprod(1 + returns), index=dates, name="GC")
        for scaled in (True, False):
            for increment in (0.25, 0.5):
                with self.subTest(scaled=scaled, increment=increment):
                    backtest = partial(
                        cdf.run_backtest, prices,
                        lookback_days=64, signal_cutoff=0, signal_mode="average",
                        position_change_threshold_fraction=0,
                        use_volatility_scaling=scaled, position_rounding_increment=increment,
                        target_annualized_volatility=0.12, trading_cost_ratio=0,
                    )
                    original = backtest()
                    rounded = backtest(round_average_positions=True)
                    np.testing.assert_allclose(
                        rounded["signal"], original["signal"], equal_nan=True
                    )
                    expected = (
                        (original["desired_position"] / increment)
                        .round()
                        .mul(increment)
                        .round(10)
                    )
                    np.testing.assert_allclose(
                        rounded["desired_position"], expected, equal_nan=True
                    )
                    positions = rounded["desired_position"].dropna().to_numpy()
                    np.testing.assert_allclose(
                        positions / increment,
                        np.round(positions / increment),
                        atol=1e-10,
                    )
                    self.assertTrue(
                        rounded["desired_position"].dropna().ne(
                            original["desired_position"].dropna()
                        ).any()
                    )
        with self.assertRaises(ValueError):
            cdf.run_backtest(prices, signal_mode="rounded", round_average_positions=True)


if __name__ == "__main__":
    unittest.main()
