import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_2y_four_strategies_target_volatility as two_year
import analyze_best_four_futures_strategies as comparison
import analyze_best_scaled_cdf_futures_strategies as configurations
import backtest_ewma_extreme_strategy as ewma
import backtest_kama_extreme_strategy as kama
import backtest_normal_cdf_signal as cdf


class TradingCostTests(unittest.TestCase):
    def test_turnover_reversal_and_year_boundary(self):
        index = pd.to_datetime([
            "2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05",
            "2026-01-06",
        ])
        for instrument, cost_constant in cdf.TRADING_COST_RATIOS.items():
            with self.subTest(instrument=instrument):
                frame = pd.DataFrame({
                    "position": [1.0, 1.0, -1.0, -0.5, 0.0],
                    "price": [100.0, 101.0, 102.0, 103.0, 104.0],
                    "daily_return": [0.01, 0.02, -0.01, 0.0, 0.01],
                }, index=index)
                cdf.apply_trading_cost(frame, instrument)
                changes = np.array([1.0, 0.0, 2.0, 0.5, 0.5])
                gross = np.array([0.01, 0.02, 0.01, 0.0, 0.0])
                previous_price = frame["price"].shift(1).fillna(frame["price"])
                expected_cost = changes * cost_constant / previous_price
                np.testing.assert_allclose(frame["position_change"], changes)
                np.testing.assert_allclose(frame["trading_cost"], expected_cost)
                np.testing.assert_allclose(frame["strategy_return"], gross - expected_cost)

    def test_backtests_deduct_costs_and_compound_net_returns(self):
        dates = pd.bdate_range("2025-11-03", periods=100)
        values = 100 + np.sin(np.arange(len(dates)) / 3) * 4 + np.arange(len(dates)) * 0.02
        start = dates[25]
        for instrument, cost_constant in cdf.TRADING_COST_RATIOS.items():
            prices = pd.Series(values, index=dates, name=instrument)
            for name, run, kwargs in (
                ("normal scaled", cdf.run_backtest, {
                    "lookback_days": 5, "volatility_lookback_days": 5,
                    "output_start_date": start, "signal_cutoff": 0,
                    "position_change_threshold_fraction": 0,
                }),
                ("normal unscaled", cdf.run_backtest, {
                    "lookback_days": 5, "volatility_lookback_days": 5,
                    "output_start_date": start, "signal_cutoff": 0,
                    "position_change_threshold_fraction": 0,
                    "use_volatility_scaling": False,
                }),
                ("EWMA", ewma.run_backtest, {
                    "ewma_span": 3, "extreme_lookback_days": 3, "start_date": start,
                }),
                ("KAMA", kama.run_backtest, {
                    "fast_n": 2, "slow_n": 3, "er_lookback_days": 3,
                    "extreme_lookback_days": 3, "start_date": start,
                }),
            ):
                with self.subTest(instrument=instrument, model=name):
                    net = run(prices, **kwargs)
                    gross = run(prices, trading_cost_ratio=0, **kwargs)
                    pd.testing.assert_series_equal(net["position"], gross["position"])
                    expected_change = net["position"].diff().fillna(net["position"]).abs()
                    self.assertGreater(expected_change.sum(), 0)
                    expected_cost = (
                        expected_change * cost_constant / net["price"].shift(1).fillna(net["price"])
                    )
                    np.testing.assert_allclose(net["trading_cost"], expected_cost)
                    np.testing.assert_allclose(net["gross_strategy_return"], gross["strategy_return"])
                    np.testing.assert_allclose(
                        net["strategy_return"], gross["strategy_return"] - expected_cost,
                    )
                    expected_wealth = (1 + net["strategy_return"]).cumprod()
                    np.testing.assert_allclose(net["wealth"], expected_wealth)
                    np.testing.assert_allclose(net["cumulative_pnl_pct"], (expected_wealth - 1) * 100)
                    if name.startswith("normal"):
                        yearly = net.groupby(net.index.year)["strategy_return"].transform(
                            lambda returns: (1 + returns).cumprod()
                        )
                        np.testing.assert_allclose(net["yearly_wealth"], yearly)

    def test_existing_position_at_fresh_period_start_is_charged(self):
        dates = pd.bdate_range("2025-12-01", periods=12)
        prices = pd.Series(np.arange(100.0, 112.0), index=dates, name="2Y future")
        for run, kwargs in (
            (ewma.run_backtest, {"ewma_span": 2, "extreme_lookback_days": 2}),
            (kama.run_backtest, {
                "fast_n": 2, "slow_n": 3, "er_lookback_days": 2,
                "extreme_lookback_days": 2,
            }),
        ):
            with self.subTest(model=run.__module__):
                frame = run(prices, start_date=dates[6], **kwargs)
                self.assertEqual(frame.loc[dates[6], "position"], 1)
                self.assertAlmostEqual(
                    frame.loc[dates[6], "trading_cost"],
                    cdf.TRADING_COST_RATIOS["2Y future"] / prices.loc[dates[5]],
                )
                self.assertEqual(frame.loc[dates[5], "trading_cost"], 0)

    def test_four_strategy_reports_use_net_returns_for_each_future(self):
        prices = cdf.load_single_future_prices(include_dbmf=False)
        configs = {
            **configurations.REQUESTED_CONFIGURATIONS,
            "2Y future": {"parameters": two_year.PARAMETERS, "rounding": two_year.ROUNDING},
        }
        for instrument, config in configs.items():
            frames = comparison.run_period(
                prices[instrument].loc[:comparison.TEST_END],
                config["parameters"], comparison.TEST_START,
                config["rounding"], config["rounding"],
            )
            for name, frame in frames.items():
                with self.subTest(instrument=instrument, model=name):
                    turnover = frame["position"].diff().fillna(frame["position"]).abs()
                    expected_cost = (
                        turnover
                        * cdf.TRADING_COST_RATIOS[instrument]
                        / prices[instrument].shift(1).reindex(frame.index)
                    )
                    np.testing.assert_allclose(
                        frame["trading_cost"], expected_cost,
                    )
                    expected = frame["gross_strategy_return"] - frame["trading_cost"]
                    np.testing.assert_allclose(frame["strategy_return"], expected)
                    expected_total = (np.prod(1 + expected.to_numpy(dtype=float)) - 1) * 100
                    self.assertAlmostEqual(comparison.statistics(frame)["total_return_pct"], expected_total)

    def test_saved_report_contains_costs_and_net_summary(self):
        instrument = "2Y future"
        prices = cdf.load_single_future_prices(include_dbmf=False)[instrument]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(comparison, "save_plots"):
                comparison.save_report(
                    instrument, prices.rename("unlabelled"), two_year.PARAMETERS, directory,
                    two_year.ROUNDING, two_year.ROUNDING,
                )
            summary = pd.read_csv(directory / "summary.csv")
            for name, slug in zip(comparison.STRATEGIES, comparison.SLUGS):
                with self.subTest(model=name):
                    daily = pd.read_csv(directory / f"2026_{slug}_daily.csv")
                    np.testing.assert_allclose(
                        daily["strategy_return"], daily["gross_strategy_return"] - daily["trading_cost"],
                        atol=2e-10,
                    )
                    expected_total = (
                        float(np.prod(1 + daily["strategy_return"].to_numpy(dtype=float))) - 1
                    ) * 100
                    row = summary.loc[summary["period"].eq("2026") & summary["strategy"].eq(name)].iloc[0]
                    self.assertAlmostEqual(row["total_return_pct"], expected_total, delta=0.00001)
                    self.assertAlmostEqual(row["position_turnover"], daily["position_change"].sum(), delta=1e-8)
                    self.assertAlmostEqual(row["trading_cost_accumulated"], daily["trading_cost"].sum(), delta=1e-8)

    def test_explicit_and_unrecognized_rates(self):
        frame = pd.DataFrame({
            "position": [1.0], "price": [100.0], "daily_return": [0.0],
        })
        cdf.apply_trading_cost(frame, "other")
        self.assertEqual(frame["trading_cost"].iloc[0], 0)
        cdf.apply_trading_cost(frame, "other", 0.01)
        self.assertEqual(frame["strategy_return"].iloc[0], -0.0001)
        for cost_constant in (-1, np.nan, np.inf):
            with self.subTest(cost_constant=cost_constant), self.assertRaises(ValueError):
                cdf.apply_trading_cost(frame, "2Y future", cost_constant)


if __name__ == "__main__":
    unittest.main()
