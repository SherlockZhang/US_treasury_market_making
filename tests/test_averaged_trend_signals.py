import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import averaged_trend_signals as trends
import analyze_2y_all_rank_sum_six_models as two_year_report
import analyze_2y_four_cdf_modes_cap3 as report
import analyze_5y_10y_all_rank_sum_six_models as other_reports
import backtest_ewma_extreme_strategy as ewma
import backtest_kama_extreme_strategy as kama
import backtest_normal_cdf_signal as cdf


class AveragedTrendTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2025-09-01", periods=130)
        t = np.arange(len(dates))
        self.prices = {
            "2Y future": pd.Series(100 + t * 0.01 + np.sin(t / 4), index=dates, name="2Y future"),
            "5Y future": pd.Series(120 - t * 0.01 + np.sin(t / 7), index=dates, name="5Y future"),
            "10Y future": pd.Series(140 + np.cos(t / 5), index=dates, name="10Y future"),
        }

    def test_exact_model_labels(self):
        expected = {
            ("2Y future", "EWMA"): "EWMA(5,40)@2y + EWMA(10,30)@2y",
            ("5Y future", "EWMA"): "EWMA(5,40)@2y + EWMA(20,20)@10y",
            ("10Y future", "EWMA"): "EWMA(25,15)@5y + EWMA(15,20)@10y",
            ("2Y future", "KAMA"): "KAMA(30,5,30,3)@2y + KAMA(30,5,30,3)@5y",
            ("5Y future", "KAMA"): "KAMA(10,5,30,3)@10y + KAMA(30,5,30,3)@2y",
            ("10Y future", "KAMA"): "KAMA(10,3,30,3)@2y + KAMA(5,5,30,3)@5y",
        }
        for (instrument, model), label in expected.items():
            with self.subTest(instrument=instrument, model=model):
                self.assertEqual(trends.model_label(instrument, model, 3), label + "; position size=3")
                row = trends.parameter_rows(instrument, model, 3).iloc[0]
                self.assertEqual(row["selection_source"], "user specified")
                self.assertEqual(row["component_1"] + " + " + row["component_2"], label)

    def test_entry_state_machine_and_period_reset(self):
        dates = pd.bdate_range("2025-12-29", periods=11)
        signal = pd.Series([0.5, 1, -0.5, -1, 0, 0.5, 0, -0.5, 1, 0, 0], index=dates)
        position = trends.positions_from_signals(signal, dates[1], 3)
        self.assertEqual(position.tolist(), [0, 1.5, 1.5, 0, -3, -3, 0, 0, -1.5, 0, 0])
        reset = trends.positions_from_signals(signal, dates[3], 3)
        self.assertEqual(reset.iloc[:4].tolist(), [0, 0, 0, -1.5])

    def test_component_signals_target_returns_and_costs(self):
        start = self.prices["2Y future"].index[75]
        end = self.prices["2Y future"].index[-1]
        history_start = self.prices["2Y future"].index[0]
        for instrument in trends.EWMA_MODELS:
            for model, models in (("EWMA", trends.EWMA_MODELS), ("KAMA", trends.KAMA_MODELS)):
                with self.subTest(instrument=instrument, model=model):
                    actual = trends.run_backtest(self.prices, instrument, model, start, end, history_start)
                    for number, component in enumerate(models[instrument], 1):
                        source = self.prices[component.instrument]
                        if model == "EWMA":
                            standalone = ewma.run_backtest(source, component.n, component.m)
                        else:
                            standalone = kama.run_backtest(
                                source, component.fast, component.slow, component.er, component.m
                            )
                        expected = standalone["long_entry_signal"].astype(int) - standalone["short_entry_signal"].astype(int)
                        pd.testing.assert_series_equal(
                            actual[f"component_{number}_signal"], expected.loc[start:], check_names=False
                        )
                    pd.testing.assert_series_equal(
                        actual["entry_signal"],
                        (actual["component_1_signal"] + actual["component_2_signal"]) / 2,
                        check_names=False,
                    )
                    pd.testing.assert_series_equal(actual["price"], self.prices[instrument].loc[start:], check_names=False)
                    expected_gross = actual["position"] * self.prices[instrument].pct_change().loc[start:]
                    np.testing.assert_allclose(actual["gross_strategy_return"], expected_gross)
                    previous_price = self.prices[instrument].shift().loc[start:]
                    expected_cost = actual["position_change"] * cdf.TRADING_COST_RATIOS[instrument] / previous_price
                    np.testing.assert_allclose(actual["trading_cost"], expected_cost)
                    self.assertTrue(set(actual["position"]).issubset({-3, -1.5, 0, 1.5, 3}))

    def test_rejects_misaligned_dates_and_invalid_target(self):
        start = self.prices["2Y future"].index[0]
        end = self.prices["2Y future"].index[-1]
        self.prices["10Y future"] = self.prices["10Y future"].iloc[1:]
        with self.assertRaisesRegex(ValueError, "price dates must match"):
            trends.run_backtest(self.prices, "5Y future", "EWMA", start, end, start)
        self.prices["5Y future"].iloc[10] = np.nan
        with self.assertRaisesRegex(ValueError, "finite, non-missing"):
            trends.run_backtest(self.prices, "5Y future", "EWMA", start, end, start)

    def test_six_model_report_entry_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {instrument: root / "averaged" / instrument.split()[0] for instrument in other_reports.OUTPUTS}
            outputs["2Y future"] = root / "averaged" / "2Y"
            trained_outputs = {instrument: root / "trained" / instrument.split()[0] for instrument in outputs}
            with (
                patch.object(two_year_report, "OUTPUT", trained_outputs["2Y future"]),
                patch.object(two_year_report, "AVERAGED_OUTPUT", outputs["2Y future"]),
                patch.object(other_reports, "OUTPUTS", {k: v for k, v in trained_outputs.items() if k != "2Y future"}),
                patch.object(other_reports, "AVERAGED_OUTPUTS", {k: v for k, v in outputs.items() if k != "2Y future"}),
                patch.object(report.charts, "save_plots"),
            ):
                two_year_report.main()
                other_reports.main()
                two_year_report.main("averaged")
                other_reports.main("averaged")
            for instrument, output in outputs.items():
                with self.subTest(instrument=instrument):
                    summary = pd.read_csv(output / "summary.csv")
                    trained_output = trained_outputs[instrument]
                    trained_summary = pd.read_csv(trained_output / "summary.csv")
                    self.assertEqual(len(summary), 12)
                    self.assertEqual(len(trained_summary), 12)
                    pd.testing.assert_frame_equal(
                        summary.loc[~summary["strategy"].isin(["EWMA", "KAMA"])].reset_index(drop=True),
                        trained_summary.loc[~trained_summary["strategy"].isin(["EWMA", "KAMA"])].reset_index(drop=True),
                    )
                    pd.testing.assert_frame_equal(
                        pd.read_csv(output / "cdf_best_parameters.csv"),
                        pd.read_csv(trained_output / "cdf_best_parameters.csv"),
                    )
                    for model in ("EWMA", "KAMA"):
                        rows = summary.loc[summary["strategy"].eq(model)]
                        self.assertEqual(len(rows), 2)
                        self.assertTrue(rows["parameters"].eq(trends.model_label(instrument, model, 3)).all())
                        parameters = pd.read_csv(output / f"{model.lower()}_best_parameters.csv")
                        self.assertEqual(parameters.loc[0, "selection_source"], "user specified")
                        trained_parameters = pd.read_csv(trained_output / f"{model.lower()}_best_parameters.csv").iloc[0]
                        winner = two_year_report.selected_trend_winner(model, instrument)
                        self.assertTrue(trained_parameters["selected_best"])
                        self.assertEqual(trained_parameters["rank_sum"], winner["rank_sum"])
                        parameter_names = (
                            ("n", "m") if model == "EWMA"
                            else ("alpha_slow_n", "alpha_fast_n", "er_lookback_days", "kama_extreme_window_m")
                        )
                        for name in parameter_names:
                            self.assertEqual(trained_parameters[name], winner[name])
                        for period in ("2016_2025", "2026"):
                            daily = pd.read_csv(output / f"{period}_{model.lower()}_daily.csv")
                            trained_daily = pd.read_csv(trained_output / f"{period}_{model.lower()}_daily.csv")
                            self.assertNotIn("component_1_signal", trained_daily.columns)
                            self.assertTrue(set(trained_daily["position"]).issubset({-3, 0, 3}))
                            np.testing.assert_allclose(
                                daily["entry_signal"],
                                (daily["component_1_signal"] + daily["component_2_signal"]) / 2,
                            )
                            self.assertTrue(set(daily["position"]).issubset({-3, -1.5, 0, 1.5, 3}))


if __name__ == "__main__":
    unittest.main()
