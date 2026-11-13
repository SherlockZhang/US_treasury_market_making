import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_rank_sum_scaled_cdf_market_factors as analysis


class RankSumScaledMarketFactorTests(unittest.TestCase):
    def test_report_and_latest_baselines(self):
        expected = {
            ("2Y future", "daily"): 252,
            ("2Y future", "weekly"): 64,
            ("5Y future", "daily"): 252,
            ("5Y future", "weekly"): 252,
            ("10Y future", "daily"): 64,
            ("10Y future", "weekly"): 64,
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            analysis.save_report(output)
            summary = pd.read_csv(output / "summary.csv")
            winners = pd.read_csv(output / "cdf_best_parameters.csv")
            coverage = pd.read_csv(output / "market_factor_data_coverage.csv")
            self.assertEqual(len(summary), 12)
            self.assertEqual(len(winners), 6)
            self.assertEqual(len(coverage), 17)
            for (instrument, frequency), lookback in expected.items():
                with self.subTest(instrument=instrument, frequency=frequency):
                    rows = summary.loc[
                        summary["instrument"].eq(instrument)
                        & summary["rebalance_frequency"].eq(frequency)
                    ]
                    self.assertEqual(set(rows["market_factor_mode"]), {"none", "instrument"})
                    self.assertTrue(rows["signal_lookback_days"].eq(lookback).all())
                    stem = instrument.split()[0].lower()
                    with_factor = pd.read_csv(
                        output / frequency / f"{stem}_with_market_factors_daily.csv"
                    )
                    np.testing.assert_allclose(
                        with_factor["signal"],
                        with_factor["raw_signal"] * with_factor["market_factor_signal_multiplier"],
                        rtol=0, atol=1e-9, equal_nan=True,
                    )
                    self.assertTrue(with_factor["market_factor_signal_multiplier"].ne(1).any())
                    if instrument == "2Y future":
                        self.assertLessEqual(with_factor["position"].abs().max(), 3)
            for frequency in analysis.FREQUENCIES:
                self.assertEqual(len(list((output / frequency).glob("*_daily.csv"))), 6)
                self.assertEqual(len(list((output / frequency).glob("*.png"))), 6)


if __name__ == "__main__":
    unittest.main()
