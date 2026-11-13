import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_10y_six_models_scaled_market_factors as analysis


class TenYearSixModelsFactorTests(unittest.TestCase):
    def test_out_of_sample_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            analysis.save_report(output)
            summary = pd.read_csv(output / "summary.csv")
            self.assertEqual(len(summary), 6)
            self.assertEqual(summary["market_factor_mode"].value_counts().to_dict(), {
                "none": 4, "instrument": 2,
            })
            self.assertEqual(len(list(output.glob("2026_*_daily.csv"))), 6)
            self.assertEqual(len(list(output.glob("2026_*.png"))), 3)
            self.assertEqual(len(pd.read_csv(output / "market_factor_data_coverage.csv")), 5)
            for frequency in ("daily", "weekly"):
                name = f"Normal CDF scaled {frequency} T=64"
                row = summary.loc[summary["strategy"].eq(name)].iloc[0]
                self.assertEqual(row["market_factor_mode"], "instrument")
                frame = pd.read_csv(output / f"2026_normal_cdf_scaled_{frequency}_t64_daily.csv")
                np.testing.assert_allclose(
                    frame["signal"], frame["raw_signal"] * frame["market_factor_signal_multiplier"],
                    rtol=0, atol=1e-9, equal_nan=True,
                )
                self.assertTrue(frame["market_factor_signal_multiplier"].ne(1).any())
            expected = {
                "Normal CDF unscaled daily T=32", "Normal CDF unscaled weekly T=64", "EWMA", "KAMA",
            }
            self.assertEqual(set(summary.loc[summary["market_factor_mode"].eq("none"), "strategy"]), expected)


if __name__ == "__main__":
    unittest.main()
