import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import analyze_10y_six_best_models_normal_grid as report
import analyze_best_four_futures_strategies as reference
import search_10y_cdf_rounding_0p5_daily_weekly as search


class TenYearFourCdfModesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = reference.cdf.load_single_future_prices(include_dbmf=False)[
            search.INSTRUMENT
        ]
        cls.grid = search.search(cls.prices)

    def test_grid_and_separate_mode_winners(self):
        grid = self.grid
        self.assertEqual(len(grid), 20)
        self.assertTrue(grid["position_rounding_increment"].eq(0.5).all())
        self.assertTrue(grid["signal_cutoff"].eq(0).all())
        self.assertTrue(grid["rebalance_threshold_fraction"].eq(0).all())
        for scaled, frequency in product((True, False), search.FREQUENCIES):
            mode = grid.loc[
                grid["volatility_scaling"].eq(scaled)
                & grid["rebalance_frequency"].eq(frequency)
            ]
            self.assertEqual(set(mode["signal_lookback_days"]), set(reference.CDF_LOOKBACKS))
            self.assertEqual(mode["sharpe_rank"].tolist(), list(range(1, 6)))
            self.assertTrue(mode["sharpe_ratio"].is_monotonic_decreasing)
            if scaled:
                np.testing.assert_allclose(mode["target_annualized_volatility_pct"], 5.0)
            else:
                self.assertTrue(mode["target_annualized_volatility_pct"].isna().all())

        with tempfile.TemporaryDirectory() as directory:
            grid_file = Path(directory) / "grid.csv"
            grid.to_csv(grid_file, index=False)
            with patch.object(report, "GRID_FILE", grid_file):
                winners = report.selected_cdf_models()
        self.assertEqual(len(winners), 4)
        self.assertEqual(
            set(zip(winners["volatility_scaling"], winners["rebalance_frequency"])),
            set(product((True, False), search.FREQUENCIES)),
        )
        for _, row in winners.iterrows():
            mode = grid.loc[
                grid["volatility_scaling"].eq(row["volatility_scaling"])
                & grid["rebalance_frequency"].eq(row["rebalance_frequency"])
            ]
            self.assertEqual(row["sharpe_ratio"], mode["sharpe_ratio"].max())
            frame = report.backtest_cdf(
                self.prices, row, reference.START, reference.TRAIN_END
            )
            self.assertAlmostEqual(
                reference.statistics(frame)["sharpe_ratio"], row["sharpe_ratio"]
            )

    def test_grid_ignores_out_of_sample_prices(self):
        prices = self.prices.copy()
        prices.loc[reference.TEST_START:] *= 2
        pd.testing.assert_frame_equal(self.grid, search.search(prices))


class TenYearPlotTests(unittest.TestCase):
    def test_position_bars_are_contiguous_across_weekends(self):
        frame = pd.DataFrame(
            {"position": [1.0, -0.5, 0.0], "price": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        fig, ax = plt.subplots()
        try:
            report.plot_price_and_position(ax, frame)
            bars = ax.containers[0].patches
            self.assertEqual(len(bars), len(frame))
            self.assertEqual([bar.get_width() for bar in bars], [3.0, 1.0, 1.0])
            for left, right in zip(bars, bars[1:]):
                self.assertAlmostEqual(left.get_x() + left.get_width(), right.get_x())
            self.assertEqual([bar.get_height() for bar in bars], [1.0, -0.5, 0.0])
        finally:
            plt.close(fig)

    def test_line_colors_distinguish_all_six_models(self):
        modes = [
            "Normal CDF scaled daily T=64",
            "Normal CDF scaled weekly T=64",
            "Normal CDF unscaled daily T=64",
            "Normal CDF unscaled weekly T=64",
            "EWMA",
            "KAMA",
        ]
        colors = [to_rgb(report.model_color(name)) for name in modes]
        self.assertEqual(len(set(colors)), 6)
        for red, green, blue in colors[:2]:
            self.assertGreater(blue, red)
        for red, green, blue in colors[2:4]:
            self.assertGreater(red, blue)
            self.assertGreater(red, green)
        for red, green, blue in colors[4:]:
            self.assertGreater(green, red)
            self.assertGreater(green, blue)


if __name__ == "__main__":
    unittest.main()
