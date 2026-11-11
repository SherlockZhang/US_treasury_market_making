import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_best_four_futures_strategies as reference


class PricePositionPlotTests(unittest.TestCase):
    def test_signed_bar_heights_and_separate_price_axis(self):
        frame = pd.DataFrame(
            {"position": [0.0, 0.5, 2.0, -1.0, -2.0], "price": [100, 101, 102, 101, 100]},
            index=pd.bdate_range("2026-01-05", periods=5),
        )
        fig, ax = plt.subplots()
        try:
            reference.plot_price_and_position(ax, frame)
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(ax.get_ylim(), (-2.0, 2.0))
            self.assertEqual(ax.get_ylabel(), "Position (units)")
            self.assertEqual(fig.axes[1].get_ylabel(), "Futures price")
            self.assertEqual([bar.get_height() for bar in ax.patches], frame["position"].tolist())
            for bar, position in zip(ax.patches, frame["position"]):
                self.assertEqual(bar.get_y(), 0)
                self.assertAlmostEqual(bar.get_width(), 1)
                if position > 0:
                    self.assertGreater(bar.get_facecolor()[1], bar.get_facecolor()[0])
                elif position < 0:
                    self.assertGreater(bar.get_facecolor()[0], bar.get_facecolor()[1])
            np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), frame["price"])
        finally:
            plt.close(fig)

    def test_flat_and_invalid_positions(self):
        frame = pd.DataFrame(
            {"position": [0.0, 0.0], "price": [100, 101]},
            index=pd.date_range("2026-01-01", periods=2),
        )
        fig, ax = plt.subplots()
        try:
            reference.plot_price_and_position(ax, frame)
            self.assertEqual(ax.get_ylim(), (-1, 1))
            self.assertTrue(all(bar.get_height() == 0 for bar in ax.patches))
        finally:
            plt.close(fig)
        frame.iloc[0, frame.columns.get_loc("position")] = np.nan
        fig, ax = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "finite position"):
                reference.plot_price_and_position(ax, frame)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
