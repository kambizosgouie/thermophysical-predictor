import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import correlation_figure, parity_figure


class PlotUtilsTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_correlation_includes_all_targets_in_order(self):
        data = pd.DataFrame({
            "input": [1, 2, 3, 4],
            "first": [2, 4, 6, 8],
            "second": [4, 3, 2, 1],
        })
        figure = correlation_figure(data, ["input"], ["second", "first"])

        for axis in figure.axes[:2]:
            self.assertEqual([label.get_text() for label in axis.get_yticklabels()],
                             ["second", "first"])
            np.testing.assert_allclose(axis.images[0].get_array(), [[-1], [1]])

    def test_correlation_with_one_target_and_constant_input(self):
        data = pd.DataFrame({"input": [1, 1, 1], "target": [1, 2, 3]})
        figure = correlation_figure(data, ["input"], ["target"])
        for axis in figure.axes[:2]:
            self.assertEqual(axis.images[0].get_array().shape, (1, 1))
            self.assertEqual(axis.texts[0].get_text(), "nan")

    def test_parity_distinguishes_train_and_test_in_full_plot(self):
        class Model:
            def predict(self, values):
                return values["input"].to_numpy() * 2

        x_train = pd.DataFrame({"input": [1, 2]})
        x_test = pd.DataFrame({"input": [3]})
        figure = parity_figure(
            x_train, x_test, pd.Series([2, 4]), pd.Series([6]), Model(), "target"
        )
        test_axis, full_axis = figure.axes
        self.assertEqual(len(test_axis.collections), 1)
        self.assertEqual(len(full_axis.collections), 2)
        self.assertEqual([c.get_label() for c in full_axis.collections], ["Train", "Test"])
        self.assertFalse(np.array_equal(
            full_axis.collections[0].get_facecolor(),
            full_axis.collections[1].get_facecolor(),
        ))
        np.testing.assert_allclose(full_axis.collections[0].get_offsets(), [[2, 2], [4, 4]])
        np.testing.assert_allclose(full_axis.collections[1].get_offsets(), [[6, 6]])
        np.testing.assert_allclose(test_axis.collections[0].get_facecolor(),
                                   full_axis.collections[1].get_facecolor())


if __name__ == "__main__":
    unittest.main()
