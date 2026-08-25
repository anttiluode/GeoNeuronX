import unittest

import numpy as np

from geoneuronx.dynamical import (
    delayed_xor_target,
    quadratic_feature_map,
    rectified_threshold_basis,
    ridge_readout_metrics,
    telegraph_signal,
)


class DynamicalNeuronHelperTests(unittest.TestCase):
    def test_telegraph_signal_shape(self):
        rng = np.random.default_rng(0)
        x = telegraph_signal(101, rng, hold_samples=4, noise_std=0.0)
        self.assertEqual(x.shape, (101,))
        self.assertTrue(np.all(np.isin(x, (-1.0, 1.0))))

    def test_delayed_xor_target(self):
        x = np.array((-1.0, -1.0, 1.0, 1.0, -1.0, 1.0))
        y = delayed_xor_target(x, start=2, lag=2)
        expected = np.array((1.0, 1.0, 1.0, -1.0))
        self.assertTrue(np.array_equal(y, expected))

    def test_rectified_threshold_basis_shape(self):
        x = np.arange(60, dtype=float).reshape(20, 3)
        features = rectified_threshold_basis(
            x,
            train_rows=10,
            thresholds=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(features.shape, (20, 9))
        self.assertTrue(np.all(features >= 0.0))

    def test_quadratic_feature_map_shape(self):
        x = np.arange(15, dtype=float).reshape(5, 3)
        features = quadratic_feature_map(x)
        # 3 linear terms + 3*4/2 quadratic terms.
        self.assertEqual(features.shape, (5, 9))

    def test_quadratic_features_solve_xor_while_linear_does_not(self):
        base = np.array(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ]
        )
        x = np.tile(base, (200, 1))
        y = np.where((x[:, 0] > 0) ^ (x[:, 1] > 0), 1.0, -1.0)

        linear = ridge_readout_metrics(x, y)
        quadratic = ridge_readout_metrics(quadratic_feature_map(x), y)
        self.assertLess(linear["accuracy"], 0.60)
        self.assertGreater(quadratic["accuracy"], 0.99)


if __name__ == "__main__":
    unittest.main()
