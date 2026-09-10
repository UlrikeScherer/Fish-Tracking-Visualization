import unittest

import numpy as np

from fishproviz.metrics.turning_angle.compute import compute_turning_angles


class TestComputeTurningAngles(unittest.TestCase):
    def test_basic_angles(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        expected = np.array([np.pi * (3 / 4), np.pi / 2, np.pi / 2])
        assert np.allclose(compute_turning_angles(points), expected)

    def test_zero_displacement_gets_fill(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        expected = np.array([0, 0, np.pi / 2, np.pi / 2])
        assert np.allclose(compute_turning_angles(points), expected)

    def test_nan_position_propagates(self):
        points = np.array([[0, 0], [1, 0], [np.nan, 0], [1, 0], [0, 1]], dtype=float)
        expected = np.array([0, 0, 0])
        assert np.allclose(compute_turning_angles(points), expected)

    def test_inf_position_propagates(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [np.inf, np.inf], [0, 1]], dtype=float)
        expected = np.array([0, 0, 0])
        assert np.allclose(compute_turning_angles(points), expected)

    def test_fill_value_nan(self):
        points = np.array([[0, 0], [1, 0], [np.nan, 0], [1, 0], [0, 1]], dtype=float)
        result = compute_turning_angles(points, fill_value=np.nan)
        assert np.all(np.isnan(result))

    def test_two_points_returns_empty(self):
        points = np.array([[0, 0], [1, 0]], dtype=float)
        result = compute_turning_angles(points)
        assert result.shape == (0,)


if __name__ == "__main__":
    unittest.main()
