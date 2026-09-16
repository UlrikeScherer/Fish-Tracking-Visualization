import unittest

import numpy as np

from fishproviz.metrics.turning_angle.compute import compute_turning_angles, compute_turning_angle_streak_lengths


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


class TestComputeStreakLengths(unittest.TestCase):
    def test_all_same_sign(self):
        angles = np.array([0.1, 0.2, 0.3])
        result = compute_turning_angle_streak_lengths(angles)
        assert list(result) == [3.0]

    def test_sign_change_splits_streak(self):
        angles = np.array([0.1, 0.2, -0.1, -0.2])
        result = compute_turning_angle_streak_lengths(angles)
        assert sorted(result) == [2.0, 2.0]

    def test_nan_breaks_streak(self):
        angles = np.array([0.1, np.nan, 0.1])
        result = compute_turning_angle_streak_lengths(angles)
        assert sorted(result) == [1.0, 1.0]

    def test_zero_treated_as_negative(self):
        # 0.0 maps to -1 (not a NaN separator), so it forms its own negative run
        angles = np.array([0.1, 0.0, 0.1])
        result = compute_turning_angle_streak_lengths(angles)
        assert sorted(result) == [1.0, 1.0, 1.0]

    def test_empty(self):
        result = compute_turning_angle_streak_lengths(np.array([]))
        assert result.shape == (0,)


if __name__ == "__main__":
    unittest.main()
