import unittest

import numpy as np
from numpy.typing import NDArray

from fishproviz.metrics.turning_angle.compute import compute_turning_angles, compute_turning_angle_streak_lengths, score_reversals_as_zero


class TestComputeTurningAngles(unittest.TestCase):
    def test_basic_angles(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        expected = np.array([np.pi * (3 / 4), np.pi / 2, np.pi / 2])
        assert np.allclose(compute_turning_angles(points), expected)

    def test_zero_displacement_gets_fill(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        expected = np.array([0, 0, np.pi / 2, np.pi / 2])
        assert np.allclose(compute_turning_angles(points), expected)

    def test_nan_position_gives_nan_regardless_of_fill(self):
        points = np.array([[0, 0], [1, 0], [np.nan, 0], [1, 0], [0, 1]], dtype=float)
        assert np.all(np.isnan(compute_turning_angles(points, fill_value=0.0)))

    def test_nan_position_only_touches_its_triplets(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [np.nan, 0], [2, 1], [2, 2], [3, 2]], dtype=float)
        result = compute_turning_angles(points, fill_value=0.0)
        assert np.isnan(result[1:4]).all()
        assert np.allclose(result[[0, 4]], [0.0, -np.pi / 2])

    def test_inf_position_gives_nan_regardless_of_fill(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [np.inf, np.inf], [0, 1]], dtype=float)
        result = compute_turning_angles(points, fill_value=0.0)
        assert np.isnan(result[1:]).all()
        assert result[0] == 0.0

    def test_zero_length_step_keeps_fill_value(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [1, 1]], dtype=float)
        assert np.array_equal(compute_turning_angles(points, fill_value=0.0), [0.0, 0.0])

    def test_fill_value_nan_applies_to_zero_length_and_gaps(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [np.nan, 0], [1, 0], [0, 1]], dtype=float)
        assert np.all(np.isnan(compute_turning_angles(points, fill_value=np.nan)))

    def test_stationary_mask_gets_fill(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [3, 1]], dtype=float)
        stationary = np.array([False, False, True, False])
        result = compute_turning_angles(points, fill_value=np.nan, stationary=stationary)
        assert result[0] == 0.0
        assert np.isnan(result[1:]).all()

    def test_stationary_mask_does_not_override_gap_nan(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [np.nan, 0], [4, 0]], dtype=float)
        result = compute_turning_angles(points, fill_value=0.0, stationary=np.array([True, True, True, True]))
        assert result[0] == 0.0
        assert np.isnan(result[1:]).all()

    def test_two_points_returns_empty(self):
        points = np.array([[0, 0], [1, 0]], dtype=float)
        result = compute_turning_angles(points)
        assert result.shape == (0,)


def _px_to_cm(px: NDArray[np.float64], origin: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    """Same form as transformation.pixel_to_cm: (px - origin) @ R(pi/4) * scale."""
    t = np.pi / 4
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (px - origin) @ rot * scale


class TestReversals(unittest.TestCase):
    """Exact reversals have no turn side and always return the canonical +pi."""

    def test_axis_and_diagonal_reversals_return_plus_pi(self):
        cases = {
            "forward_back": [[0, 0], [1, 0], [0, 0]],
            "back_forward": [[0, 0], [-1, 0], [0, 0]],
            "vertical": [[5, 5], [5, 9], [5, 7]],
            "diagonal": [[0, 0], [2, 3], [-2, -3]],
            "unequal_lengths": [[0, 0], [1, 0], [-1, 0]],
        }
        for name, pts in cases.items():
            with self.subTest(case=name):
                result = compute_turning_angles(np.array(pts, dtype=float))
                self.assertEqual(result[0], np.pi)

    def test_reversals_after_cm_transform_are_always_plus_pi(self):
        rng = np.random.default_rng(42)
        n = 5000
        for scale in (0.0227, 0.0194, 0.0311):
            with self.subTest(scale=scale):
                origin = rng.integers(0, 1500, size=2).astype(float)
                p0 = rng.integers(0, 1500, size=(n, 2))
                d = rng.integers(-20, 21, size=(n, 2))
                d[(d == 0).all(axis=1)] = [1, 0]
                a = rng.integers(1, 6, size=(n, 1))
                b = rng.integers(1, 6, size=(n, 1))
                p1 = p0 + a * d
                p2 = p1 - b * d
                angles = np.array([compute_turning_angles(_px_to_cm(np.array([q0, q1, q2], dtype=float), origin, scale))[0] for q0, q1, q2 in zip(p0, p1, p2)])
                self.assertFalse((angles == -np.pi).any())
                self.assertTrue((angles == np.pi).all())

    def test_near_reversal_keeps_geometric_sign(self):
        # (3,0) then (-3,1): cross = +3, so positive (a right turn on y-down pixels)
        px = np.array([[0, 0], [3, 0], [0, 1]], dtype=float)
        for points in (px, _px_to_cm(px, np.array([700.0, 400.0]), 0.0227)):
            with self.subTest(points=points.tolist()):
                angle = compute_turning_angles(points)[0]
                self.assertGreater(angle, 0.0)
                self.assertLess(angle, np.pi)
        mirrored = np.array([[0, 0], [3, 0], [0, -1]], dtype=float)
        self.assertLess(compute_turning_angles(mirrored)[0], 0.0)

    def test_gap_and_fill_rules_unchanged_around_reversal(self):
        points = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [np.nan, 0]], dtype=float)
        result = compute_turning_angles(points, fill_value=0.0)
        self.assertEqual(result[0], np.pi)
        self.assertEqual(result[1], 0.0)
        self.assertTrue(np.isnan(result[2]))


class TestScoreReversalsAsZero(unittest.TestCase):
    def test_maps_pi_only_and_does_not_mutate(self):
        angles = np.array([np.pi, -np.pi, 0.5, np.nan, 0.0])
        original = angles.copy()
        result = score_reversals_as_zero(angles)
        np.testing.assert_array_equal(result, [0.0, -np.pi, 0.5, np.nan, 0.0])
        np.testing.assert_array_equal(angles, original)


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
