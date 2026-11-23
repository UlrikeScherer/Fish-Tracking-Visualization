import unittest
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from fishproviz.metrics.turning_angle.compute import compute_turning_angles, compute_turning_angle_streaks, drop_reversals, straight_turns


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

    def test_zero_and_one_point_return_empty(self):
        for n in (0, 1):
            with self.subTest(n=n):
                self.assertEqual(compute_turning_angles(np.zeros((n, 2))).shape, (0,))


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


class TestDropReversals(unittest.TestCase):
    def test_maps_pi_only_and_does_not_mutate(self):
        angles = np.array([np.pi, -np.pi, 0.5, np.nan, 0.0])
        original = angles.copy()
        result = drop_reversals(angles)
        np.testing.assert_array_equal(result, [np.nan, -np.pi, 0.5, np.nan, 0.0])
        np.testing.assert_array_equal(angles, original)


class TestStraightTurns(unittest.TestCase):
    def test_lateral_threshold_and_direction(self):
        # incoming step (10, 0); outgoing steps 1 px, 2 px sideways, straight on, then backwards
        points = np.array([[0, 0], [10, 0], [20, 1], [30, 1], [40, 1], [30, 1]], dtype=float)
        # angle 0: lateral 1 px; angle 1: lateral 1 px (incoming (10,1), outgoing (10,0)); angle 2: 0 px; angle 3: reversal
        np.testing.assert_array_equal(straight_turns(points, 1.5), [True, True, True, False])
        np.testing.assert_array_equal(straight_turns(points, 0.0), [False, False, True, False])

    def test_lateral_is_judged_against_incoming_step_length(self):
        # cross 20 over incoming length 10 gives 2 px sideways
        points = np.array([[0, 0], [10, 0], [20, 2]], dtype=float)
        np.testing.assert_array_equal(straight_turns(points, 1.9), [False])
        np.testing.assert_array_equal(straight_turns(points, 2.0), [True])

    def test_nan_and_zero_length_steps_are_false_without_warnings(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [2, 0], [np.nan, np.nan], [4, 0], [5, 0], [6, 0]], dtype=float)
        with np.errstate(all="raise"):
            result = straight_turns(points, 1.5)
        np.testing.assert_array_equal(result, [False, False, False, False, False, True])

    def test_short_inputs_return_empty(self):
        for n in (0, 1, 2):
            with self.subTest(n=n):
                result = straight_turns(np.zeros((n, 2)), 1.5)
                self.assertEqual(result.shape, (0,))
                self.assertEqual(result.dtype, np.bool_)


class TestComputeTurningAngleStreaks(unittest.TestCase):
    @staticmethod
    def _streaks(angles: list[float], breaks: Optional[list[bool]] = None, straight: Optional[list[bool]] = None) -> list[list[int]]:
        n = len(angles)
        brk = np.zeros(n, dtype=bool) if breaks is None else np.array(breaks)
        st = np.zeros(n, dtype=bool) if straight is None else np.array(straight)
        result = compute_turning_angle_streaks(np.array(angles, dtype=float), brk, st)
        assert result.dtype == np.int64 and result.shape[1] == 4
        return result.tolist()

    def test_same_side_one_streak(self):
        self.assertEqual(self._streaks([0.1, 0.2, 0.3]), [[0, 2, 1, 3]])

    def test_side_change_splits_in_time_order(self):
        self.assertEqual(self._streaks([-0.1, -0.2, 0.1, 0.2, 0.3]), [[0, 1, -1, 2], [2, 4, 1, 3]])

    def test_nan_breaks(self):
        self.assertEqual(self._streaks([0.1, np.nan, 0.1]), [[0, 0, 1, 1], [2, 2, 1, 1]])

    def test_break_mask_breaks(self):
        self.assertEqual(self._streaks([0.1, 0.2, 0.1], breaks=[False, True, False]), [[0, 0, 1, 1], [2, 2, 1, 1]])

    def test_exact_reversal_breaks_near_reversal_keeps_side(self):
        self.assertEqual(self._streaks([0.1, np.pi, 0.1]), [[0, 0, 1, 1], [2, 2, 1, 1]])
        self.assertEqual(self._streaks([-0.1, -np.pi + 1e-3, -0.1]), [[0, 2, -1, 3]])

    def test_straight_is_neutral(self):
        # R, S, R is one streak of 2; L, S, R gives two streaks of 1
        self.assertEqual(self._streaks([0.1, 0.01, 0.2], straight=[False, True, False]), [[0, 2, 1, 2]])
        self.assertEqual(self._streaks([-0.1, 0.01, 0.2], straight=[False, True, False]), [[0, 0, -1, 1], [2, 2, 1, 1]])

    def test_only_straight_gives_none(self):
        self.assertEqual(self._streaks([0.01, -0.01, 0.0], straight=[True, True, True]), [])

    def test_break_beats_straight(self):
        self.assertEqual(self._streaks([0.1, 0.01, 0.2], breaks=[False, True, False], straight=[False, True, False]), [[0, 0, 1, 1], [2, 2, 1, 1]])

    def test_exact_zero_is_neutral(self):
        self.assertEqual(self._streaks([0.1, 0.0, 0.2]), [[0, 2, 1, 2]])

    def test_first_and_last_are_counted_angles(self):
        self.assertEqual(self._streaks([0.0, 0.1, 0.0, 0.2, 0.0]), [[1, 3, 1, 2]])

    def test_empty(self):
        result = compute_turning_angle_streaks(np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
        self.assertEqual(result.shape, (0, 4))
        self.assertEqual(result.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
