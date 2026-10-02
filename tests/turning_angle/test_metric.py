import unittest
import unittest.mock

import numpy as np
from numpy.typing import NDArray

from fishproviz.metrics.turning_angle.metric import (
    turning_angle,
    absolute_angles,
    turning_angle_streak_length,
)


def _no_errors(n: int) -> NDArray[np.bool_]:
    return np.zeros(n, dtype=bool)


def _ccw_square_loop(n_laps: int = 2) -> NDArray[np.float64]:
    """Counterclockwise square traversal; each lap produces 4 turns of +pi/2."""
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    pts = corners * n_laps + [[0, 0]]
    return np.array(pts, dtype=float)


def _alternating_zigzag(n_steps: int = 6) -> NDArray[np.float64]:
    """Points that alternate between +pi/2 and -pi/2 turns."""
    pts = []
    for i in range(n_steps + 1):
        pts.append([i, i % 2])
    return np.array(pts, dtype=float)


class TestTurningAngle(unittest.TestCase):
    def test_straight_line_mean_zero(self):
        points = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        result = turning_angle(points, [], _no_errors(10), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        self.assertAlmostEqual(result[0, 0], 0.0)
        self.assertEqual(result[0, 2], 8)

    def test_ccw_turns_positive_mean(self):
        points = _ccw_square_loop(n_laps=2)
        result = turning_angle(points, [], _no_errors(len(points)), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        self.assertAlmostEqual(result[0, 0], np.pi / 2)

    def test_skip_reduces_point_count(self):
        points = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
        r0 = turning_angle(points, [], _no_errors(20), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        r1 = turning_angle(points, [], _no_errors(20), skip=1, remove_zero_vecs=False, wall_threshold_cm=0.0)
        self.assertGreater(r0[0, 2], r1[0, 2])

    def test_remove_zero_vecs_reduces_count(self):
        # Stationary point at index 2 makes positions 0 and 1 in angle-space non-computable.
        points = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        r_keep = turning_angle(points, [], _no_errors(6), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        r_remove = turning_angle(points, [], _no_errors(6), skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0)
        self.assertGreater(r_keep[0, 2], r_remove[0, 2])

    def test_returns_shape(self):
        points = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        result = turning_angle(points, [], _no_errors(10), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        self.assertEqual(result.shape, (1, 3))


class TestAbsoluteAngles(unittest.TestCase):
    def test_result_non_negative(self):
        points = _alternating_zigzag(n_steps=8)
        result = absolute_angles(points, [], _no_errors(len(points)), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        self.assertGreaterEqual(result[0, 0], 0.0)

    def test_mean_geq_abs_signed_mean(self):
        # E(|X|) >= |E(X)| always.
        points = _alternating_zigzag(n_steps=8)
        ta = turning_angle(points, [], _no_errors(len(points)), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        aa = absolute_angles(points, [], _no_errors(len(points)), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        self.assertGreaterEqual(aa[0, 0], abs(ta[0, 0]))

    def test_point_count_matches_turning_angle(self):
        points = _ccw_square_loop(n_laps=2)
        n = len(points)
        ta = turning_angle(points, [], _no_errors(n), skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0)
        aa = absolute_angles(points, [], _no_errors(n), skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0)
        self.assertEqual(ta[0, 2], aa[0, 2])

    def test_uniform_turns_mean_equals_abs_angle(self):
        # All turns are +pi/2 so |mean(signed)| == mean(absolute).
        points = _ccw_square_loop(n_laps=2)
        n = len(points)
        ta = turning_angle(points, [], _no_errors(n), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        aa = absolute_angles(points, [], _no_errors(n), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0)
        self.assertAlmostEqual(ta[0, 0], aa[0, 0])


class TestWallExclusion(unittest.TestCase):
    """Wall-adjacent angles must be dropped, never counted as 0.0, whatever remove_zero_vecs is."""

    # Pixel tank 100 x 100; px2cm is patched to identity so thresholds are in pixel units.
    AREA = ("test_cam", np.array([[0, 0], [100, 0], [100, 100], [50, 100], [0, 100]], dtype=float))
    THRESHOLD = 15.0

    def setUp(self) -> None:
        # Circle of radius 40 around the tank centre: constant 10 degree turns, 10 px from the wall at its extremes.
        angles = np.deg2rad(np.arange(0, 360, 10))
        self.points = np.column_stack([50 + 40 * np.cos(angles), 50 + 40 * np.sin(angles)])
        self.no_errors = _no_errors(len(self.points))
        patcher = unittest.mock.patch.dict(turning_angle.__globals__, {"px2cm": lambda a, fish_key=None: a})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, metric, *, wall: float, remove_zero_vecs: bool) -> NDArray[np.float64]:
        return metric(
            self.points,
            [],
            self.no_errors,
            area=self.AREA,
            data_px=self.points,
            skip=0,
            remove_zero_vecs=remove_zero_vecs,
            wall_threshold_cm=wall,
        )

    def test_wall_exclusion_removes_angles(self):
        free = self._run(turning_angle, wall=0.0, remove_zero_vecs=False)
        walled = self._run(turning_angle, wall=self.THRESHOLD, remove_zero_vecs=False)
        self.assertLess(walled[0, 2], free[0, 2])

    def test_wall_excluded_angles_not_counted_as_zero(self):
        # Every real angle is 10 degrees, so any 0.0 fill leaking into the mean pulls it below 10.
        for remove_zero_vecs in (False, True):
            with self.subTest(remove_zero_vecs=remove_zero_vecs):
                walled = self._run(turning_angle, wall=self.THRESHOLD, remove_zero_vecs=remove_zero_vecs)
                self.assertAlmostEqual(walled[0, 0], np.deg2rad(10))
                self.assertAlmostEqual(walled[0, 1], 0.0)

    def test_wall_result_independent_of_remove_zero_vecs(self):
        # No stationary steps in this path, so the flag must not change anything.
        for metric in (turning_angle, absolute_angles):
            with self.subTest(metric=metric.__name__):
                keep = self._run(metric, wall=self.THRESHOLD, remove_zero_vecs=False)
                remove = self._run(metric, wall=self.THRESHOLD, remove_zero_vecs=True)
                np.testing.assert_array_equal(keep, remove)

    def test_streak_length_excludes_wall_angles(self):
        kwargs = dict(area=self.AREA, data_px=self.points, skip=0, wall_threshold_cm=self.THRESHOLD, unaveraged=True)
        keep = turning_angle_streak_length(self.points, [], self.no_errors, remove_zero_vecs=False, **kwargs)
        remove = turning_angle_streak_length(self.points, [], self.no_errors, remove_zero_vecs=True, **kwargs)
        np.testing.assert_array_equal(keep, remove)
        free = turning_angle_streak_length(self.points, [], self.no_errors, skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, unaveraged=True)
        self.assertLess(keep.sum(), free.sum())


class TestTurningAngleStreakLength(unittest.TestCase):
    def test_all_same_sign_one_streak(self):
        points = _ccw_square_loop(n_laps=2)
        n = len(points)
        result = turning_angle_streak_length(
            points,
            [],
            _no_errors(n),
            skip=0,
            remove_zero_vecs=False,
            wall_threshold_cm=0.0,
            unaveraged=True,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], n - 2)

    def test_alternating_sign_unit_streaks(self):
        points = _alternating_zigzag(n_steps=8)
        n = len(points)
        result = turning_angle_streak_length(
            points,
            [],
            _no_errors(n),
            skip=0,
            remove_zero_vecs=False,
            wall_threshold_cm=0.0,
            unaveraged=True,
        )
        self.assertTrue(np.all(result == 1.0))

    def test_unaveraged_false_returns_summary(self):
        points = _ccw_square_loop(n_laps=2)
        n = len(points)
        result = turning_angle_streak_length(
            points,
            [],
            _no_errors(n),
            skip=0,
            remove_zero_vecs=False,
            wall_threshold_cm=0.0,
            unaveraged=False,
        )
        self.assertEqual(result.shape, (3,))

    def test_remove_zero_vecs_nan_fill_excluded_from_streaks(self):
        # With remove_zero_vecs=True, stationary point produces NaN fill → excluded.
        points = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0]], dtype=float)
        r_keep = turning_angle_streak_length(
            points,
            [],
            _no_errors(7),
            skip=0,
            remove_zero_vecs=False,
            wall_threshold_cm=0.0,
            unaveraged=True,
        )
        r_remove = turning_angle_streak_length(
            points,
            [],
            _no_errors(7),
            skip=0,
            remove_zero_vecs=True,
            wall_threshold_cm=0.0,
            unaveraged=True,
        )
        self.assertGreater(sum(r_keep), sum(r_remove))


if __name__ == "__main__":
    unittest.main()
