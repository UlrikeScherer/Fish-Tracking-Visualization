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


def _rzv(metric, remove_zero_vecs: bool) -> dict[str, bool]:
    """remove_zero_vecs kwarg for metrics that accept it; absolute_angles always drops stationary angles."""
    return {} if metric is absolute_angles else {"remove_zero_vecs": remove_zero_vecs}


class TestTurningAngle(unittest.TestCase):
    def test_straight_line_mean_zero(self):
        points = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        result = turning_angle(points, [], _no_errors(10), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertAlmostEqual(result[0, 0], 0.0)
        self.assertEqual(result[0, 2], 8)

    def test_ccw_turns_positive_mean(self):
        points = _ccw_square_loop(n_laps=2)
        result = turning_angle(points, [], _no_errors(len(points)), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertAlmostEqual(result[0, 0], np.pi / 2)

    def test_skip_reduces_point_count(self):
        points = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
        r0 = turning_angle(points, [], _no_errors(20), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        r1 = turning_angle(points, [], _no_errors(20), skip=1, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertGreater(r0[0, 2], r1[0, 2])

    def test_remove_zero_vecs_reduces_count(self):
        # Stationary point at index 2 makes positions 0 and 1 in angle-space non-computable.
        points = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        r_keep = turning_angle(points, [], _no_errors(6), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        r_remove = turning_angle(points, [], _no_errors(6), skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertGreater(r_keep[0, 2], r_remove[0, 2])

    def test_returns_shape(self):
        points = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        result = turning_angle(points, [], _no_errors(10), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertEqual(result.shape, (1, 3))


class TestAbsoluteAngles(unittest.TestCase):
    def test_result_non_negative(self):
        points = _alternating_zigzag(n_steps=8)
        result = absolute_angles(points, [], _no_errors(len(points)), skip=0, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertGreaterEqual(result[0, 0], 0.0)

    def test_mean_geq_abs_signed_mean(self):
        # E(|X|) >= |E(X)| always.
        points = _alternating_zigzag(n_steps=8)
        ta = turning_angle(points, [], _no_errors(len(points)), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        aa = absolute_angles(points, [], _no_errors(len(points)), skip=0, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertGreaterEqual(aa[0, 0], abs(ta[0, 0]))

    def test_point_count_matches_turning_angle(self):
        points = _ccw_square_loop(n_laps=2)
        n = len(points)
        ta = turning_angle(points, [], _no_errors(n), skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0, min_step_px=0.0)
        aa = absolute_angles(points, [], _no_errors(n), skip=0, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertEqual(ta[0, 2], aa[0, 2])

    def test_uniform_turns_mean_equals_abs_angle(self):
        # All turns are +pi/2 so |mean(signed)| == mean(absolute).
        points = _ccw_square_loop(n_laps=2)
        n = len(points)
        ta = turning_angle(points, [], _no_errors(n), skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        aa = absolute_angles(points, [], _no_errors(n), skip=0, wall_threshold_cm=0.0, min_step_px=0.0)
        self.assertAlmostEqual(ta[0, 0], aa[0, 0])

    def test_stationary_angles_excluded_like_turning_angle(self):
        # Zero-length step (duplicate point) plus 1 px jitter steps; only the two 90 degree turns between 5 px steps survive.
        px = np.array([[10, 10], [11, 10], [11, 11], [10, 11], [15, 11], [15, 11], [15, 16], [10, 16], [10, 11]], dtype=float)
        kw = dict(data_px=px, skip=0, wall_threshold_cm=0.0, min_step_px=3.0)
        ta = turning_angle(px, [], _no_errors(len(px)), remove_zero_vecs=True, **kw)
        aa = absolute_angles(px, [], _no_errors(len(px)), **kw)
        self.assertEqual(aa[0, 2], ta[0, 2])
        self.assertEqual(aa[0, 2], 2)
        self.assertAlmostEqual(aa[0, 0], np.pi / 2)

    def test_ignores_remove_zero_vecs_config(self):
        points = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        results = []
        for flag in (False, True):
            with unittest.mock.patch.object(absolute_angles.__globals__["config"], "REMOVE_0_VECS", flag):
                results.append(absolute_angles(points, [], _no_errors(len(points)), skip=0, wall_threshold_cm=0.0, min_step_px=0.0))
        np.testing.assert_array_equal(results[0], results[1])
        self.assertEqual(results[0][0, 2], 2)

    def test_remove_zero_vecs_parameter_removed(self):
        points = _ccw_square_loop(n_laps=2)
        with self.assertRaises(TypeError):
            absolute_angles(points, [], _no_errors(len(points)), remove_zero_vecs=True)


class TestUnaveraged(unittest.TestCase):
    """unaveraged=True returns one [row, angle, excluded] line per angle instead of interval means."""

    KW = dict(skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
    KW_ABS = dict(skip=0, wall_threshold_cm=0.0, min_step_px=0.0)

    def test_shape_and_rows(self):
        points = _ccw_square_loop(n_laps=2)  # 9 points -> 7 angles at middle points 1..7
        result = turning_angle(points, [], _no_errors(len(points)), unaveraged=True, **self.KW)
        self.assertEqual(result.shape, (7, 3))
        np.testing.assert_array_equal(result[:, 0], np.arange(1, 8))
        np.testing.assert_allclose(result[:, 1], np.pi / 2)
        self.assertFalse(result[:, 2].any())

    def test_rows_with_skip(self):
        points = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
        result = turning_angle(points, [], _no_errors(20), unaveraged=True, skip=1, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0)
        # subsampled points are original rows 0, 2, 4, ...; the angle at subsampled point j+1 sits on original row 2*(j+1)
        np.testing.assert_array_equal(result[:, 0], 2 * (np.arange(len(result)) + 1))

    def test_absolute_takes_abs(self):
        points = _alternating_zigzag(n_steps=8)
        signed = turning_angle(points, [], _no_errors(len(points)), unaveraged=True, **self.KW)
        absolute = absolute_angles(points, [], _no_errors(len(points)), unaveraged=True, **self.KW_ABS)
        np.testing.assert_allclose(absolute[:, 1], np.abs(signed[:, 1]))
        self.assertTrue((signed[:, 1] < 0).any())

    def test_error_points_flag_their_three_triplets(self):
        points = _ccw_square_loop(n_laps=2)
        errors = _no_errors(len(points))
        errors[4] = True  # point 4 is touched by the triplets with middle points 3, 4 and 5
        result = turning_angle(points, [], errors, unaveraged=True, **self.KW)
        flagged = result[result[:, 2] == 1.0, 0]
        np.testing.assert_array_equal(flagged, [3, 4, 5])

    def test_nan_position_is_nan_and_excluded(self):
        points = _ccw_square_loop(n_laps=2)
        points[4] = np.nan
        errors = _no_errors(len(points))
        errors[4] = True
        result = turning_angle(points, [], errors, unaveraged=True, **self.KW)
        touched = np.isin(result[:, 0], [3, 4, 5])
        self.assertTrue(np.isnan(result[touched, 1]).all())
        self.assertTrue((result[touched, 2] == 1.0).all())
        np.testing.assert_allclose(result[~touched, 1], np.pi / 2)
        self.assertFalse(result[~touched, 2].any())

    def test_mean_of_included_equals_averaged(self):
        rng = np.random.default_rng(0)
        points = np.cumsum(rng.normal(size=(500, 2)), axis=0)
        errors = rng.random(500) < 0.05
        raw = turning_angle(points, [], errors, unaveraged=True, **self.KW)
        averaged = turning_angle(points, [], errors, unaveraged=False, **self.KW)
        included = raw[raw[:, 2] == 0.0, 1]
        self.assertAlmostEqual(included.mean(), averaged[0, 0])
        self.assertEqual(len(included), averaged[0, 2])

    def test_defaults_to_config(self):
        points = _ccw_square_loop(n_laps=2)
        with unittest.mock.patch.object(turning_angle.__globals__["config"], "UNAVERAGED", True):
            result = turning_angle(points, [], _no_errors(len(points)), **self.KW)
        self.assertEqual(result.shape, (7, 3))


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
            wall_threshold_cm=wall,
            **_rzv(metric, remove_zero_vecs),
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
        keep = self._run(turning_angle, wall=self.THRESHOLD, remove_zero_vecs=False)
        remove = self._run(turning_angle, wall=self.THRESHOLD, remove_zero_vecs=True)
        np.testing.assert_array_equal(keep, remove)
        self.assertEqual(self._run(absolute_angles, wall=self.THRESHOLD, remove_zero_vecs=True)[0, 2], remove[0, 2])

    def test_streak_length_excludes_wall_angles(self):
        kwargs = dict(area=self.AREA, data_px=self.points, skip=0, wall_threshold_cm=self.THRESHOLD, unaveraged=True)
        keep = turning_angle_streak_length(self.points, [], self.no_errors, remove_zero_vecs=False, **kwargs)
        remove = turning_angle_streak_length(self.points, [], self.no_errors, remove_zero_vecs=True, **kwargs)
        np.testing.assert_array_equal(keep, remove)
        free = turning_angle_streak_length(self.points, [], self.no_errors, skip=0, remove_zero_vecs=False, wall_threshold_cm=0.0, min_step_px=0.0, unaveraged=True)
        self.assertLess(keep.sum(), free.sum())


class TestMinStepPx(unittest.TestCase):
    """Steps shorter than min_step_px tracker pixels are treated like zero-length steps."""

    # Pixel path: three 1 px jitter steps, then 5 px moves with a constant +90 degree turn.
    PX = np.array([[10, 10], [11, 10], [11, 11], [10, 11], [15, 11], [15, 16], [10, 16], [10, 11]], dtype=float)
    CM = PX * 0.05  # angles are computed in cm; the cut-off must still be judged in pixels

    def _run(self, metric, *, min_step_px: float, remove_zero_vecs: bool, **kw) -> NDArray[np.float64]:
        return metric(self.CM, [], _no_errors(len(self.PX)), data_px=self.PX, skip=0, wall_threshold_cm=0.0, min_step_px=min_step_px, **_rzv(metric, remove_zero_vecs), **kw)

    def test_short_steps_dropped_with_remove_zero_vecs(self):
        for metric in (turning_angle, absolute_angles):
            with self.subTest(metric=metric.__name__):
                result = self._run(metric, min_step_px=3.0, remove_zero_vecs=True)
                self.assertEqual(result[0, 2], 3)  # angles at middle points 4, 5, 6 only
                self.assertAlmostEqual(result[0, 0], np.pi / 2)

    def test_short_steps_count_as_zero_without_remove_zero_vecs(self):
        # Angles at middle points 1-3 touch a jitter step and fill 0.0; points 4-6 are +90 degrees.
        result = self._run(turning_angle, min_step_px=3.0, remove_zero_vecs=False)
        self.assertEqual(result[0, 2], 6)
        self.assertAlmostEqual(result[0, 0], 3 * (np.pi / 2) / 6)

    def test_zero_threshold_keeps_jitter_angles(self):
        result = self._run(absolute_angles, min_step_px=0.0, remove_zero_vecs=True)
        self.assertEqual(result[0, 2], 6)

    def test_threshold_is_inclusive_at_cut_off(self):
        # 5 px steps survive a 5 px cut-off, 1 px steps do not
        self.assertEqual(self._run(turning_angle, min_step_px=5.0, remove_zero_vecs=True)[0, 2], 3)

    def test_skip_judges_subsampled_steps(self):
        # 20 points one pixel apart on a line: 1 px steps, but 2 px apart after skip=1
        px = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
        kw = dict(data_px=px, remove_zero_vecs=True, wall_threshold_cm=0.0, min_step_px=2.0)
        self.assertEqual(turning_angle(px, [], _no_errors(20), skip=0, **kw)[0, 2], 0)
        self.assertEqual(turning_angle(px, [], _no_errors(20), skip=1, **kw)[0, 2], 8)

    def test_streak_length_ignores_short_steps(self):
        result = self._run(turning_angle_streak_length, min_step_px=3.0, remove_zero_vecs=True, unaveraged=True)
        np.testing.assert_array_equal(result, [3.0])

    def test_requires_pixel_positions(self):
        with self.assertRaises(ValueError):
            turning_angle(self.CM, [], _no_errors(len(self.CM)), skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0, min_step_px=3.0)


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
            min_step_px=0.0,
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
            min_step_px=0.0,
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
            min_step_px=0.0,
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
            min_step_px=0.0,
            unaveraged=True,
        )
        r_remove = turning_angle_streak_length(
            points,
            [],
            _no_errors(7),
            skip=0,
            remove_zero_vecs=True,
            wall_threshold_cm=0.0,
            min_step_px=0.0,
            unaveraged=True,
        )
        self.assertGreater(sum(r_keep), sum(r_remove))


if __name__ == "__main__":
    unittest.main()
