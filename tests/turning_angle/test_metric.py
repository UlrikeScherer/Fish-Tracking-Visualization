import unittest
import unittest.mock
from typing import Optional

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


class TestIntervalBoundaries(unittest.TestCase):
    """Each angle is counted in the interval that holds its middle point."""

    def _counts(self, metric, n_points: int, split: list[int], skip: int) -> list[float]:
        points = np.column_stack([np.arange(n_points, dtype=float), np.zeros(n_points)])
        kw = dict(skip=skip, wall_threshold_cm=0.0, min_step_px=0.0, unaveraged=False, **_rzv(metric, False))
        return metric(points, split, _no_errors(n_points), **kw)[:, 2].tolist()

    def test_split_counts(self):
        # (n_points, split rows, skip) -> angles per interval; middle rows are 1..n-2 (skip 0) or 2, 4, ... (skip 1)
        cases = {
            (10, (5,), 0): [4.0, 4.0],
            (20, (10,), 1): [4.0, 4.0],
            (20, (11,), 1): [5.0, 3.0],
            (10, (0, 5), 0): [0.0, 4.0, 4.0],
        }
        for metric in (turning_angle, absolute_angles):
            for (n, split, skip), expected in cases.items():
                with self.subTest(metric=metric.__name__, n=n, split=split, skip=skip):
                    self.assertEqual(self._counts(metric, n, list(split), skip), expected)


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

    def test_stationary_nan_angles_are_excluded(self):
        points = _ccw_square_loop(n_laps=2)
        points = np.insert(points, 4, points[4], axis=0)  # repeat point 4: zero-length step 4 -> 5
        n = len(points)
        signed = turning_angle(points, [], _no_errors(n), unaveraged=True, skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0, min_step_px=0.0)
        absolute = absolute_angles(points, [], _no_errors(n), unaveraged=True, **self.KW_ABS)
        for result in (signed, absolute):
            stationary = np.isin(result[:, 0], [4, 5])
            self.assertTrue(np.isnan(result[stationary, 1]).all())
            self.assertTrue((result[stationary, 2] == 1.0).all())
            self.assertFalse(result[~stationary, 2].any())

    def test_stationary_zero_fill_stays_included(self):
        points = _ccw_square_loop(n_laps=2)
        points = np.insert(points, 4, points[4], axis=0)
        result = turning_angle(points, [], _no_errors(len(points)), unaveraged=True, **self.KW)
        stationary = np.isin(result[:, 0], [4, 5])
        np.testing.assert_array_equal(result[stationary, 1], 0.0)
        self.assertFalse(result[:, 2].any())

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

    def test_streak_length_breaks_at_wall_angles(self):
        # straight_px=0: the 10 degree turns on this circle deviate about 1.2 px sideways
        kwargs = dict(area=self.AREA, data_px=self.points, skip=0, min_step_px=0.0, straight_px=0.0, unaveraged=True)
        free = turning_angle_streak_length(self.points, [], self.no_errors, wall_threshold_cm=0.0, **kwargs)
        walled = turning_angle_streak_length(self.points, [], self.no_errors, wall_threshold_cm=self.THRESHOLD, **kwargs)
        self.assertEqual(len(free), 1)
        self.assertGreater(len(walled), 1)
        self.assertLess(walled[:, 3].sum(), free[0, 3])


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

    def test_requires_pixel_positions(self):
        with self.assertRaises(ValueError):
            turning_angle(self.CM, [], _no_errors(len(self.CM)), skip=0, remove_zero_vecs=True, wall_threshold_cm=0.0, min_step_px=3.0)


class TestReversalScoring(unittest.TestCase):
    """Signed turning_angle scores an exact reversal 0.0; absolute_angles keeps pi; both count it."""

    # 90 degree right turn, then an exact reversal at middle point 2, then straight
    POINTS = np.array([[0, 0], [2, 0], [2, 2], [2, 0], [2, -2]], dtype=float)
    KW = dict(skip=0, wall_threshold_cm=0.0, min_step_px=0.0)

    def test_averaged_scores_and_counts(self):
        n = len(self.POINTS)
        ta = turning_angle(self.POINTS, [], _no_errors(n), remove_zero_vecs=True, **self.KW)
        aa = absolute_angles(self.POINTS, [], _no_errors(n), **self.KW)
        self.assertEqual(ta[0, 2], 3)
        self.assertEqual(aa[0, 2], ta[0, 2])
        self.assertAlmostEqual(ta[0, 0], (np.pi / 2 + 0.0 + 0.0) / 3)
        self.assertAlmostEqual(aa[0, 0], (np.pi / 2 + np.pi + 0.0) / 3)

    def test_unaveraged_reversal_row(self):
        n = len(self.POINTS)
        ta = turning_angle(self.POINTS, [], _no_errors(n), remove_zero_vecs=True, unaveraged=True, **self.KW)
        aa = absolute_angles(self.POINTS, [], _no_errors(n), unaveraged=True, **self.KW)
        self.assertEqual(ta[1, 0], 2)
        self.assertEqual(ta[1, 1], 0.0)
        self.assertEqual(aa[1, 1], np.pi)


class TestReversalEdgeCases(unittest.TestCase):
    KW = dict(skip=0, wall_threshold_cm=0.0)

    def test_reversal_next_to_short_step_is_stationary_not_pi(self):
        # Steps (10,0), (0,10), (0,-1), (0,10): both reversal angles touch the 1 px step (< min_step_px).
        px = np.array([[0, 0], [10, 0], [10, 10], [10, 9], [10, 19]], dtype=float)
        cm = px * 0.05
        n = len(px)
        aa = absolute_angles(cm, [], _no_errors(n), data_px=px, min_step_px=3.0, **self.KW)
        self.assertEqual(aa[0, 2], 1)
        self.assertAlmostEqual(aa[0, 0], np.pi / 2)
        ta = turning_angle(cm, [], _no_errors(n), data_px=px, min_step_px=3.0, remove_zero_vecs=False, unaveraged=True, **self.KW)
        np.testing.assert_allclose(ta[:, 1], [np.pi / 2, 0.0, 0.0])


def _path(steps: list[tuple[int, int]], start: tuple[int, int] = (100, 100)) -> NDArray[np.float64]:
    """Integer pixel path from a start point and a list of step vectors."""
    return np.cumsum(np.vstack([start, steps]), axis=0).astype(float)


# On tracker pixels (y down) these step pairs turn right (+) and left (-)
SQUARE_RIGHT = [(5, 0), (0, 5), (-5, 0), (0, -5)] * 2  # 7 right turns at rows 1..7
RIGHT_THEN_LEFT = [(5, 0), (0, 5), (-5, 0), (0, -5), (-5, 0), (0, 5), (5, 0)]  # R, R, R at rows 1..3, L, L, L at rows 4..6


class TestTurningAngleStreakLength(unittest.TestCase):
    """Streaks count consecutive same-side turns; removed, stationary and reversal angles break them, straight ones are neutral."""

    KW = dict(skip=0, wall_threshold_cm=0.0, min_step_px=0.0, straight_px=0.0)

    def _raw(self, px: NDArray[np.float64], errors: Optional[NDArray[np.bool_]] = None, **kw) -> list[list[float]]:
        errors = _no_errors(len(px)) if errors is None else errors
        result = turning_angle_streak_length(px, [], errors, data_px=px, unaveraged=True, **{**self.KW, **kw})
        self.assertEqual(result.shape[1:], (4,))
        self.assertEqual(result.dtype, np.float64)
        return result.tolist()

    def test_same_side_one_streak(self):
        self.assertEqual(self._raw(_path(SQUARE_RIGHT)), [[1, 7, 1, 7]])

    def test_side_change_splits_in_time_order(self):
        self.assertEqual(self._raw(_path(RIGHT_THEN_LEFT)), [[1, 3, 1, 3], [4, 6, -1, 3]])

    def test_gap_breaks(self):
        px = _path(SQUARE_RIGHT)
        px[4] = np.nan  # angles at middle rows 3, 4, 5 are NaN
        self.assertEqual(self._raw(px), [[1, 2, 1, 2], [6, 7, 1, 2]])

    def test_error_point_breaks(self):
        errors = _no_errors(9)
        errors[4] = True
        self.assertEqual(self._raw(_path(SQUARE_RIGHT), errors), [[1, 2, 1, 2], [6, 7, 1, 2]])

    def test_zero_length_step_breaks(self):
        steps = SQUARE_RIGHT[:4] + [(0, 0)] + SQUARE_RIGHT[4:]  # angles at rows 4 and 5 touch the zero step
        self.assertEqual(self._raw(_path(steps)), [[1, 3, 1, 3], [6, 8, 1, 3]])

    def test_short_step_breaks_only_under_min_step_px(self):
        # a 1 px step continuing the incoming direction: straight (neutral) without a cut-off, stationary with one
        steps = [(5, 0), (0, 5), (-5, 0), (-1, 0), (0, -5), (5, 0)]
        px = _path(steps)
        self.assertEqual(self._raw(px), [[1, 5, 1, 4]])
        self.assertEqual(self._raw(px, min_step_px=3.0), [[1, 2, 1, 2], [5, 5, 1, 1]])

    def test_exact_reversal_breaks(self):
        steps = [(5, 0), (0, 5), (0, -5), (5, 0), (0, 5)]  # R, reversal, R, R
        self.assertEqual(self._raw(_path(steps)), [[1, 1, 1, 1], [3, 4, 1, 2]])

    def test_straight_is_neutral(self):
        # R, S, R is one streak of 2; L, S, R gives two streaks of 1
        self.assertEqual(self._raw(_path([(5, 0), (0, 5), (0, 5), (-5, 0)])), [[1, 3, 1, 2]])
        self.assertEqual(self._raw(_path([(5, 0), (0, -5), (0, -5), (5, 0)])), [[1, 1, -1, 1], [3, 3, 1, 1]])

    def test_only_straight_gives_no_streak(self):
        px = _path([(5, 0)] * 6)
        self.assertEqual(self._raw(px), [])
        averaged = turning_angle_streak_length(px, [], _no_errors(len(px)), data_px=px, unaveraged=False, **self.KW)
        self.assertEqual(averaged.shape, (1, 3))
        self.assertTrue(np.isnan(averaged[0, :2]).all())
        self.assertEqual(averaged[0, 2], 0)

    def test_break_beats_straight(self):
        steps = [(5, 0), (0, 5), (0, 5), (0, 5), (0, 5), (-5, 0)]  # R, S, S, S, R
        px = _path(steps)
        self.assertEqual(self._raw(px), [[1, 5, 1, 2]])
        errors = _no_errors(len(px))
        errors[3] = True  # removes the straight angles at rows 2, 3, 4
        self.assertEqual(self._raw(px, errors), [[1, 1, 1, 1], [5, 5, 1, 1]])

    def test_rounded_straight_path_has_no_streaks_at_default_threshold(self):
        # constant 7 px per frame at 23 degrees, rounded to integer pixels like BioTracker output
        t = np.arange(200)
        px = np.round(np.column_stack([100 + 7 * np.cos(np.deg2rad(23)) * t, 100 + 7 * np.sin(np.deg2rad(23)) * t]))
        self.assertEqual(self._raw(px, straight_px=1.5), [])
        self.assertGreater(len(self._raw(px, straight_px=0.0)), 20)

    def test_requires_pixel_positions(self):
        px = _path(SQUARE_RIGHT)
        for straight_px in (0.0, 1.5):
            with self.subTest(straight_px=straight_px), self.assertRaises(ValueError):
                turning_angle_streak_length(px, [], _no_errors(len(px)), unaveraged=True, **{**self.KW, "straight_px": straight_px})

    def test_negative_straight_px_raises(self):
        with self.assertRaises(ValueError):
            self._raw(_path(SQUARE_RIGHT), straight_px=-1.0)

    def test_straightness_judged_on_pixels_not_cm(self):
        # collinear pixel motion stays straight at straight_px=0 even though the rotated cm positions give cross ~ 1e-14
        px = np.array([[0, 0], [3, 1], [6, 2], [9, 3], [12, 4], [15, 5]], dtype=float)
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        cm = ((px - 450) @ np.array([[c, -s], [s, c]])) * 0.05
        result = turning_angle_streak_length(cm, [], _no_errors(len(px)), data_px=px, unaveraged=True, **self.KW)
        self.assertEqual(result.tolist(), [])

    def test_interval_by_first_angle(self):
        px = _path(RIGHT_THEN_LEFT)
        n = len(px)

        def averaged(split: list[int]) -> NDArray[np.float64]:
            return turning_angle_streak_length(px, split, _no_errors(n), data_px=px, unaveraged=False, **self.KW)

        # split at row 3: the right streak (rows 1..3) spans the boundary but stays whole in interval 0
        np.testing.assert_array_equal(averaged([3]), [[3, 0, 1], [3, 0, 1]])
        # split at rows 4 and 5: the left streak starts in interval 1, interval 2 is empty
        np.testing.assert_array_equal(averaged([4, 5]), [[3, 0, 1], [3, 0, 1], [np.nan, np.nan, 0]])
        # split at row 5: both streaks start before it
        np.testing.assert_array_equal(averaged([5]), [[3, 0, 2], [np.nan, np.nan, 0]])

    def test_skip_row_mapping(self):
        corners = _path(SQUARE_RIGHT)
        px = np.empty((2 * len(corners) - 1, 2))
        px[::2] = corners
        px[1::2] = (corners[:-1] + corners[1:]) / 2  # odd rows are skipped
        self.assertEqual(self._raw(px, skip=1), [[2, 14, 1, 7]])

    def test_averaged_matches_unaveraged(self):
        rng = np.random.default_rng(0)
        px = np.round(np.cumsum(rng.normal(scale=5.0, size=(500, 2)), axis=0))
        errors = rng.random(500) < 0.05
        kw = dict(data_px=px, skip=0, wall_threshold_cm=0.0, min_step_px=3.0, straight_px=1.5)
        raw = turning_angle_streak_length(px, [], errors, unaveraged=True, **kw)
        averaged = turning_angle_streak_length(px, [], errors, unaveraged=False, **kw)
        self.assertEqual(averaged[0, 2], len(raw))
        self.assertAlmostEqual(averaged[0, 0], raw[:, 3].mean())

    def test_short_inputs(self):
        for n in (0, 1, 2):
            with self.subTest(n=n):
                px = _path(SQUARE_RIGHT)[:n]
                self.assertEqual(self._raw(px), [])
                averaged = turning_angle_streak_length(px, [], _no_errors(n), data_px=px, unaveraged=False, **self.KW)
                self.assertEqual(averaged.shape, (1, 3))
                self.assertEqual(averaged[0, 2], 0)

    def test_remove_zero_vecs_parameter_removed(self):
        px = _path(SQUARE_RIGHT)
        with self.assertRaises(TypeError):
            turning_angle_streak_length(px, [], _no_errors(len(px)), remove_zero_vecs=True, **self.KW)


if __name__ == "__main__":
    unittest.main()
