"""Tests for tortuosity: the window kernel, runs split at excluded steps, bins by window start and raw window output."""

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fishproviz.config as config
import fishproviz.metrics as metrics
from fishproviz.methods import tortuosity_windows
from fishproviz.metrics.metrics import _result_table, included_step_runs, steps_and_exclusions, tortuosity
from fishproviz.utils.error_filter import error_default_points
from fishproviz.utils.utile import read_batch_csv
from tests.test_step_metrics import StepMetricTestCase

_WINDOW_CM = 10.0
_HEADER = "FRAME;MillisecsByFPS;objectName;valid;id;coordinateUnit;x;y;rad;deg;xpx;ypx;time;timeString"


def _line(n_points: int, step: float = 1.0, start: float = 0.0) -> np.ndarray:
    """Straight path along x with n_points positions spaced step apart."""
    return np.column_stack([start + step * np.arange(n_points, dtype=np.float64), np.zeros(n_points)])


def _ratios(windows: np.ndarray) -> np.ndarray:
    return windows[:, 2] / windows[:, 3]


def _day_windows(data: np.ndarray, error_index: np.ndarray) -> np.ndarray:
    """Raw tortuosity windows of a synthetic day: [start_row, end_row, C, L, tortuosity]."""
    return tortuosity(data, [], error_index, unaveraged=True)


def _assert_no_window_spans(case: unittest.TestCase, windows: np.ndarray, excluded_steps: np.ndarray) -> None:
    for start, end in windows[:, :2].astype(int):
        case.assertFalse(excluded_steps[start:end].any(), f"window rows {start}..{end} span an excluded step")


class TestKernel(unittest.TestCase):
    def test_straight_line_is_one(self) -> None:
        windows = tortuosity_windows(_line(31), _WINDOW_CM)
        self.assertEqual(windows.shape, (3, 4))
        np.testing.assert_allclose(_ratios(windows), 1.0)

    def test_arc_of_radius_five(self) -> None:
        theta = np.linspace(0.0, 4.0, 40001)
        arc = 5.0 * np.column_stack([np.cos(theta), np.sin(theta)])
        windows = tortuosity_windows(arc, _WINDOW_CM)
        self.assertGreater(len(windows), 0)
        np.testing.assert_allclose(_ratios(windows), 10.0 / (10.0 * np.sin(1.0)), atol=1e-3)

    def test_zigzag_at_45_degrees_is_sqrt_two(self) -> None:
        zigzag = np.column_stack([np.arange(65, dtype=np.float64), np.arange(65) % 2])
        windows = tortuosity_windows(zigzag, _WINDOW_CM)
        self.assertGreater(len(windows), 0)
        np.testing.assert_allclose(_ratios(windows), np.sqrt(2.0))

    def test_closed_square(self) -> None:
        square = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]])
        windows = tortuosity_windows(square, _WINDOW_CM)
        self.assertEqual(windows.shape, (2, 4))
        np.testing.assert_allclose(_ratios(windows), 10.0 / (5.0 * np.sqrt(2.0)))

    def test_windows_tile_the_run(self) -> None:
        rng = np.random.default_rng(0)
        walk = np.cumsum(rng.uniform(-1.5, 1.5, size=(2000, 2)), axis=0)
        windows = tortuosity_windows(walk, _WINDOW_CM)
        self.assertGreater(len(windows), 10)
        self.assertEqual(windows[0, 0], 0)
        np.testing.assert_array_equal(windows[1:, 0], windows[:-1, 1])
        self.assertTrue((windows[:, 2] >= _WINDOW_CM).all())
        steps = np.linalg.norm(np.diff(walk, axis=0), axis=1)
        for start, end, path, straight in windows:
            self.assertAlmostEqual(path, steps[int(start) : int(end)].sum(), places=9)
            self.assertLess(steps[int(start) : int(end) - 1].sum(), _WINDOW_CM)
            self.assertAlmostEqual(straight, np.linalg.norm(walk[int(end)] - walk[int(start)]), places=9)

    def test_short_tail_adds_no_window(self) -> None:
        base = tortuosity_windows(_line(21), _WINDOW_CM)
        for extra in (1, 2, 3):
            with self.subTest(extra=extra):
                np.testing.assert_array_equal(tortuosity_windows(_line(21 + extra), _WINDOW_CM), base)

    def test_fewer_points_than_one_window_is_empty(self) -> None:
        for n_points in (0, 1, 2, 10):
            with self.subTest(n_points=n_points):
                windows = tortuosity_windows(_line(n_points), _WINDOW_CM)
                self.assertEqual(windows.shape, (0, 4))

    def test_non_finite_input_raises_naming_row(self) -> None:
        for bad in (np.nan, np.inf):
            run = _line(30)
            run[5, 1] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "row 5"):
                tortuosity_windows(run, _WINDOW_CM)

    def test_duplicate_points_do_not_change_path(self) -> None:
        line = _line(21)
        doubled = np.repeat(line, 2, axis=0)
        windows, doubled_windows = tortuosity_windows(line, _WINDOW_CM), tortuosity_windows(doubled, _WINDOW_CM)
        np.testing.assert_array_equal(doubled_windows[:, 2:], windows[:, 2:])


class TestFloor(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(config, "TORTUOSITY_WINDOW_CM", _WINDOW_CM)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_out_and_back_uses_floor_set_at_runtime(self) -> None:
        out_and_back = np.column_stack([np.concatenate([np.arange(6.0), np.arange(4.0, -1.0, -1.0)]), np.zeros(11)])
        error_index = np.zeros(len(out_and_back), dtype=bool)
        for floor in (0.1, 0.5):
            with self.subTest(floor=floor), patch.object(config, "TORTUOSITY_MIN_L_CM", floor):
                windows = _day_windows(out_and_back, error_index)
                self.assertEqual(len(windows), 1)
                self.assertEqual(windows[0, 3], 0.0)
                self.assertAlmostEqual(windows[0, 4], 10.0 / floor)


class TestRuns(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(config, "TORTUOSITY_WINDOW_CM", _WINDOW_CM)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_included_step_runs(self) -> None:
        excluded = np.array([True, False, False, True, True, False, False, False])
        np.testing.assert_array_equal(included_step_runs(excluded), [[1, 3], [5, 8]])
        self.assertEqual(included_step_runs(np.ones(4, dtype=bool)).shape, (0, 2))
        np.testing.assert_array_equal(included_step_runs(np.zeros(3, dtype=bool)), [[0, 3]])

    def _assert_split_at(self, data: np.ndarray, error_index: np.ndarray, expected_runs: list[list[int]]) -> None:
        windows = _day_windows(data, error_index)
        _, excluded = steps_and_exclusions(data, error_index)
        np.testing.assert_array_equal(included_step_runs(excluded), expected_runs)
        _assert_no_window_spans(self, windows, excluded)
        np.testing.assert_allclose(windows[:, 4], 1.0)
        # each run of 30 one-cm steps holds 3 full windows; a bridging window would make 6 or more from one run start
        self.assertEqual(len(windows), 3 * len(expected_runs))
        np.testing.assert_array_equal(np.unique(windows[:, 0])[::3], [run[0] for run in expected_runs])

    def test_nan_row_splits(self) -> None:
        data = _line(63, start=1.0)
        data[31] = np.nan
        self._assert_split_at(data, error_default_points(data), [[0, 30], [32, 62]])

    def test_zero_row_splits(self) -> None:
        data = _line(63, start=1.0)
        data[31] = 0.0
        self._assert_split_at(data, error_default_points(data), [[0, 30], [32, 62]])

    def test_spike_step_splits(self) -> None:
        data = np.concatenate([_line(31, start=1.0), _line(31, start=31.0 + config.SPIKE_THRESHOLD + 1.0)])
        self._assert_split_at(data, error_default_points(data), [[0, 30], [31, 61]])

    def test_missing_frame_on_dense_grid_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "BATCH_SIZE", 63):
            path = pathlib.Path(tmp) / "batch.csv"
            rows = [f"{frame};0;0;true;2;cm;0;0;0;0;{frame + 1};5;0;x" for frame in range(63) if frame != 31]
            path.write_text("\n".join(["meta", "meta", "meta", _HEADER, *rows]) + "\n")
            data = read_batch_csv(str(path))[["xpx", "ypx"]].to_numpy()
        self.assertTrue(np.isnan(data[31]).all())
        self._assert_split_at(data, error_default_points(data), [[0, 30], [32, 62]])


class TestBins(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.data = np.cumsum(rng.uniform(-1.0, 1.0, size=(3000, 2)), axis=0)
        self.data[1500] = np.nan
        self.error_index = error_default_points(self.data)
        self.windows = _day_windows(self.data, self.error_index)

    def test_window_counts_in_bin_of_its_start(self) -> None:
        straddling = self.windows[0]
        split = [int(straddling[0]) + 1]
        self.assertGreater(straddling[1], split[0])
        result = tortuosity(self.data, split, self.error_index)
        self.assertEqual(result[0, 2], 1)
        self.assertAlmostEqual(result[0, 0], straddling[4])
        self.assertEqual(result[1, 2], len(self.windows) - 1)

    def test_weighted_bin_means_equal_whole_day_mean(self) -> None:
        whole = tortuosity(self.data, [], self.error_index)
        self.assertEqual(whole.shape, (1, 3))
        self.assertEqual(whole[0, 2], len(self.windows))
        self.assertAlmostEqual(whole[0, 0], self.windows[:, 4].mean())
        for width in (37, 100, 500):
            with self.subTest(width=width):
                binned = tortuosity(self.data, list(range(width, len(self.data), width)), self.error_index)
                filled = binned[:, 2] > 0
                self.assertEqual(binned[:, 2].sum(), whole[0, 2])
                np.testing.assert_allclose(np.average(binned[filled, 0], weights=binned[filled, 2]), whole[0, 0], rtol=1e-12)

    def test_bin_without_window_start_is_empty(self) -> None:
        result = tortuosity(self.data, [1, 2], self.error_index)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.isnan(result[1, :2]).all())
        self.assertEqual(result[1, 2], 0)
        no_windows = tortuosity(_line(5), [2], np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(no_windows[:, 2], [0, 0])
        self.assertTrue(np.isnan(no_windows[:, :2]).all())

    def test_n_counts_windows(self) -> None:
        binned = tortuosity(self.data, [1000, 2000], self.error_index)
        starts = self.windows[:, 0]
        np.testing.assert_array_equal(binned[:, 2], [(starts < 1000).sum(), ((starts >= 1000) & (starts < 2000)).sum(), (starts >= 2000).sum()])


class TestRawOutput(StepMetricTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = patch.object(config, "TORTUOSITY_WINDOW_CM", _WINDOW_CM)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_raw_windows_match_averaged(self) -> None:
        averaged = metrics.tortuosity_per_interval(fish_ids=[0], all_points=True)["results"]
        raw = metrics.tortuosity_per_interval(fish_ids=[0], all_points=True, every_point=True, write_to_csv=True)["results"]
        columns = ["frame_start", "frame_end", "path_cm", "straight_cm", "tortuosity"]
        for fish_key, days in raw.items():
            for day, table in days.items():
                with self.subTest(fish_key=fish_key, day=day):
                    csv = pd.read_csv(self.results / "tortuosity" / "raw" / f"{fish_key}_{day}.csv")
                    self.assertEqual(list(csv.columns), columns)
                    self.assertGreater(len(csv), 0)
                    self.assertTrue(csv["frame_start"].is_monotonic_increasing and csv["frame_start"].is_unique)
                    self.assertTrue((csv["frame_end"] > csv["frame_start"]).all())
                    self.assertTrue((csv["frame_start"].iloc[1:].to_numpy() >= csv["frame_end"].iloc[:-1].to_numpy()).all())
                    self.assertTrue((csv["path_cm"] >= config.TORTUOSITY_WINDOW_CM).all())
                    mean, _, n = averaged[fish_key][day].iloc[0]
                    self.assertEqual(len(csv), n)
                    self.assertAlmostEqual(csv["tortuosity"].mean(), mean, places=9)

    def test_window_end_is_next_start_within_a_run(self) -> None:
        data = np.concatenate([_line(41, start=1.0), np.full((1, 2), np.nan), _line(41, start=100.0)])
        windows = _day_windows(data, error_default_points(data))
        frames = pd.RangeIndex(5000, 5000 + len(data))
        table = _result_table("tortuosity", windows, frames, data, np.array([5000]), every_point=True)
        np.testing.assert_array_equal(table["frame_start"], [5000, 5010, 5020, 5030, 5042, 5052, 5062, 5072])
        np.testing.assert_array_equal(table["frame_end"], [5010, 5020, 5030, 5040, 5052, 5062, 5072, 5082])


if __name__ == "__main__":
    unittest.main()
