"""Tests for the entropy grid (cell count over the compartment bounding box plus a margin), its exclusion count and its CSV output."""

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fishproviz.config as config
import fishproviz.metrics as metrics
from fishproviz.metrics.compute_primitives import entropy_for_chunk, entropy_heatmap
from fishproviz.metrics.metrics import _averaged_width, _metric_for_day
from fishproviz.metrics.results_to_csv import ENTROPY_COLUMNS, metric_result_to_csv
from fishproviz.utils.transformation import _area_functions, clear_transformation_cache

_REPO = pathlib.Path(__file__).resolve().parent.parent
_TEST_DATA = _REPO / "test_data"
# 800 px square: with a 100 px margin and 10 cells per side the grid spans [-100, 900] in cells of 100 px
_SQUARE = np.array([[0, 0], [800, 0], [800, 800], [0, 800], [0, 0]], dtype=np.float64)
_BINS, _MARGIN = 10, 100


def _points(*xy_counts: tuple[float, float, int]) -> np.ndarray:
    """count copies of every (x, y)."""
    return np.array([[x, y] for x, y, count in xy_counts for _ in range(count)], dtype=np.float64).reshape((-1, 2))


class EntropyConfigCase(unittest.TestCase):
    """Sets the entropy grid config for the test and restores it afterwards."""

    def setUp(self) -> None:
        patcher = patch.multiple(config, ENTROPY_BINS=_BINS, ENTROPY_MARGIN_PX=_MARGIN)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestKernel(EntropyConfigCase):
    def test_one_cell_is_zero(self) -> None:
        self.assertEqual(entropy_for_chunk(_points((10, 10, 5), (60, 90, 2)), _SQUARE), (0.0, 7, 0))

    def test_two_equal_cells_is_ln2(self) -> None:
        h, n, n_out = entropy_for_chunk(_points((10, 10, 3), (310, 10, 3)), _SQUARE)
        self.assertAlmostEqual(h, np.log(2))
        self.assertEqual((n, n_out), (6, 0))

    def test_every_cell_equal_is_ln_of_cell_count(self) -> None:
        centres = -50 + 100 * np.arange(_BINS, dtype=np.float64)
        points = np.array([[x, y] for x in centres for y in centres for _ in range(2)])
        hist, n_out = entropy_heatmap(points, _SQUARE)
        np.testing.assert_array_equal(hist, np.full((_BINS, _BINS), 2.0))
        h, n, n_out = entropy_for_chunk(points, _SQUARE)
        self.assertAlmostEqual(h, np.log(_BINS**2))
        self.assertEqual((n, n_out), (2 * _BINS**2, 0))

    def test_no_points(self) -> None:
        h, n, n_out = entropy_for_chunk(np.empty((0, 2)), _SQUARE)
        self.assertTrue(np.isnan(h))
        self.assertEqual((n, n_out), (0, 0))

    def test_points_beyond_margin_are_counted_not_histogrammed(self) -> None:
        points = _points((10, 10, 4), (901, 10, 1), (10, -101, 1), (5000, 5000, 1))
        hist, n_out = entropy_heatmap(points, _SQUARE)
        self.assertEqual((hist.sum(), n_out), (4, 3))
        self.assertEqual(entropy_for_chunk(points, _SQUARE), (0.0, 4, 3))

    def test_all_out_of_range(self) -> None:
        h, n, n_out = entropy_for_chunk(_points((5000, 5000, 3), (-500, 10, 2)), _SQUARE)
        self.assertTrue(np.isnan(h))
        self.assertEqual((n, n_out), (0, 5))

    def test_outer_range_edge_is_inclusive(self) -> None:
        points = _points((900, 900, 1), (-100, -100, 1), (900, -100, 1), (900.001, 10, 1), (10, -100.001, 1))
        hist, n_out = entropy_heatmap(points, _SQUARE)
        self.assertEqual((hist[-1, -1], hist[0, 0], hist[-1, 0]), (1, 1, 1))
        self.assertEqual(n_out, 2)
        self.assertEqual(hist.sum() + n_out, len(points))

    def test_counts_always_add_up(self) -> None:
        rng = np.random.default_rng(0)
        points = rng.uniform(-300, 1100, size=(5000, 2))
        hist, n_out = entropy_heatmap(points, _SQUARE)
        self.assertEqual(hist.sum() + n_out, len(points))

    def test_grid_follows_config_at_runtime(self) -> None:
        points = _points((110, 10, 3), (210, 10, 3), (950, 10, 1))
        self.assertEqual(entropy_heatmap(points, _SQUARE)[0].shape, (_BINS, _BINS))
        with patch.object(config, "ENTROPY_BINS", 5):
            # cells of 200 px: both clusters share one cell
            hist, _ = entropy_heatmap(points, _SQUARE)
            self.assertEqual(hist.shape, (5, 5))
            self.assertEqual(entropy_for_chunk(points, _SQUARE), (0.0, 6, 1))
        with patch.object(config, "ENTROPY_MARGIN_PX", 200):
            self.assertEqual(entropy_for_chunk(points, _SQUARE)[1:], (7, 0))

    def test_threshold_area_px_has_no_effect(self) -> None:
        points = _points((10, 10, 3), (310, 450, 2), (1500, 10, 1))
        expected = entropy_for_chunk(points, _SQUARE)
        for threshold in (0, 50, 5000):
            with self.subTest(threshold=threshold), patch.object(config, "THRESHOLD_AREA_PX", threshold):
                self.assertEqual(entropy_for_chunk(points, _SQUARE), expected)

    def test_degenerate_range_raises(self) -> None:
        line = np.array([[0, 0], [800, 0], [0, 0]], dtype=np.float64)
        with patch.object(config, "ENTROPY_MARGIN_PX", 0), self.assertRaisesRegex(ValueError, "ENTROPY_MARGIN_PX=0"):
            entropy_heatmap(_points((10, 0, 1)), line)


class TestEntropyPerBin(EntropyConfigCase):
    def test_error_rows_neither_histogrammed_nor_counted(self) -> None:
        data = _points((10, 10, 3), (310, 10, 3), (5000, 5000, 2))
        error_index = np.array([False, False, False, True, True, False, False, True])
        h, n, n_out = metrics.entropy(data, [], error_index, ("cam_front", _SQUARE))[0]
        self.assertAlmostEqual(h, -(3 / 4) * np.log(3 / 4) - (1 / 4) * np.log(1 / 4))
        self.assertEqual((n, n_out), (4, 1))

    def test_bins_with_partial_last_and_empty_bin(self) -> None:
        # bins of 5 rows, the last holding 3: [one cell], [two cells + one out], [all errors], [partial]
        data = _points((10, 10, 5), (10, 10, 2), (310, 10, 2), (5000, 10, 1), (0, 0, 5), (10, 10, 1), (9000, 9000, 2))
        error_index = np.zeros(len(data), dtype=bool)
        error_index[10:15] = True
        result = metrics.entropy(data, [5, 10, 15], error_index, ("cam_front", _SQUARE))
        self.assertEqual(result.shape, (4, 3))
        np.testing.assert_array_equal(result[0], [0, 5, 0])
        self.assertAlmostEqual(result[1, 0], np.log(2))
        np.testing.assert_array_equal(result[1, 1:], [4, 1])
        np.testing.assert_array_equal(result[2], [np.nan, 0, 0])
        np.testing.assert_array_equal(result[3], [0, 1, 2])


_CAM = "23442333"  # has an area polygon in test_data/area_config
_DAY = "20211113_060000"
_BATCH_SIZE = 10
_HEADER = "FRAME;MillisecsByFPS;objectName;valid;id;coordinateUnit;x;y;rad;deg;xpx;ypx;time;timeString"
_INSIDE_A, _INSIDE_B, _OUT, _ERROR = (1000, 1000), (2000, 2000), (9000, 9000), (0, 0)
# day frames 0..19 in bins of 5: [A x5], [A A B B OUT], [ERROR x5], [OUT x5]
_DAY_POSITIONS = [_INSIDE_A] * 7 + [_INSIDE_B] * 2 + [_OUT] + [_ERROR] * 5 + [_OUT] * 5


def _write_batch(root: pathlib.Path, key: int, positions: list[tuple[int, int]]) -> None:
    day_dir = root / _CAM / f"{_DAY}.{_CAM}"
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = [f"{frame};{frame * 40.0:.6f};0;true;2;cm;0;0;1.57;90.0;{x};{y};{1636805810000 + frame * 40};Sat Nov 13 12:16:50 2021" for frame, (x, y) in enumerate(positions)]
    (day_dir / f"{_CAM}_{_DAY}.{_CAM}_{key:06d}_2021-11-13T12_16_50_3484.csv").write_text("\n".join(["# Source name: synthetic", "# Source FPS: 25.", "# Generation time: 2021-11-13T12:16:50.000", _HEADER, *rows]) + "\n")


class TestWriter(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        front, back, self.results = root / "front", root / "back", root / "results"
        back.mkdir()
        for key in range(2):
            _write_batch(front, key, _DAY_POSITIONS[key * _BATCH_SIZE : (key + 1) * _BATCH_SIZE])
        saved_paths = {name: getattr(config, name) for name in ("CONFIG_DATA", "RESULTS_PATH", "err_file")}
        self.addCleanup(lambda: [setattr(config, name, value) for name, value in saved_paths.items()])
        patcher = patch.multiple(
            config,
            dir_front=str(front),
            dir_back=str(back),
            area_front=str(_TEST_DATA / "area_config" / "areas_front"),
            area_back=str(_TEST_DATA / "area_config" / "areas_back"),
            CONFIG_DATA=str(root / "config_data"),
            RESULTS_PATH=str(self.results),
            err_file=str(root / "log_error.csv"),
            BATCH_SIZE=_BATCH_SIZE,
            MIN_BATCH_IDX=0,
            MAX_BATCH_IDX=1,
            FRAMES_PER_SECOND=5,
            AREA_FILTER=0,
            DIRT_FILTER=0,
            ENTROPY_BINS=18,
            ENTROPY_MARGIN_PX=50,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        (root / "config_data").mkdir()
        clear_transformation_cache()
        self.addCleanup(clear_transformation_cache)
        self.area = _area_functions()(f"{_CAM}_front")

    def _read(self, file_name: str) -> pd.DataFrame:
        table = pd.read_csv(self.results / "entropy" / file_name, sep=config.sep, index_col=0)
        self.assertEqual(list(table.columns[:6]), ["cam_pos", "day", "df_index", "entropy", "num_valid_points", "num_out_of_range"])
        for column in ("num_valid_points", "num_out_of_range"):
            self.assertTrue(pd.api.types.is_integer_dtype(table[column]), column)
        return table

    def test_whole_day(self) -> None:
        metrics.entropy_per_interval(all_points=True, write_to_csv=True)
        table = self._read("entropy.csv")
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        self.assertEqual((row.cam_pos, str(row.day), row.df_index), (f"{_CAM}_front", _DAY, 0))
        self.assertAlmostEqual(row.entropy, entropy_for_chunk(np.array([_INSIDE_A] * 7 + [_INSIDE_B] * 2, dtype=np.float64), self.area)[0])
        self.assertAlmostEqual(row.entropy, -(7 / 9) * np.log(7 / 9) - (2 / 9) * np.log(2 / 9))
        self.assertEqual((row.num_valid_points, row.num_out_of_range), (9, 6))

    def test_binned(self) -> None:
        metrics.entropy_per_interval(time_interval=1, write_to_csv=True)
        (written,) = [path.name for path in (self.results / "entropy").glob("*.csv")]
        table = self._read(written)
        self.assertEqual(table.df_index.tolist(), [0, 5, 10, 15])
        self.assertEqual(table.num_valid_points.tolist(), [5, 4, 0, 0])
        self.assertEqual(table.num_out_of_range.tolist(), [0, 1, 0, 5])
        np.testing.assert_allclose(table.entropy.to_numpy(), [0, np.log(2), np.nan, np.nan])

    def test_empty_day_has_the_entropy_column_count(self) -> None:
        empty_day = _metric_for_day(
            metrics.entropy,
            pd.DataFrame(columns=["xpx", "ypx"]),
            (f"{_CAM}_front", self.area),
            _DAY,
            step_frames=None,
            ellipse=None,
            metric_kwargs={},
            every_point=False,
            out_dim=_averaged_width("entropy", False),
        )
        self.assertEqual(empty_day.shape, (0, len(ENTROPY_COLUMNS)))
        full_day = pd.DataFrame([[0.5, 9, 6]], columns=range(3), index=[0])
        results = {f"{_CAM}_front": {"20211113_000000": empty_day, _DAY: full_day}}
        metric_result_to_csv(results, "entropy", all_points=True)
        table = self._read("entropy.csv")
        self.assertEqual(table.day.astype(str).tolist(), [_DAY])
        self.assertEqual(table.iloc[0][ENTROPY_COLUMNS].tolist(), [0.5, 9, 6])


if __name__ == "__main__":
    unittest.main()
