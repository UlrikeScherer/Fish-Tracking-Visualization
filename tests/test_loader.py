"""Tests for the batch loader: every frame gap must surface as NaN on a dense, absolute frame grid."""

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fishproviz.config as config
from fishproviz.metrics.compute_primitives import compute_step_lengths
from fishproviz.metrics.metrics import update_filter_three_points, update_filter_two_points
from fishproviz.utils import all_error_filters, csv_of_the_day

_CAM = "12345678"
_DAY = "20211113_060000"
_BATCH_SIZE = 10
_MIN_BATCH_IDX = 0
_MAX_BATCH_IDX = 2
_HEADER = "FRAME;MillisecsByFPS;objectName;valid;id;coordinateUnit;x;y;rad;deg;xpx;ypx;time;timeString"
_BLANK_FRAME = 3  # blank row in batch 0
_MISSING_KEY = "000001"  # no file for this batch
_SHORT_KEY = "000002"  # file ends before BATCH_SIZE
_SHORT_LAST_FRAME = 7


def _px(frame: int) -> tuple[float, float]:
    """Synthetic fish position: a slow diagonal walk well below the spike threshold."""
    return 100.0 + frame, 200.0 + frame


def _row(frame: int) -> str:
    xpx, ypx = _px(frame)
    ms = frame * 40.0
    return f"{frame};{ms:.6f};0;true;2;cm;{xpx / 10};{ypx / 10};1.57;90.0;{xpx:.0f};{ypx:.0f};{1636805810000 + int(ms)};Sat Nov 13 12:16:50 2021"


def _blank_row(frame: int) -> str:
    return f"{frame};{frame * 40.0:.6f};;;;;;;;;;;;"


def _write_batch(directory: pathlib.Path, key: str, lines: list[str]) -> None:
    day_dir = directory / _CAM / f"{_DAY}.{_CAM}"
    day_dir.mkdir(parents=True, exist_ok=True)
    comments = ["# Source name: synthetic", "# Source FPS: 25.", "# Generation time: 2021-11-13T12:16:50.000"]
    (day_dir / f"{_CAM}_{_DAY}.{_CAM}_{key}_2021-11-13T12_16_50_3484.csv").write_text("\n".join([*comments, _HEADER, *lines]) + "\n")


class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        _write_batch(root, "000000", [_blank_row(f) if f == _BLANK_FRAME else _row(f) for f in range(_BATCH_SIZE)])
        _write_batch(root, _SHORT_KEY, [_row(f) for f in range(_SHORT_LAST_FRAME + 1)])
        patcher = patch.multiple(
            config,
            dir_front=str(root),
            BATCH_SIZE=_BATCH_SIZE,
            MIN_BATCH_IDX=_MIN_BATCH_IDX,
            MAX_BATCH_IDX=_MAX_BATCH_IDX,
            AREA_FILTER=0,
            DIRT_FILTER=0,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.keys, self.batches = csv_of_the_day(_CAM, _DAY, is_back=False)
        self.day = pd.concat(self.batches)

    def test_keys_contiguous_and_batches_dense(self) -> None:
        self.assertEqual(self.keys, ["000000", "000001", "000002"])
        for batch in self.batches:
            self.assertEqual(len(batch), _BATCH_SIZE)
            np.testing.assert_array_equal(batch.FRAME.to_numpy(), np.arange(_BATCH_SIZE))

    def test_absolute_index(self) -> None:
        for key, batch in zip(self.keys, self.batches):
            np.testing.assert_array_equal(batch.index.to_numpy(), batch.FRAME.to_numpy() + int(key) * _BATCH_SIZE)
        np.testing.assert_array_equal(self.day.index.to_numpy(), np.arange(len(self.keys) * _BATCH_SIZE))
        self.assertFalse(self.day.FRAME.isna().any())

    def test_blank_row_is_nan(self) -> None:
        self.assertTrue(self.day.loc[_BLANK_FRAME, ["xpx", "ypx"]].isna().all())
        valid = self.batches[0].drop(index=_BLANK_FRAME)
        self.assertFalse(valid[["xpx", "ypx"]].isna().any().any())
        self.assertEqual(tuple(self.day.loc[_BLANK_FRAME + 1, ["xpx", "ypx"]]), _px(_BLANK_FRAME + 1))

    def test_missing_batch_is_all_nan(self) -> None:
        missing = self.batches[self.keys.index(_MISSING_KEY)]
        self.assertTrue(missing[["x", "y", "xpx", "ypx", "time"]].isna().all().all())

    def test_short_file_padded(self) -> None:
        short = self.batches[self.keys.index(_SHORT_KEY)]
        self.assertFalse(short.loc[short.FRAME <= _SHORT_LAST_FRAME, ["xpx", "ypx"]].isna().any().any())
        self.assertTrue(short.loc[short.FRAME > _SHORT_LAST_FRAME, ["xpx", "ypx"]].isna().all().all())

    def test_gap_excluded_from_steps_and_angles(self) -> None:
        data = self.batches[0][["xpx", "ypx"]].to_numpy()
        err = all_error_filters(data, None)
        steps = compute_step_lengths(data)
        step_excluded = update_filter_two_points(steps, err)
        triplet_excluded = update_filter_three_points(steps, err)
        # step i joins frames (i, i+1); triplet j spans frames (j, j+1, j+2)
        expected_steps = np.array([_BLANK_FRAME in (i, i + 1) for i in range(len(steps))])
        expected_triplets = np.array([_BLANK_FRAME in (j, j + 1, j + 2) for j in range(len(steps) - 1)])
        np.testing.assert_array_equal(step_excluded, expected_steps)
        np.testing.assert_array_equal(triplet_excluded, expected_triplets)
        self.assertFalse(np.isnan(steps[~step_excluded]).any())

    def test_no_batch_files_returns_empty(self) -> None:
        self.assertEqual(csv_of_the_day(_CAM, "20211114_060000", is_back=False), ([], []))


if __name__ == "__main__":
    unittest.main()
