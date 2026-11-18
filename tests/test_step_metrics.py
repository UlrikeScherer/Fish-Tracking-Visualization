"""Tests for the shared step metrics (activity, step_length), their raw output, time intervals and the main entry point."""

import argparse
import contextlib
import dataclasses
import io
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fishproviz
import fishproviz.config as config
import fishproviz.metrics as metrics
from fishproviz.metrics.exploration_trials import exploration_trials
from fishproviz.metrics.metrics import _day_frame, _metric_for_day, interval_split
from fishproviz.metrics.results_to_csv import STEP_METRICS, metric_result_to_csv
from fishproviz.utils import csv_of_the_day, get_camera_pos_keys
from fishproviz.utils.transformation import clear_transformation_cache

import main

_TEST_DATA = pathlib.Path(fishproviz.__file__).parent.parent / "test_data"
_CONFIG_GLOBALS = tuple(field.name for field in dataclasses.fields(config.RootPaths))
_STEP_PER_INTERVAL = (metrics.activity_per_interval, metrics.step_length_per_interval)


def _day_frames(fish_key: str, day: str) -> pd.Index:
    """Day-level frame index of a camera-day as metric_per_interval loads it."""
    camera, position = fish_key.split("_")
    _, batches = csv_of_the_day(camera, day, is_back=position == config.BACK)
    return pd.concat(batches).index


class StepMetricTestCase(unittest.TestCase):
    """Points config at test_data and sends results to a temporary directory."""

    def setUp(self) -> None:
        saved = {name: getattr(config, name) for name in _CONFIG_GLOBALS}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(clear_transformation_cache)
        self.addCleanup(lambda: [setattr(config, name, value) for name, value in saved.items()])
        config.set_config_paths(str(_TEST_DATA))
        config.dir_front = str(_TEST_DATA / "front")
        config.dir_back = str(_TEST_DATA / "back")
        config.area_back = str(_TEST_DATA / "area_config" / "areas_back")
        config.area_front = str(_TEST_DATA / "area_config" / "areas_front")
        self.results = pathlib.Path(tmp.name)
        config.RESULTS_PATH = tmp.name
        config.err_file = str(self.results / "log_error.csv")
        pathlib.Path(config.CONFIG_DATA).mkdir(exist_ok=True)
        self.fish_keys = get_camera_pos_keys()
        self.fish_ids = list(range(len(self.fish_keys)))


class TestStepMetrics(StepMetricTestCase):
    def test_activity_equals_step_length(self) -> None:
        for kwargs in (dict(all_points=True), dict(time_interval=100)):
            with self.subTest(**kwargs):
                act, step = (f(fish_ids=self.fish_ids, **kwargs)["results"] for f in _STEP_PER_INTERVAL)
                for fish_key, days in act.items():
                    for day, table in days.items():
                        pd.testing.assert_frame_equal(table, step[fish_key][day])

    def test_raw_matches_averaged(self) -> None:
        for per_interval in _STEP_PER_INTERVAL:
            averaged = per_interval(fish_ids=self.fish_ids, all_points=True)["results"]
            raw = per_interval(fish_ids=self.fish_ids, all_points=True, every_point=True, write_to_csv=True)
            name = raw["metric_name"]
            self.assertIn(name, STEP_METRICS)
            for fish_key, days in raw["results"].items():
                for day, table in days.items():
                    with self.subTest(metric=name, fish_key=fish_key, day=day):
                        frames = _day_frames(fish_key, day)
                        self.assertEqual(list(table.columns), ["frame", "xpx", "ypx", name, "excluded"])
                        self.assertEqual(len(table), len(frames) - 1)
                        np.testing.assert_array_equal(table["frame"].to_numpy(), frames[:-1])
                        self.assertTrue(table["frame"].is_monotonic_increasing)
                        self.assertTrue(table.loc[table[name].isna(), "excluded"].all())
                        mean, _, n = averaged[fish_key][day].iloc[0]
                        included = table.loc[~table["excluded"], name]
                        self.assertEqual(len(included), n)
                        self.assertAlmostEqual(included.mean(), mean)
                        csv = self.results / name / "raw" / f"{fish_key}_{day}.csv"
                        self.assertEqual(list(pd.read_csv(csv, nrows=0).columns), list(table.columns))

    def test_raw_mode_follows_every_point_only(self) -> None:
        for per_interval in (*_STEP_PER_INTERVAL, metrics.distance_to_wall_per_interval):
            for unaveraged_config in (False, True):
                with self.subTest(metric=per_interval.__name__, unaveraged_config=unaveraged_config), patch.object(config, "UNAVERAGED", unaveraged_config):
                    averaged = per_interval(fish_ids=[0], all_points=True)["results"]
                    raw = per_interval(fish_ids=[0], all_points=True, every_point=True)["results"]
                    for day, table in averaged[self.fish_keys[0]].items():
                        self.assertEqual(table.shape, (1, 3))
                        self.assertGreater(len(raw[self.fish_keys[0]][day]), 1000)

    def test_interval_split(self) -> None:
        time_points, split = interval_split(pd.Index([0]), None)
        np.testing.assert_array_equal(time_points, [0])
        self.assertEqual(split, [])
        time_points, split = interval_split(pd.Index(range(10000, 10010)), 4)
        np.testing.assert_array_equal(time_points, [10000, 10004, 10008])
        self.assertEqual(split, [4, 8])

    def test_interval_split_has_no_degenerate_last_bin(self) -> None:
        # last - first is a multiple of the step: the last frame joins the previous bin, as before
        frames = pd.Index(range(0, 13))
        time_points, split = interval_split(frames, 4)
        np.testing.assert_array_equal(time_points, np.arange(0, 12, 4))
        self.assertEqual(split, [4, 8])

    def test_day_outside_window_is_empty_not_an_error(self) -> None:
        fish_key, day = self.fish_keys[0], "20210911_060000"
        camera, position = fish_key.split("_")
        keys, batches = csv_of_the_day(camera, day, is_back=position == config.BACK)
        late = 24 * 3600 - 1  # window at the end of the day, after every loaded frame
        windows = {f"{day.split('_')[0]}_{fish_key}": (late, late)}
        df = _day_frame(keys, batches, windows, day, fish_key)
        self.assertTrue(df.empty)
        self.assertTrue(_day_frame([], [], None, day, fish_key).empty)
        area_tuple = (fish_key, None)
        for kwargs, width in ((dict(), 3), (dict(include_median=True), 4)):
            table = _metric_for_day(metrics.activity, df, area_tuple, day, step_frames=None, ellipse=None, metric_kwargs=kwargs, every_point=False, out_dim=width)
            self.assertEqual(table.shape, (0, width))
        raw = _metric_for_day(metrics.activity, df, area_tuple, day, step_frames=None, ellipse=None, metric_kwargs=dict(unaveraged=True), every_point=True, out_dim=3)
        self.assertEqual(len(raw), 0)

    def test_median_with_raw_output_raises(self) -> None:
        data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "include_median"):
            metrics.activity(data, [], np.zeros(3, dtype=bool), include_median=True, unaveraged=True)

    def test_raw_mode_raises_for_metrics_without_one(self) -> None:
        for per_interval, name in ((metrics.tortuosity_per_interval, "tortuosity"), (metrics.entropy_per_interval, "entropy")):
            with self.subTest(metric=name), self.assertRaisesRegex(ValueError, f"'{name}'"):
                per_interval(fish_ids=[0], all_points=True, every_point=True, write_to_csv=True)
        self.assertEqual(list(self.results.iterdir()), [])

    def test_hour_column_counts_bins_from_first_loaded_frame(self) -> None:
        with patch.object(config, "MIN_BATCH_IDX", 1):
            metrics.activity_per_interval(fish_ids=[0], time_interval=3600, write_to_csv=True)
        table = pd.read_csv(self.results / "activity" / "hour_activity.csv", sep=config.sep)
        for _, day in table.groupby(["cam_pos", "day"]):
            self.assertEqual(day["df_index"].iloc[0], config.BATCH_SIZE)
            self.assertEqual(list(day["hour"]), list(range(len(day))))

    def test_bins_start_at_first_loaded_frame(self) -> None:
        with patch.object(config, "MIN_BATCH_IDX", 1):
            first = config.BATCH_SIZE
            whole = metrics.activity_per_interval(fish_ids=[0], all_points=True)["results"][self.fish_keys[0]]
            binned = metrics.activity_per_interval(fish_ids=[0], time_interval=100)["results"][self.fish_keys[0]]
            for day, table in whole.items():
                self.assertEqual(list(table.index), [first])
                self.assertEqual(binned[day].index[0], first)
                self.assertGreater(binned[day].iloc[0, 2], 0)
                self.assertEqual(binned[day].iloc[:, 2].sum(), table.iloc[0, 2])


class TestCsvWriter(StepMetricTestCase):
    def test_no_data_raises_naming_metric(self) -> None:
        for results in ({}, {"12345678_front": {}}):
            with self.assertRaisesRegex(ValueError, "'activity'"):
                metric_result_to_csv(results=results, metric_name="activity", all_points=True)

    def test_fish_with_only_empty_days_raises_naming_fish(self) -> None:
        day = pd.DataFrame([[1.0, 0.5, 10]], index=[0])
        empty = pd.DataFrame(np.empty([0, 3]))
        results = {"12345678_front": {"20210911_060000": day}, "87654321_back": {"20210911_060000": empty}}
        with self.assertRaisesRegex(ValueError, "'87654321_back'.*'activity'"):
            metric_result_to_csv(results=results, metric_name="activity", all_points=True)
        with self.assertRaisesRegex(ValueError, "No fish produced any data for metric 'activity'"):
            metric_result_to_csv(results={"87654321_back": {"20210911_060000": empty}}, metric_name="activity", all_points=True)

    def test_fish_without_days_raises_naming_fish(self) -> None:
        day = pd.DataFrame([[1.0, 0.5, 10]], index=[0])
        results = {"12345678_front": {"20210911_060000": day}, "87654321_back": {}}
        with self.assertRaisesRegex(ValueError, "'87654321_back'.*'activity'"):
            metric_result_to_csv(results=results, metric_name="activity", all_points=True)


class TestMain(StepMetricTestCase):
    def _parse_ti(self, value: str) -> object:
        return main.build_parser().parse_args(["activity", "-ti", value]).time_interval

    def test_time_interval_parsing(self) -> None:
        self.assertEqual(self._parse_ti("30"), 30)
        self.assertEqual(self._parse_ti("hour"), 3600)
        self.assertEqual(self._parse_ti("day"), "day")
        self.assertIsNone(main.build_parser().parse_args(["activity"]).time_interval)
        for value in ("0", "-1", "29", "week"):
            with self.subTest(value=value), self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(io.StringIO()):
                self._parse_ti(value)
            self.assertEqual(ctx.exception.code, 2)
            with self.assertRaises(argparse.ArgumentTypeError):
                main.parse_time_interval(value)

    def test_ti_day_writes_one_row_per_camera_day(self) -> None:
        with patch.object(config, "UNAVERAGED", False), contextlib.redirect_stdout(io.StringIO()):
            main.main("activity", time_interval="day")
            main.main("activity")
        day_file = pd.read_csv(self.results / "activity" / "day_activity.csv", sep=config.sep, index_col=0)
        default_file = pd.read_csv(self.results / "activity" / "activity.csv", sep=config.sep, index_col=0)
        n_camera_days = len(list((_TEST_DATA / "front").glob("*/*"))) + len(list((_TEST_DATA / "back").glob("*/*")))
        self.assertEqual(len(day_file), n_camera_days)
        self.assertNotIn("hour", day_file.columns)
        self.assertFalse(day_file.duplicated(["cam_pos", "day"]).any())
        pd.testing.assert_frame_equal(day_file, default_file)

    def test_hour_column_only_when_binning_by_hour(self) -> None:
        metrics.activity_per_interval(fish_ids=[0], time_interval=3600, write_to_csv=True)
        metrics.activity_per_interval(fish_ids=[0], time_interval=3600, all_points=True, write_to_csv=True)
        self.assertIn("hour", pd.read_csv(self.results / "activity" / "hour_activity.csv", sep=config.sep).columns)
        self.assertNotIn("hour", pd.read_csv(self.results / "activity" / "activity.csv", sep=config.sep).columns)

    def test_all_with_include_median_applies_to_activity_only(self) -> None:
        with patch.object(config, "UNAVERAGED", False), contextlib.redirect_stdout(io.StringIO()):
            main.main("all", fish_id="0", include_median=True)
        activity_csv = pd.read_csv(self.results / "activity" / "activity.csv", sep=config.sep)
        step_csv = pd.read_csv(self.results / "step_length" / "step_length.csv", sep=config.sep)
        self.assertEqual(list(activity_csv.columns)[-4:], ["mean", "std", "median", "num_valid_points"])
        self.assertEqual(list(step_csv.columns)[-3:], ["mean", "std", "num_valid_points"])
        for name in ("tortuosity", "entropy", "distance_to_wall", "turning_angle", "absolute_angles", "turning_angle_streak_length"):
            self.assertTrue(any((self.results / name).glob("*.csv")), name)

    def test_include_median_rejected_for_other_programs(self) -> None:
        with patch.object(config, "UNAVERAGED", False), self.assertRaisesRegex(ValueError, "include_median"):
            main.main("step_length", include_median=True)
        self.assertFalse((self.results / "step_length").exists())

    def test_include_median_rejected_with_raw_output(self) -> None:
        for program in ("activity", "all"):
            with self.subTest(program=program), patch.object(config, "UNAVERAGED", True), self.assertRaisesRegex(ValueError, "include_median.*UNAVERAGED"):
                main.main(program, include_median=True)
        self.assertEqual(list(self.results.iterdir()), [])

    def test_raw_mode_validated_before_running(self) -> None:
        with patch.object(config, "UNAVERAGED", True):
            with self.assertRaisesRegex(ValueError, "'tortuosity'"):
                main.main("tortuosity")
            with self.assertRaisesRegex(ValueError, r"\['tortuosity', 'entropy'\]"):
                main.main("all")
        self.assertEqual(list(self.results.iterdir()), [])


class TestTrialTimes(StepMetricTestCase):
    def test_writes_only_its_block(self) -> None:
        trials_dir = self.results / "trials"
        trials_dir.mkdir()
        trials_csv = trials_dir / "trial_times.csv"
        pd.DataFrame(
            {
                "date": ["11.09.21", "12.09.21", "11.09.21"],
                "block": [1, 1, 2],
                "trial_start": ["06:00", "06:00", "06:00"],
                "trial_end": ["06:10", "06:10", "06:10"],
            }
        ).to_csv(trials_csv, sep=config.sep, index=False)
        exploration_trials(str(trials_csv))
        mean = pd.read_csv(self.results / "trials_mean.csv", sep=config.sep)
        ndf = pd.read_csv(self.results / "trials_ndf.csv", sep=config.sep)
        other_block = mean["block"] == 2
        for fish_key in self.fish_keys:
            self.assertTrue(mean.loc[other_block, fish_key].isna().all(), fish_key)
            self.assertTrue(ndf.loc[other_block, fish_key].isna().all(), fish_key)
        filled = mean.loc[~other_block, self.fish_keys].notna().sum().sum()
        self.assertEqual(filled, len(self.fish_keys))
        self.assertTrue((ndf.loc[~other_block, self.fish_keys].fillna(0).to_numpy() <= 10 * 60 * config.FRAMES_PER_SECOND).all())
        self.assertGreater(ndf.loc[~other_block, self.fish_keys].sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
