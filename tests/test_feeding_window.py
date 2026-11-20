"""Feeding and trial windows are looked up with the right key: date for feeding, date_<cam_pos> for novel object and sociability."""

import unittest
import warnings
from unittest.mock import patch

import pandas as pd

import fishproviz.config as config
import fishproviz.metrics as metrics
from fishproviz.metrics.results_to_csv import DISTANCE_TO_WALL_COLUMNS, num_datapoints
from tests.test_step_metrics import StepMetricTestCase

_DAY = "20210911_060000"
_WINDOW_START, _WINDOW_END = "06:10", "06:20"  # recording starts 06:00:00, so frames 3000 to 6000
_WINDOW_FRAMES = 10 * 60 * 5 + 1
_MISSING_WINDOW_WARNING = "No start or end time"


class FeedingWindowTestCase(StepMetricTestCase):
    def setUp(self) -> None:
        super().setUp()
        saved = (config.SERVER_FEEDING_TIMES_FILE, config.SERVER_FEEDING_TIMES_SEP)
        self.addCleanup(lambda: setattr(config, "SERVER_FEEDING_TIMES_FILE", saved[0]))
        self.addCleanup(lambda: setattr(config, "SERVER_FEEDING_TIMES_SEP", saved[1]))
        config.SERVER_FEEDING_TIMES_SEP = ";"
        self.times_file = self.results / "times.csv"
        config.SERVER_FEEDING_TIMES_FILE = str(self.times_file)

    def _valid_points(self, **flags: bool) -> tuple[int, list[str]]:
        """Valid points n of the first fish's test day and the missing-window warnings raised on the way."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = metrics.distance_to_wall_per_interval(fish_ids=self.fish_ids[:1], all_points=True, **flags)["results"][self.fish_keys[0]][_DAY]
        return int(table.iloc[0, DISTANCE_TO_WALL_COLUMNS.index(num_datapoints)]), [str(w.message) for w in caught if _MISSING_WINDOW_WARNING in str(w.message)]

    def test_feeding_window_is_applied(self) -> None:
        pd.DataFrame({"day": ["2021-09-11"], "time_in_start": [_WINDOW_START], "time_out_stop": [_WINDOW_END]}).to_csv(self.times_file, sep=";", index=False)
        windowed, warned = self._valid_points(is_feeding=True)
        self.assertEqual(warned, [])
        config.SERVER_FEEDING_TIMES_FILE = str(self.results / "absent.csv")
        unwindowed, _ = self._valid_points(is_feeding=True)
        self.assertGreater(windowed, 0)
        self.assertLessEqual(windowed, _WINDOW_FRAMES)
        self.assertLess(windowed, unwindowed)

    def test_trial_window_keeps_tank_id(self) -> None:
        pd.DataFrame({"date": ["2021-09-11"], "tank_ID": [self.fish_keys[0]], "trial_start": [_WINDOW_START], "trial_end": [_WINDOW_END]}).to_csv(self.times_file, sep=";", index=False)
        for flag in ("is_novel_object", "is_sociability"):
            with self.subTest(flag):
                windowed, warned = self._valid_points(**{flag: True})
                self.assertEqual(warned, [])
                self.assertGreater(windowed, 0)
                self.assertLessEqual(windowed, _WINDOW_FRAMES)

    def test_day_frame_receives_the_tank_id_only_for_trials(self) -> None:
        for flags, expected in ((dict(is_feeding=True), None), (dict(is_novel_object=True), self.fish_keys[0]), (dict(is_sociability=True), self.fish_keys[0])):
            with self.subTest(**flags), patch("fishproviz.metrics.metrics._day_frame", wraps=metrics.metrics._day_frame) as spy, patch("fishproviz.metrics.metrics.feeding_times_start_end_dict", return_value=None):
                metrics.distance_to_wall_per_interval(fish_ids=self.fish_ids[:1], all_points=True, **flags)
                self.assertEqual({call.args[4] for call in spy.call_args_list}, {expected})


if __name__ == "__main__":
    unittest.main()
