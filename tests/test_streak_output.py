import pathlib
import tempfile
import unittest

import pandas as pd

import fishproviz
import fishproviz.config as config
import fishproviz.metrics as metrics
from fishproviz.metrics.results_to_csv import TURNING_ANGLE_STREAK_LENGTH

_TEST_DATA = pathlib.Path(fishproviz.__file__).parent.parent / "test_data"
_CONFIG_GLOBALS = ("RESULTS_PATH", "dir_front", "dir_back", "area_front", "area_back", "err_file")


class TestStreakOutput(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {name: getattr(config, name) for name in _CONFIG_GLOBALS}
        self._tmp = tempfile.TemporaryDirectory()
        config.set_config_paths(str(_TEST_DATA))
        config.dir_front = str(_TEST_DATA / "front")
        config.dir_back = str(_TEST_DATA / "back")
        config.area_back = str(_TEST_DATA / "area_config" / "areas_back")
        config.area_front = str(_TEST_DATA / "area_config" / "areas_front")
        config.RESULTS_PATH = self._tmp.name
        config.err_file = str(pathlib.Path(self._tmp.name) / "log_error.csv")

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(config, name, value)
        self._tmp.cleanup()

    def test_averaged_streak_filename_and_columns(self) -> None:
        metrics.turning_angle_streak_length_per_interval(fish_ids=[0], all_points=True, write_to_csv=True)
        path = pathlib.Path(self._tmp.name) / TURNING_ANGLE_STREAK_LENGTH / f"{TURNING_ANGLE_STREAK_LENGTH}_skip{config.TANGLE_N_SKIP}.csv"
        self.assertTrue(path.exists(), path)
        df = pd.read_csv(path, sep=config.sep)
        self.assertEqual(list(df.columns)[-3:], ["mean", "std", "num_valid_points"])
        self.assertTrue(pd.api.types.is_integer_dtype(df["num_valid_points"]))

    def test_raw_streak_files(self) -> None:
        metrics.turning_angle_streak_length_per_interval(fish_ids=[0], all_points=True, every_point=True, write_to_csv=True)
        files = sorted((pathlib.Path(self._tmp.name) / TURNING_ANGLE_STREAK_LENGTH / "raw").glob("*.csv"))
        self.assertTrue(files)
        for path in files:
            self.assertTrue(path.name.endswith(f"_skip{config.TANGLE_N_SKIP}.csv"), path.name)
            df = pd.read_csv(path)
            self.assertEqual(list(df.columns), ["frame_start", "frame_end", "side", "length"])
            for column in df.columns:
                self.assertTrue(pd.api.types.is_integer_dtype(df[column]), column)
            self.assertTrue(df["frame_start"].is_monotonic_increasing)
            self.assertTrue((df["frame_end"] >= df["frame_start"]).all())
            self.assertTrue(df["side"].isin([-1, 1]).all())

    def test_absolute_angles_averaged_filename(self) -> None:
        metrics.absolute_angle_per_interval(fish_ids=[0], all_points=True, write_to_csv=True)
        path = pathlib.Path(self._tmp.name) / "absolute_angles" / f"absolute_angles_skip{config.TANGLE_N_SKIP}.csv"
        self.assertTrue(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
