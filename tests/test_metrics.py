import pathlib
import time
import unittest

import numpy as np
import pandas as pd

import fishproviz
import fishproviz.config as config
import fishproviz.metrics as metrics
import fishproviz.utils as utils

_REPO = pathlib.Path(fishproviz.__file__).parent.parent
_TEST_DATA = _REPO / "test_data"
_REF = _TEST_DATA / "results"
_RTOL = 1e-6


def _cmp(subdir: str, filename: str) -> None:
    ref = pd.read_csv(_REF / subdir / filename, sep=";", index_col=0)
    gen = pd.read_csv(pathlib.Path(config.RESULTS_PATH) / subdir / filename, sep=";", index_col=0)
    pd.testing.assert_frame_equal(gen, ref, rtol=_RTOL, check_exact=False)


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.set_config_paths(str(_TEST_DATA))
        config.dir_front = str(_TEST_DATA / "front")
        config.dir_back = str(_TEST_DATA / "back")
        config.area_back = str(_TEST_DATA / "area_config" / "areas_back")
        config.area_front = str(_TEST_DATA / "area_config" / "areas_front")
        config.RESULTS_PATH = str(_TEST_DATA / "results_generated")
        config.err_file = str(_TEST_DATA / "results_generated" / "log_error.csv")
        config.create_directories()

        cls.fish_keys = utils.get_camera_pos_keys()
        cls.fish_ids = np.arange(len(cls.fish_keys))

    def test_activity(self):
        t = time.time()
        metrics.activity_per_interval(fish_ids=self.fish_ids, time_interval=100, write_to_csv=True)
        metrics.activity_per_interval(fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True)
        print("activity: %.02fs" % (time.time() - t))
        _cmp("activity", "100_sec_activity.csv")
        _cmp("activity", "hour_activity.csv")

        t = time.time()
        metrics.distance_to_wall_per_interval(fish_ids=self.fish_ids, time_interval=100, write_to_csv=True)
        metrics.distance_to_wall_per_interval(fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True)
        print("distance_to_wall: %.02fs" % (time.time() - t))
        _cmp("distance_to_wall", "100_sec_distance_to_wall.csv")
        _cmp("distance_to_wall", "hour_distance_to_wall.csv")

        t = time.time()
        metrics.turning_angle_per_interval(fish_ids=self.fish_ids, time_interval=100, write_to_csv=True)
        metrics.turning_angle_per_interval(fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True)
        print("turning_angle: %.02fs" % (time.time() - t))
        _cmp("turning_angle", f"100_sec_turning_angle_skip{config.TANGLE_N_SKIP}.csv")
        _cmp("turning_angle", f"hour_turning_angle_skip{config.TANGLE_N_SKIP}.csv")

    def test_turning_angle_unaveraged_matches_averaged(self):
        for metric in (metrics.turning_angle_per_interval, metrics.absolute_angle_per_interval):
            with self.subTest(metric=metric.__name__):
                averaged = metric(fish_ids=self.fish_ids, all_points=True)["results"]
                raw = metric(fish_ids=self.fish_ids, all_points=True, every_point=True, write_to_csv=True)
                name = raw["metric_name"]
                for fish_key, days in raw["results"].items():
                    for day, table in days.items():
                        self.assertEqual(list(table.columns), ["frame", "xpx", "ypx", name, "excluded"])
                        self.assertTrue(table["frame"].is_monotonic_increasing)
                        included = table.loc[~table["excluded"], name].dropna()
                        mean, _, n = averaged[fish_key][day].iloc[0]
                        self.assertAlmostEqual(included.mean(), mean)
                        self.assertEqual(len(included), n)
                        csv = pathlib.Path(config.RESULTS_PATH) / name / "raw" / f"{fish_key}_{day}_skip{config.TANGLE_N_SKIP}.csv"
                        self.assertEqual(list(pd.read_csv(csv, nrows=0).columns), list(table.columns))


if __name__ == "__main__":
    unittest.main()
