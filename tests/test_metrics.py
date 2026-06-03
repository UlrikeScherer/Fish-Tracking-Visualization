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

    def test_compute_turning_angles(self):
        points1 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        expected_angles1 = np.array([np.pi * (3 / 4), np.pi / 2, np.pi / 2])
        assert np.allclose(metrics.compute_turning_angles(points1), expected_angles1), "Test 1 failed"

        points2 = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        expected_angles2 = np.array([0, np.pi * (3 / 4), np.pi / 2, np.pi / 2])
        assert np.allclose(metrics.compute_turning_angles(points2), expected_angles2), "Test 2 failed"

        points3 = np.array([[0, 0], [1, 0], [np.nan, 0], [1, 0], [0, 1]], dtype=float)
        expected_angles3 = np.array([0, 0, np.pi * (3 / 4)])
        assert np.allclose(metrics.compute_turning_angles(points3), expected_angles3), "Test 3 failed"

        points4 = np.array([[0, 0], [1, 0], [1, 0], [np.inf, np.inf], [0, 1]], dtype=float)
        expected_angles4 = np.array([0, 0, 0])
        assert np.allclose(metrics.compute_turning_angles(points4), expected_angles4), "Test 4 failed"


if __name__ == "__main__":
    unittest.main()
