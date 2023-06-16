import fishproviz
import fishproviz.utils as utils
import fishproviz.config as config
import fishproviz.metrics as metrics
import unittest
import numpy as np
import os
import time

# Path: tests/test_utils.py
fpv_path = os.path.dirname(fishproviz.__file__)  # path to fishproviz
# get parent dir of fpv_path
fpv_path = os.path.dirname(fpv_path)
print(fpv_path)


class TestUtils(unittest.TestCase):
    # set the config variables to test_data folder
    global config
    config.set_config_paths(f"{fpv_path}/test_data")
    config.dir_front = f"{fpv_path}/test_data/front"
    config.dir_back = f"{fpv_path}/test_data/back"
    config.area_back = f"{fpv_path}/test_data/area_config/areas_back"
    config.area_front = f"{fpv_path}/test_data/area_config/areas_front"
    print(config.DIR_CSV_LOCAL)
    config.create_directories()

    fish_keys = utils.get_camera_pos_keys()
    n_fishes = len(fish_keys)
    fish_ids = np.arange(n_fishes)

    def test_activity(self):
        print("Fish Keys for test:", self.fish_keys)
        print("Testing activity")
        t = time.time()
        results = metrics.activity_per_interval(
            fish_ids=self.fish_ids, time_interval=100, write_to_csv=True
        )
        results = metrics.activity_per_interval(
            fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True
        )
        print("Time taken: %.02f seconds" % (time.time() - t))
        print("Testing distance to wall")
        t = time.time()
        results = metrics.distance_to_wall_per_interval(
            fish_ids=self.fish_ids, time_interval=100, write_to_csv=True
        )
        results = metrics.distance_to_wall_per_interval(
            fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True
        )
        print("Time taken: %.02f seconds" % (time.time() - t))
        print("Testing turning angle")
        t = time.time()
        results = metrics.turning_angle_per_interval(
            fish_ids=self.fish_ids, time_interval=100, write_to_csv=True
        )
        results = metrics.turning_angle_per_interval(
            fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True
        )
        print("Time taken: %.02f seconds" % (time.time() - t))

    def test_compute_turning_angles(self):
        # Test 1
        points1 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        expected_angles1 = np.array([np.pi * (3 / 4), np.pi / 2, np.pi / 2])
        computed_angles1 = metrics.compute_turning_angles(points1)
        assert np.allclose(computed_angles1, expected_angles1), "Test 1 failed"

        # Test 2
        points2 = np.array(
            [[0, 0], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float
        )
        expected_angles2 = np.array([0, np.pi * (3 / 4), np.pi / 2, np.pi / 2])
        computed_angles2 = metrics.compute_turning_angles(points2)
        assert np.allclose(computed_angles2, expected_angles2), "Test 2 failed"

        # Test 3
        points3 = np.array([[0, 0], [1, 0], [np.nan, 0], [1, 0], [0, 1]], dtype=float)
        expected_angles3 = np.array([0, 0, np.pi * (3 / 4)])
        computed_angles3 = metrics.compute_turning_angles(points3)
        assert np.allclose(computed_angles3, expected_angles3), "Test 3 failed"

        # Test 4 - including infinite points
        points4 = np.array(
            [[0, 0], [1, 0], [1, 0], [np.inf, np.inf], [0, 1]], dtype=float
        )
        expected_angles4 = np.array([0, 0, 0])
        computed_angles4 = metrics.compute_turning_angles(points4)
        assert np.allclose(computed_angles4, expected_angles4), "Test 4 failed"

        print("All tests passed")


if __name__ == "__main__":
    unittest.main()
