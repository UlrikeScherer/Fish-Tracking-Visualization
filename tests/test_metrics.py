import fishproviz
import fishproviz.utils as utils
import fishproviz.config as config
import fishproviz.metrics as metrics
import unittest
import numpy as np
import os

# Path: tests/test_utils.py
fpv_path = os.path.dirname(fishproviz.__file__)  # path to fishproviz
# get parent dir of fpv_path
fpv_path = os.path.dirname(fpv_path)


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
        results = metrics.activity_per_interval(
            fish_ids=self.fish_ids, time_interval=100, write_to_csv=True
        )
        results = metrics.activity_per_interval(
            fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True
        )
        print("Testing distance to wall")
        results = metrics.distance_to_wall_per_interval(
            fish_ids=self.fish_ids, time_interval=100, write_to_csv=True
        )
        results = metrics.distance_to_wall_per_interval(
            fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True
        )
        print("Testing turning angle")
        results = metrics.turning_angle_per_interval(
            fish_ids=self.fish_ids, time_interval=100, write_to_csv=True
        )
        results = metrics.turning_angle_per_interval(
            fish_ids=self.fish_ids, time_interval=(60**2), write_to_csv=True
        )


if __name__ == "__main__":
    unittest.main()
