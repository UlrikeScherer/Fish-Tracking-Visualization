from src.utils.error_filter import error_dirt_points
from src.trajectory import Trajectory
from src.utils import csv_of_the_day
from src.metrics import metric_per_interval
from array_for_test import a1, sol1
import unittest
import time
import os
import numpy as np
from matplotlib.testing.compare import compare_images


class TestTrajectory(unittest.TestCase):
    camera_id = "23520289"
    day = "20210912_060000"
    is_back = False

    def test_trajectory_plot(self):
        img1 = "plots/trajectory/060000/block1/front/{}/{}/{}.pdf".format(
            self.camera_id, self.day, "000001"
        )
        img2 = "./output/test.pdf"

        tstart = time.time()
        keys, day_df = csv_of_the_day(
            self.camera_id, self.day, is_back=self.is_back, drop_out_of_scope=True
        )
        tmid = time.time()
        print(
            "Reading %d CSVs of %s at day %s CTIME:\t %f"
            % (len(day_df), self.camera_id, self.day, tmid - tstart)
        )
        T = Trajectory()
        for batch in day_df[1:2]:
            _ = T.subplot_function(
                batch, self.day, "output", "test", None, is_back=self.is_back
            )
        tmid2 = time.time()
        print("Trajectory Plotting Time:%s %f" % ("\t" * 4, tmid2 - tmid))
        self.assertTrue((tmid2 - tmid) < 3)  # smaller one second
        compare_images(img1, img2, 0.001)
        print("Compare Time:%s %f" % ("\t" * 6, time.time() - tmid2))

        img2_size = os.path.getsize(img2)
        print("IMG Size is :", img2_size, "bytes")
        self.assertTrue((img2_size) < 90000)

    def test_metric(self):
        metric_per_interval(fish_ids=[1, 3], day_interval=[0, 4])

    def test_filters(self):
        print("Testing filters")
        flt = error_dirt_points(a1, threshold=4)
        self.assertTrue(np.alltrue(a1[flt] == sol1))


if __name__ == "__main__":
    unittest.main()
