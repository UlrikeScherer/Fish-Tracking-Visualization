import unittest
import time 
import os
from matplotlib.testing.compare import compare_images
from src.visualisation import set_figure, subplot_trajectory
from src.utile import csv_of_the_day
from src.metrics import metric_per_interval

class TestTrajectory(unittest.TestCase):
    camera_id="23520289"
    day= "20210910_1550"
    is_back=False 

    def test_trajectory_plot(self):
        img1 = "plots/block1/front/{}/{}/{}.pdf".format(self.camera_id, self.day, "01")
        img2 = "./output/test.pdf"

        tstart = time.time()
        day_df = csv_of_the_day(self.camera_id, self.day, is_back=self.is_back, drop_out_of_scope=True)
        tmid = time.time()
        print("Reading CSVs of %s at day %s CTIME:\t %f"%(self.camera_id, self.day, tmid-tstart))

        fig_o = set_figure(self.is_back, marker_char="")
        for batch in day_df:
            subplot_trajectory(fig_o, batch, self.day, "./output", "test", is_back=self.is_back)
        tmid2 = time.time()
        print("Trajectory Plotting Time:%s %f"%("\t"*4,tmid2-tmid))
        self.assertTrue( (tmid2-tmid) < 2 )                                 # smaller one second
        compare_images(img1, img2, 0.001)
        print("Compare Time:%s %f"%("\t"*6,time.time()-tmid2))

        img2_size = os.path.getsize(img2)
        print("IMG Size is :", img2_size, "bytes")
        self.assertTrue( (img2_size) < 70000)  


    def test_metric(self):
        day_df = csv_of_the_day(self.camera_id, self.day, is_back=self.is_back, drop_out_of_scope=True)
        metric_per_interval()

if __name__ == '__main__':
    unittest.main()
