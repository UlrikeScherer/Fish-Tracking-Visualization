import unittest
import time 
import os
from matplotlib.testing.compare import compare_images
from src.visualisation import Trajectory
from src.utile import csv_of_the_day
from src.metrics import metric_per_interval

class TestTrajectory(unittest.TestCase):
    camera_id="23520289"
    day= "20210912_060000"
    is_back=False 

    def test_trajectory_plot(self):
        img1 = "plots/060000/block1/front/{}/{}/{}.pdf".format(self.camera_id, self.day, "01")
        img2 = "./output/test.pdf"

        tstart = time.time()
        keys, day_df = csv_of_the_day(self.camera_id, self.day, is_back=self.is_back, drop_out_of_scope=True)
        tmid = time.time()
        print("Reading %d CSVs of %s at day %s CTIME:\t %f"%(len(day_df), self.camera_id, self.day, tmid-tstart))
        T = Trajectory()
        for batch in day_df[1:2]:
            fig = T.subplot_function(batch, self.day, "output", "test",None, is_back=self.is_back)
        tmid2 = time.time()
        print("Trajectory Plotting Time:%s %f"%("\t"*4,tmid2-tmid))
        self.assertTrue( (tmid2-tmid) < 3 )                                 # smaller one second
        compare_images(img1, img2, 0.001)
        print("Compare Time:%s %f"%("\t"*6,time.time()-tmid2))

        img2_size = os.path.getsize(img2)
        print("IMG Size is :", img2_size, "bytes")
        self.assertTrue( (img2_size) < 70000)  


    def test_metric(self):
        metric_per_interval(fish_ids=[1,3], day_interval=[0,4])

if __name__ == '__main__':
    unittest.main()
