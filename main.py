import time
import sys

from src.utile import get_camera_names, get_days_in_order, DIR_CSV_LOCAL
from src.visualisation import plots_for_tex
from src.metrics import activity_per_interval, turning_angle_per_interval

TRAJECTORY="trajectory"
ACTIVITY="activity"
programs = [TRAJECTORY, ACTIVITY]

def main(program=None, test=0, time_interval=100):
    """param:   test, 0,1 when test==1 run test mode
                program: trajectory, activity, turning_angle
                time_interval: kwarg for the programs activity, turning_angle
    """
    cameras = get_camera_names()
    days = get_days_in_order()

    if program == TRAJECTORY:
        print(test)
        if int(test) == 1:
            print("Test RUN ", TRAJECTORY)
            cameras=cameras[1:2]
            days=days[1:2]
        plots_for_tex(cameras, days, dpi=100)

    elif program == ACTIVITY:
        results = activity_per_interval(time_interval=time_interval, write_to_csv=True)

    else:
        print("Please provide a program name that you want to run. One of: ", programs)

        
    
if __name__ == '__main__':
    tstart = time.time()
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
    tend = time.time()
    print("Running time:", tend-tstart, "sec.")
    