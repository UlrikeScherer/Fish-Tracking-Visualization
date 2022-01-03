import time

from src.utile import get_camera_names, get_days_in_order
from src.visualisation import plots_for_tex


def main():
    cameras = get_camera_names()
    days = get_days_in_order()
    plots_for_tex(cameras, days, dpi=100)
    
if __name__ == '__main__':
    tstart = time.time()
    main()
    tend = time.time()
    print("Running time:", tend-tstart, "sec.")
    