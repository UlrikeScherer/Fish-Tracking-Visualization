import time
import sys
from src.utile import get_camera_names, get_days_in_order, DIR_CSV_LOCAL, fish2camera
from src.visualisation import plots_for_tex
from src.metrics import activity_per_interval, turning_angle_per_interval, tortuosity_per_interval
from src.activity_plotting import sliding_window

TRAJECTORY="trajectory"
ACTIVITY="activity"
TURNING_ANGLE="turning_angle"
TORTUOSITY="tortuosity"
programs = [TRAJECTORY, ACTIVITY, TURNING_ANGLE, TORTUOSITY]

def map_r_to_idx(results, fish_idx): 
    return [results[i] for i in fish_idx]

def plotting_odd_even(results, name, ylabel, **kwargs):
    N_FISHES = len(fish2camera)
    fish_ids_even = [i for i in range(0,N_FISHES,2)]
    fish_ids_odd = [i for i in range(1, N_FISHES, 2)]
    size_even, size_odd = int(len(fish_ids_even)/2), int(len(fish_ids_odd)/2)
    fish_ids_even_1 = fish_ids_even[:size_even]
    fish_ids_even_2 = fish_ids_even[size_even:]
    fish_ids_odd_1 = fish_ids_odd[:size_odd]
    fish_ids_odd_2 = fish_ids_odd[size_odd:]

    kwargs["write_fig"]=True
    print("Plotting even 1 %s"%fish_ids_even_1)
    f_even = sliding_window(map_r_to_idx(results,fish_ids_even_1), time_interval=100, sw=10, fish_ids=fish_ids_even_1, ylabel=ylabel, name="%s_front_1"%name, **kwargs)
    print("Plotting even 2 %s"%fish_ids_even_2)
    f_even = sliding_window(map_r_to_idx(results,fish_ids_even_2), time_interval=100, sw=10, fish_ids=fish_ids_even_2, ylabel=ylabel, name="%s_front_2"%name, **kwargs)
    print("Plotting odd 1 %s"%fish_ids_odd_1)
    f_odd = sliding_window(map_r_to_idx(results,fish_ids_odd_1), time_interval=100, sw=10, fish_ids=fish_ids_odd_1, ylabel=ylabel, name="%s_back_1"%name, **kwargs)
    print("Plotting odd 1 %s"%fish_ids_odd_2)
    f_odd = sliding_window(map_r_to_idx(results,fish_ids_odd_2), time_interval=100, sw=10, fish_ids=fish_ids_odd_2, ylabel=ylabel, name="%s_back_2"%name, **kwargs)

def main(program=None, test=0, time_interval=100):
    """param:   test, 0,1 when test==1 run test mode
                program: trajectory, activity, turning_angle
                time_interval: kwarg for the programs activity, turning_angle
    """
    cameras = get_camera_names()
    days = get_days_in_order()
    time_interval=int(time_interval)

    if program == TRAJECTORY:
        print(test)
        if int(test) == 1:
            print("Test RUN ", TRAJECTORY)
            cameras=cameras[1:2]
            days=days[1:2]
        plots_for_tex(cameras, days)

    elif program == ACTIVITY:
        results = activity_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(results, name="%s_activity"%time_interval, ylabel="average step length per frame")

    elif program == TORTUOSITY:
        results = tortuosity_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(results, name="%s_tortuosity"%time_interval, ylabel="average tortuosity", logscale=True, baseline=1)

    elif program == TURNING_ANGLE:
        results = turning_angle_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(results, name="%s_turning_angle"%time_interval, ylabel="average turning angle per frame", baseline=0)
        
    else:
        print("Please provide a program name that you want to run. One of: ", programs)

        
    
if __name__ == '__main__':
    tstart = time.time()
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
    tend = time.time()
    print("Running time:", tend-tstart, "sec.")
    