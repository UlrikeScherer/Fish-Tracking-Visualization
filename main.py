import time
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from src.utile import get_camera_names, get_days_in_order, DIR_CSV_LOCAL, MEAN_GLOBAL, print_tex_table, get_fish2camera_map, N_FISHES, get_fish_ids, N_SECONDS_PER_HOUR
from src.visualisation import Trajectory
from src.metrics import activity_per_interval, turning_angle_per_interval, tortuosity_per_interval, entropy_per_interval, metric_per_hour_csv, distance_to_wall_per_interval
from src.activity_plotting import sliding_window, sliding_window_figures_for_tex
from src.feeding import FeedingTrajectory
TRAJECTORY="trajectory"
FEEDING="feeding"
ACTIVITY="activity"
TURNING_ANGLE="turning_angle"
TORTUOSITY="tortuosity"
ENTROPY="entropy"
WALL_DISTANCE="wall_distance"
ALL_METRICS="all"
programs = [TRAJECTORY,FEEDING, ACTIVITY, TURNING_ANGLE, TORTUOSITY, ENTROPY, WALL_DISTANCE]
metric_names = [ACTIVITY, TURNING_ANGLE, TORTUOSITY, ENTROPY, WALL_DISTANCE]
#time_intervals = [100,100,100,200]

def map_r_to_idx(results, fish_idx): 
    return [results[i] for i in fish_idx]

def plotting_odd_even(results, time_interval=None, metric_name=None, name=None, ylabel=None, sw=10, **kwargs):
    fish_keys = list(results.keys())
    fish_ids_even = [i for i in range(0, N_FISHES, 2)]
    fish_ids_odd = [i for i in range(1, N_FISHES, 2)]
    size_even, size_odd = int(len(fish_ids_even)/2), int(len(fish_ids_odd)/2)
    fish_ids_even_1 = fish_ids_even[:size_even]
    fish_ids_even_2 = fish_ids_even[size_even:]
    fish_ids_odd_1 = fish_ids_odd[:size_odd]
    fish_ids_odd_2 = fish_ids_odd[size_odd:]

    kwargs["write_fig"]=True
    fish_batches = [fish_ids_even_1, fish_ids_even_2, fish_ids_odd_1, fish_ids_odd_2]
    fish_labels = get_fish_ids()
    positions = ["front_1", "front_2", "back_1", "back_2"]
    names = ["%s_%s"%(name, p) for p in positions]

    for i, batch in enumerate(fish_batches):
        print("Plotting %s %s"%(names[i], batch))
        print_tex_table(batch, positions[i])
        batch_keys = [fish_keys[i] for i in batch]
        #select_data = map_r_to_idx(results,batch_keys)
        f_ = sliding_window(results, time_interval, sw=sw, fish_keys=batch_keys, fish_labels=fish_labels[batch], ylabel=ylabel, name=names[i], **kwargs)
        plt.close(f_)
        sliding_window_figures_for_tex(results, time_interval, sw=sw, fish_keys=batch_keys, fish_labels=fish_labels[batch], ylabel=ylabel, name=names[i], **kwargs)
        
def main(program=None, test=0, time_interval=100, sw=10, fish_id=None):
    """param:   test, 0,1 when test==1 run test mode
                program: trajectory, activity, turning_angle
                time_interval: kwarg for the programs activity, turning_angle
    """
    if not os.path.isdir(DIR_CSV_LOCAL): 
        print("TERMINATED: Please connect to external hard drive with path %s or edit path in scripts/env.sh"%DIR_CSV_LOCAL)
        return None
    is_feeding = program==FEEDING
    cameras = get_camera_names(is_feeding=is_feeding)
    days = get_days_in_order(is_feeding=is_feeding)
    fish2camera = get_fish2camera_map(is_feeding=is_feeding)
    N_FISHES = len(fish2camera) 
    if time_interval == "hour": # change of parameters when computing metrics by hour 
        time_interval=N_SECONDS_PER_HOUR
        sw=1
    else:
        time_interval=int(time_interval)
    file_name = "%s_%s"%(time_interval, program)

    fish_ids = np.arange(N_FISHES)
    if fish_id!=None:
        fish_ids=np.array([int(fish_id)])

    if int(test) == 1:
        print("Test RUN ", program)
        fish_ids=fish_ids[8:9]
        print("For days: %s, fish indices: %s"%(",".join(days),fish_ids))

    if program == TRAJECTORY:
        T = Trajectory()
        T.plots_for_tex(fish_ids)

    elif program == FEEDING:
        FT = FeedingTrajectory()
        FT.plots_for_tex(fish_ids)
        FT.feeding_data_to_csv()
        FT.feeding_data_to_tex()

    elif program == ACTIVITY:
        results = activity_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(**results, name=file_name, ylabel="activity", set_title=True, set_legend=True, baseline=MEAN_GLOBAL, sw=sw)

    elif program == TORTUOSITY:
        results = tortuosity_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(**results, name=file_name, ylabel="tortuosity", logscale=True, baseline=1, sw=sw)

    elif program == TURNING_ANGLE:
        results = turning_angle_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(**results, name=file_name, ylabel="turning angle", baseline=0, sw=sw)
    elif program == ENTROPY:
        results = entropy_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(**results, name=file_name, ylabel="entropy", sw=sw)
    elif program == WALL_DISTANCE:
        results = distance_to_wall_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(**results, name=file_name, ylabel="distance to the wall", baseline=0, sw=sw)

    elif program == ALL_METRICS:
        for p in metric_names: 
            main(p, time_interval=time_interval, fish_id=fish_id)
    else:
        print("Please provide the program name which you want to run. One of: ", programs)


    if time_interval == N_SECONDS_PER_HOUR:
        metric_per_hour_csv(**results)
        
    
if __name__ == '__main__':
    tstart = time.time()
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
    tend = time.time()
    print("Running time:", tend-tstart, "sec.")
    