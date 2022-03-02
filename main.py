import time
import sys
import matplotlib.pyplot as plt
from src.utile import get_camera_names, get_days_in_order, DIR_CSV_LOCAL, fish2camera, MEAN_GLOBAL, print_tex_table
from src.visualisation import plots_for_tex
from src.metrics import activity_per_interval, turning_angle_per_interval, tortuosity_per_interval, entropy_per_interval
from src.activity_plotting import sliding_window, sliding_window_figures_for_tex

TRAJECTORY="trajectory"
ACTIVITY="activity"
TURNING_ANGLE="turning_angle"
TORTUOSITY="tortuosity"
ENTROPY="entropy"
ALL_METRICS="all"
programs = [TRAJECTORY, ACTIVITY, TURNING_ANGLE, TORTUOSITY, ENTROPY]
metric_names = [ACTIVITY, TURNING_ANGLE, TORTUOSITY, ENTROPY]
time_intervals = [100,100,100,200]

def map_r_to_idx(results, fish_idx): 
    return [results[i] for i in fish_idx]

def plotting_odd_even(results, time_interval, name, ylabel, **kwargs):
    N_FISHES = len(fish2camera)
    fish_ids_even = [i for i in range(0 ,N_FISHES,2)]
    fish_ids_odd = [i for i in range(1, N_FISHES, 2)]
    size_even, size_odd = int(len(fish_ids_even)/2), int(len(fish_ids_odd)/2)
    fish_ids_even_1 = fish_ids_even[:size_even]
    fish_ids_even_2 = fish_ids_even[size_even:]
    fish_ids_odd_1 = fish_ids_odd[:size_odd]
    fish_ids_odd_2 = fish_ids_odd[size_odd:]

    kwargs["write_fig"]=True
    fish_batches = [fish_ids_even_1, fish_ids_even_2, fish_ids_odd_1, fish_ids_odd_2]
    positions = ["front_1", "front_2", "back_1", "back_2"]
    names = ["%s_%s"%(name, p) for p in positions]

    for i, batch in enumerate(fish_batches):
        print("Plotting %s %s"%(names[i], batch))
        print_tex_table(batch, positions[i])
        select_data = map_r_to_idx(results,batch)
        f_ = sliding_window(select_data, time_interval, sw=10, fish_ids=batch, ylabel=ylabel, name=names[i], **kwargs)
        plt.close(f_)
        sliding_window_figures_for_tex(select_data, time_interval, sw=10, fish_ids=batch, ylabel=ylabel, name=names[i], **kwargs)
        
def main(program=None, test=0, time_interval=100):
    """param:   test, 0,1 when test==1 run test mode
                program: trajectory, activity, turning_angle
                time_interval: kwarg for the programs activity, turning_angle
    """
    cameras = get_camera_names()
    days = get_days_in_order()
    time_interval=int(time_interval)
    file_name = "%s_%s"%(time_interval, program)

    if program == TRAJECTORY:
        print(test)
        if int(test) == 1:
            print("Test RUN ", TRAJECTORY)
            cameras=cameras[1:2]
            days=days[1:2]
        plots_for_tex(days)

    elif program == ACTIVITY:
        results = activity_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(results, time_interval=time_interval, name=file_name, ylabel="activity", set_title=True, set_legend=True, baseline=MEAN_GLOBAL)

    elif program == TORTUOSITY:
        results = tortuosity_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(results, time_interval=time_interval, name=file_name, ylabel="tortuosity", logscale=True, baseline=1)

    elif program == TURNING_ANGLE:
        results = turning_angle_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(results, time_interval=time_interval, name=file_name, ylabel="turning angle", baseline=0)
    elif program == ENTROPY:
        results = entropy_per_interval(time_interval=time_interval, write_to_csv=True)
        plotting_odd_even(results, time_interval=time_interval, name=file_name, ylabel="entropy", baseline=0)

    elif program == ALL:
        for p in metric_names: 
            main(p)
    else:
        print("Please provide a program name that you want to run. One of: ", programs)

        
    
if __name__ == '__main__':
    tstart = time.time()
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
    tend = time.time()
    print("Running time:", tend-tstart, "sec.")
    