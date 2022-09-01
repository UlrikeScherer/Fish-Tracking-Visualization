import shutil
import time
import sys, os, inspect
import matplotlib.pyplot as plt
import numpy as np
from src.utils import print_tex_table, get_camera_pos_keys
from src.config import (
    DIR_CSV_LOCAL,
    HOURS_PER_DAY,
    MEAN_GLOBAL,
    N_SECONDS_PER_HOUR,
    dir_feeding_back,
    PLOTS_TRAJECTORY,
    DATA_results,
)
from src.trajectory import Trajectory, FeedingTrajectory
from src.metrics import (
    activity_per_interval,
    turning_angle_per_interval,
    tortuosity_per_interval,
    entropy_per_interval,
    metric_per_hour_csv,
    distance_to_wall_per_interval,
    absolute_angle_per_interval,
)
from src.visualizations.activity_plotting import (
    sliding_window,
    sliding_window_figures_for_tex,
)
from src.utils import is_valid_dir

TRAJECTORY = "trajectory"
ACTIVITY = "activity"
TURNING_ANGLE = "turning_angle"
ABS_ANGLE = "abs_angle"
TORTUOSITY = "tortuosity"
ENTROPY = "entropy"
WALL_DISTANCE = "wall_distance"
ALL_METRICS = "all"
CLEAR = "clear"
metric_names = [ACTIVITY, TURNING_ANGLE, ABS_ANGLE, TORTUOSITY, ENTROPY, WALL_DISTANCE]
programs = [TRAJECTORY, *metric_names, ALL_METRICS, CLEAR]


def plotting_odd_even(
    results, time_interval=None, name=None, ylabel=None, sw=10, visualize=None, **kwargs
):
    if visualize is None:
        return None
    fish_keys = list(results.keys())
    fk_len = len(fish_keys)
    batch_size = 6
    fish_batches = []
    for start in range(0, fk_len, batch_size):
        end = min(start + batch_size, fk_len)
        fish_batches.append(list(range(start, end)))
    kwargs["write_fig"] = True

    fish_labels = get_camera_pos_keys()
    positions = ["front_1", "front_2", "back_1", "back_2"]
    names = ["%s_%s" % (name, p) for p in positions]

    for i, batch in enumerate(fish_batches):
        print("Plotting %s %s" % (names[i], batch))
        print_tex_table(batch, positions[i])
        batch_keys = [fish_keys[i] for i in batch]
        f_ = sliding_window(
            results,
            time_interval,
            sw=sw,
            fish_keys=batch_keys,
            fish_labels=fish_labels[batch],
            ylabel=ylabel,
            name=names[i],
            **kwargs
        )
        plt.close(f_)
        sliding_window_figures_for_tex(
            results,
            time_interval,
            sw=sw,
            fish_keys=batch_keys,
            fish_labels=fish_labels[batch],
            ylabel=ylabel,
            name=names[i],
            **kwargs
        )


def main_trajectory(is_feeding, fish_ids):
    if is_feeding:
        FT = FeedingTrajectory()
        FT.plots_for_tex(fish_ids)
        FT.feeding_data_to_csv()
        FT.feeding_data_to_tex()
    else:
        T = Trajectory()
        T.plots_for_tex(fish_ids)


def main_metrics(
    program,
    time_interval=100,
    sw=10,
    visualize=False,
    include_median=None,
    **kwargs_metrics
):

    # TIME INTERVAL for the metrics
    if time_interval in [
        "hour",
        "day",
    ]:  # change of parameters when computing metrics by hour or day
        sw = 1
        if time_interval == "hour":
            time_interval = N_SECONDS_PER_HOUR
        if time_interval == "day":
            time_interval = N_SECONDS_PER_HOUR * HOURS_PER_DAY
    else:
        time_interval = int(time_interval)

    if include_median and program != ACTIVITY:
        raise ValueError("include_median is only valid for activity")
    # put it back in the kwargs
    kwargs_metrics.update(time_interval=time_interval)

    if program == ACTIVITY:
        results = activity_per_interval(include_median=include_median, **kwargs_metrics)
        plotting_odd_even(
            **results,
            ylabel="activity",
            set_title=True,
            set_legend=True,
            baseline=MEAN_GLOBAL,
            sw=sw
        )

    elif program == TORTUOSITY:
        results = tortuosity_per_interval(**kwargs_metrics)
        plotting_odd_even(
            **results,
            ylabel="tortuosity",
            logscale=True,
            baseline=1,
            visualize=visualize
        )

    elif program == TURNING_ANGLE:
        results = turning_angle_per_interval(**kwargs_metrics)
        plotting_odd_even(
            **results, ylabel="turning angle", baseline=0, visualize=visualize
        )

    elif program == ABS_ANGLE:
        results = absolute_angle_per_interval(**kwargs_metrics)
        plotting_odd_even(**results, ylabel="absolute angle", visualize=visualize)

    elif program == ENTROPY:
        results = entropy_per_interval(**kwargs_metrics)
        plotting_odd_even(**results, ylabel="entropy", visualize=visualize)

    elif program == WALL_DISTANCE:
        results = distance_to_wall_per_interval(**kwargs_metrics)
        plotting_odd_even(
            **results, ylabel="distance to the wall", baseline=0, visualize=visualize
        )
    else:
        print("TERMINATED: Invalid program")

    if (
        time_interval == N_SECONDS_PER_HOUR
        or time_interval == N_SECONDS_PER_HOUR * HOURS_PER_DAY
    ):
        metric_per_hour_csv(**results)


def get_fish_ids_to_run(program, fish_id, is_feeding):
    fish_keys = get_camera_pos_keys(is_feeding=is_feeding)
    n_fishes = len(fish_keys)

    fish_ids = np.arange(n_fishes)
    if fish_id is not None:
        if fish_id.isnumeric():
            fish_ids = np.array([int(fish_id)])
        elif fish_id in fish_keys:
            fish_ids = np.array([fish_keys.index(fish_id)])
        else:
            raise ValueError(
                "fish_id=%s does not appear in the data, please provid the fish_id as camera_position or index integer in [0 to %s]. \n\n The following ids are valid: %s"
                % (fish_id, n_fishes - 1, fish_keys)
            )
        print("program", program, "will run for fish:", fish_keys[fish_ids[0]])
    return fish_ids


def main(
    program=None,
    time_interval=100,
    sw=10,
    fish_id=None,
    visualize=None,
    feeding=0,
    include_median=None,
):
    """param:   test, 0,1 when test==1 run test mode
    program: trajectory, activity, turning_angle
    time_interval: kwarg for the programs activity, turning_angle
    """
    is_feeding = bool(int(feeding))
    if is_feeding:
        if not is_valid_dir(dir_feeding_back):
            return None
    else:
        if not is_valid_dir(DIR_CSV_LOCAL):
            return None

    fish_ids = get_fish_ids_to_run(program, fish_id, is_feeding)
    kwargs_metrics = dict(
        fish_ids=fish_ids,
        time_interval=time_interval,
        write_to_csv=True,
        is_feeding=is_feeding,
        visualize=visualize,
        sw=sw,
        include_median=include_median,
    )
    # PROGRAM METRICS or TRAJECTORY or CLEAR
    if program == TRAJECTORY:
        main_trajectory(is_feeding, fish_ids)
    elif program in metric_names:
        main_metrics(program, **kwargs_metrics)
    elif program == ALL_METRICS:
        for p in metric_names:
            main_metrics(p, **kwargs_metrics)
    elif program == CLEAR:  # clear all data remove directories DANGEROUS!
        for path in [PLOTS_TRAJECTORY, DATA_results]:  # VIS_DIR
            if os.path.isdir(path):
                shutil.rmtree(path)
                print("Removed directory: %s" % path)
    else:
        print(
            "TERMINATED: Please provide the program name which you want to run. One of: ",
            programs,
        )
    return None


if __name__ == "__main__":
    tstart = time.time()
    main_kwargs = dict(inspect.signature(main).parameters)
    try:
        kwargs = dict(arg.split("=") for arg in sys.argv[1:])
    except Exception as e:
        raise ValueError(
            "Please supply the keyword arguments as follows: kwarg1=value ... ", e
        )
    for k in kwargs.keys():
        if k not in main_kwargs:
            raise ValueError(
                "Please use the following keyword arguments %s" % main_kwargs.keys()
            )
    main(**kwargs)
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
