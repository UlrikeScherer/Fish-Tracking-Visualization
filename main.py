import shutil
import time
import os, inspect
import json
import numpy as np
import argparse
from fishproviz.metrics.exploration_trials import exploration_trials
from fishproviz.utils import get_camera_pos_keys
from fishproviz.config import (
    DIR_CSV_LOCAL,
    HOURS_PER_DAY,
    N_SECONDS_PER_HOUR,
    PLOTS_DIR,
    RESULTS_PATH,
    create_directories,
)
from fishproviz.trajectory import Trajectory, FeedingTrajectory, NovelObjectTrajectory, SociabilityTrajectory
from fishproviz.metrics import (
    activity_per_interval,
    turning_angle_per_interval,
    tortuosity_per_interval,
    entropy_per_interval,
    distance_to_wall_per_interval,
    distance_to_object_per_interval,
    absolute_angle_per_interval,
)

TRAJECTORY = "trajectory"
FEEDING = "feeding"
NOVEL_OBJECT = "novel_object"
SOCIABILITY = "sociability"
TRIAL_TIMES = "trial_times"
ACTIVITY = "activity"
TURNING_ANGLE = "turning_angle"
ABS_ANGLE = "abs_angle"
TORTUOSITY = "tortuosity"
ENTROPY = "entropy"
WALL_DISTANCE = "wall_distance"
NOVEL_OBJECT_DISTANCE = "novel_object_distance"
ALL_METRICS = "all"
CLEAR = "clear"
metric_names = [ACTIVITY, TURNING_ANGLE, ABS_ANGLE, TORTUOSITY, ENTROPY, WALL_DISTANCE, NOVEL_OBJECT_DISTANCE]
programs = [TRAJECTORY, FEEDING, TRIAL_TIMES, *metric_names, ALL_METRICS, CLEAR, NOVEL_OBJECT, SOCIABILITY]


def main_metrics(program, time_interval=100, include_median=None, **kwargs_metrics):
    '''
    updates overloaded time-interval in kwargs_metrics and writes out metric results to csv-file
    params:
        program: str
        time_interval: int
        include_median: boolean
        kwargs_metrics
    returns:
        int status code
    '''
    if time_interval in ["hour", "day"]:
        time_interval = {
            "hour": N_SECONDS_PER_HOUR,
            "day": int(N_SECONDS_PER_HOUR * HOURS_PER_DAY),
        }[time_interval]
    else:
        time_interval = int(time_interval)

    if time_interval < 30:
        raise ValueError("time_interval must be at least 30 seconds otherwise the csv files will be too large")

    if include_median and program != ACTIVITY:
        raise ValueError("include_median is only valid for activity")

    kwargs_metrics.update(time_interval=time_interval)
    
    metric_functions = {
        ACTIVITY: activity_per_interval,
        TORTUOSITY: tortuosity_per_interval,
        TURNING_ANGLE: turning_angle_per_interval,
        ABS_ANGLE: absolute_angle_per_interval,
        ENTROPY: entropy_per_interval,
        WALL_DISTANCE: distance_to_wall_per_interval,
        NOVEL_OBJECT_DISTANCE: distance_to_object_per_interval
    }

    if program not in metric_functions:
        print("TERMINATED: Invalid program")
        return -1

    results = metric_functions[program](include_median=include_median, **kwargs_metrics)
    return None

def get_fish_ids_to_run(program, fish_id):
    '''
    calculates fish-ids from camera positions to distinguish program runs for fishes
    params: 
        program: str
        fish_id: int
    returns: np-array of ids
    '''
    fish_keys = get_camera_pos_keys()

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
    fish_id=None,
    include_median=None,
    parallel=False,
    print_logs=False,
):
    """
    params:  
        program: str (trajectory, activity, turning_angle)
        time_interval: int 
        fish_id: int
        include_median: bool
        kwargs for the programs activity, turning_angle
    """
    fish_ids = get_fish_ids_to_run(program, fish_id)
    kwargs_metrics = dict(
        fish_ids=fish_ids,
        time_interval=time_interval,
        write_to_csv=True,
        include_median=include_median,
        print_logs=print_logs,
    )
    # PROGRAM METRICS or TRAJECTORY or CLEAR
    if program == TRAJECTORY:
        T = Trajectory(
            parallel = json.loads(
                str(parallel).lower()
            )
        )
        T.plots_for_tex(fish_ids)
    elif program == FEEDING:
        FT = FeedingTrajectory()
        FT.plots_for_tex(fish_ids)
        FT.feeding_data_to_csv()
        FT.feeding_data_to_tex()
    elif program == NOVEL_OBJECT:
        NT = NovelObjectTrajectory()
        NT.plots_for_tex(fish_ids)
        NT.object_data_to_csv()
        NT.object_data_to_tex()
    elif program == SOCIABILITY:
        ST = SociabilityTrajectory()
        ST.plots_for_tex(fish_ids)
        ST.object_data_to_csv()
        ST.object_data_to_tex()
    elif program == TRIAL_TIMES:
        exploration_trials()
    elif program in metric_names:
        main_metrics(program, **kwargs_metrics)
    elif program == ALL_METRICS:
        for p in metric_names:
            main_metrics(p, **kwargs_metrics)
    elif program == CLEAR:  # clear all data remove directories DANGEROUS!
        for path in [PLOTS_DIR, RESULTS_PATH]:  # VIS_DIR
            if os.path.isdir(path):
                shutil.rmtree(path)
                print("Removed directory: %s" % path)
    return None


def set_args():
    parser = argparse.ArgumentParser(
        prog="python3 main.py",
        description="This program computes metrics and visualizations for fish trajectories, the results are saved in the directory %s"
        % DIR_CSV_LOCAL,
        epilog="Example of use: python3 main.py trajectory -fid 0",
    )
    parser.add_argument(
        "program",
        help="Select the program you want to execute",
        type=str,
        choices=list(programs),
    )
    parser.add_argument(
        "-ti",
        "--time_interval",
        help="Choose a time interval in second to compute averages of metrics, also possible [day, hour]",
        type=str,
        default=100,
    )
    parser.add_argument(
        "-fid",
        "--fish_id",
        help="Fish id to run can be by 'camera_position' or index, default is all fish_ids",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--include_median",
        help="Include median or not only for activity",
        action="store_true",
    )
    parser.add_argument(
        "-mp",
        "--parallel",
        help="compute trajectories in parallel",
        type=str,
        default=False,
    )
    parser.add_argument(
        "-logs",
        "--print_logs",
        help="Print logs from duplicate file detection other file missmatches",
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    tstart = time.time()
    create_directories()
    main_kwargs = dict(inspect.signature(main).parameters)

    main(**args.__dict__)
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
