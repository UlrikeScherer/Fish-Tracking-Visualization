import shutil
import time
import os
import json
from typing import Literal, Optional

import numpy as np
import argparse
from fishproviz.metrics.exploration_trials import exploration_trials
from fishproviz.utils import get_camera_pos_keys
from fishproviz.utils.utile import DAY, HOUR, SECONDS_PER_HOUR
import fishproviz.config as config
from fishproviz.trajectory import (
    ExperimentalTrajectory,
    FeedingTrajectory,
    NovelObjectTrajectory,
    SociabilityTrajectory,
)
from fishproviz.metrics import (
    metric_per_interval,
    activity,
    turning_angle,
    turning_angle_streak_length,
    tortuosity,
    entropy,
    distance_to_wall,
    distance_to_object,
    absolute_angles,
    step_length,
)
from fishproviz.metrics.results_to_csv import require_raw_mode

TRAJECTORY = "trajectory"
FEEDING = "feeding"
NOVEL_OBJECT = "novel_object"
SOCIABILITY = "sociability"
TRIAL_TIMES = "trial_times"
ACTIVITY = "activity"
TURNING_ANGLE = "turning_angle"
TURNING_ANGLE_STREAK_LENGTH = "turning_angle_streak_length"
ABS_ANGLE = "abs_angle"
TORTUOSITY = "tortuosity"
ENTROPY = "entropy"
WALL_DISTANCE = "wall_distance"
NOVEL_OBJECT_DISTANCE = "novel_object_distance"
STEP_LENGTH = "step_length"
ALL_METRICS = "all"
CLEAR = "clear"
metric_names = [
    ACTIVITY,
    TURNING_ANGLE,
    TURNING_ANGLE_STREAK_LENGTH,
    ABS_ANGLE,
    TORTUOSITY,
    ENTROPY,
    WALL_DISTANCE,
    STEP_LENGTH,
]
programs = [
    TRAJECTORY,
    FEEDING,
    TRIAL_TIMES,
    *metric_names,
    NOVEL_OBJECT_DISTANCE,
    ALL_METRICS,
    CLEAR,
    NOVEL_OBJECT,
    SOCIABILITY,
]


MIN_TIME_INTERVAL = 30  # seconds; shorter bins make the csv files too large

metric_functions = {
    ACTIVITY: activity,
    TORTUOSITY: tortuosity,
    TURNING_ANGLE: turning_angle,
    TURNING_ANGLE_STREAK_LENGTH: turning_angle_streak_length,
    ABS_ANGLE: absolute_angles,
    ENTROPY: entropy,
    WALL_DISTANCE: distance_to_wall,
    NOVEL_OBJECT_DISTANCE: distance_to_object,
    STEP_LENGTH: step_length,
}


def parse_time_interval(value: str) -> int | Literal["day"]:
    """argparse type for -ti: seconds (an int >= 30), 'hour' (3600 s) or 'day' (one bin per loaded day)."""
    if value == DAY:
        return DAY
    if value == HOUR:
        return SECONDS_PER_HOUR
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid time interval '{value}': use an integer number of seconds >= {MIN_TIME_INTERVAL}, '{HOUR}' or '{DAY}'") from None
    if seconds < MIN_TIME_INTERVAL:
        raise argparse.ArgumentTypeError(f"time interval {seconds} s is below {MIN_TIME_INTERVAL} s: shorter bins make the csv files too large")
    return seconds


def metric_programs(program: str) -> list[str]:
    """Metric programs that program runs: every metric for 'all', itself for a single metric, none otherwise."""
    if program == ALL_METRICS:
        return metric_names
    return [program] if program in metric_functions else []


def validate_include_median(program: str, include_median: bool, unaveraged: bool) -> None:
    """Raise unless include_median is off, or the program runs activity with averaged output."""
    if not include_median:
        return
    if program not in (ACTIVITY, ALL_METRICS):
        raise ValueError(f"--include_median is only valid for '{ACTIVITY}' (or '{ALL_METRICS}', where it applies to {ACTIVITY} only), not for '{program}'")
    if unaveraged:
        raise ValueError("--include_median applies to averaged output only, but UNAVERAGED=1 requests raw output; set UNAVERAGED=0 or drop --include_median")


def validate_raw_mode(program: str, unaveraged: bool) -> None:
    """With raw output requested, raise naming every metric the program runs that has no raw mode."""
    if unaveraged:
        require_raw_mode([metric_functions[p].__name__ for p in metric_programs(program)])


def main_metrics(program: str, include_median: bool = False, **kwargs_metrics) -> None:
    """Computes one metric program and writes its results to csv."""
    if program not in metric_functions:
        raise ValueError(f"'{program}' is not a metric program; choose one of {list(metric_functions)}")
    metric_per_interval(metric=metric_functions[program], include_median=include_median, **kwargs_metrics)


def get_fish_ids_to_run(program, fish_id):
    """
    calculates fish-ids from camera positions to distinguish program runs for fishes
    params:
        program: str
        fish_id: int
    returns: np-array of ids
    """
    fish_keys = get_camera_pos_keys()

    n_fishes = len(fish_keys)

    fish_ids = np.arange(n_fishes)
    if fish_id is not None:
        if fish_id.isnumeric():
            fish_ids = np.array([int(fish_id)])
        elif fish_id in fish_keys:
            fish_ids = np.array([fish_keys.index(fish_id)])
        else:
            raise ValueError("fish_id=%s does not appear in the data, please provid the fish_id as camera_position or index integer in [0 to %s]. \n\n The following ids are valid: %s" % (fish_id, n_fishes - 1, fish_keys))
        print("program", program, "will run for fish:", fish_keys[fish_ids[0]])
    return fish_ids


def main(
    program: str,
    time_interval: Optional[int | Literal["day"]] = None,
    fish_id: Optional[str] = None,
    include_median: bool = False,
    parallel: str | bool = False,
    print_logs: bool = False,
) -> None:
    """
    params:
        program: one of programs
        time_interval: bin width in seconds or 'day' (one bin per day, day_<metric>.csv); None for one bin per day in <metric>.csv
        fish_id: camera_position or index, default all fish
        include_median: activity only (for 'all', applied to activity alone)
    """
    validate_include_median(program, include_median, config.UNAVERAGED)
    validate_raw_mode(program, config.UNAVERAGED)
    fish_ids = get_fish_ids_to_run(program, fish_id)
    kwargs_metrics = dict(
        fish_ids=fish_ids,
        write_to_csv=True,
        print_logs=print_logs,
        is_feeding=program == FEEDING,
        is_novel_object=program == NOVEL_OBJECT_DISTANCE,
        is_sociability=program == SOCIABILITY,
        all_points=time_interval is None,
        every_point=config.UNAVERAGED,
    )
    if time_interval is not None:
        kwargs_metrics.update(time_interval=time_interval)
    # PROGRAM METRICS or TRAJECTORY or CLEAR
    if program == TRAJECTORY:
        T = ExperimentalTrajectory(parallel=json.loads(str(parallel).lower()))
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
    elif metric_programs(program):
        for p in metric_programs(program):
            main_metrics(p, include_median=include_median and p == ACTIVITY, **kwargs_metrics)
    elif program == CLEAR:  # clear all data remove directories DANGEROUS!
        for path in [config.PLOTS_DIR, config.RESULTS_PATH]:  # VIS_DIR
            if os.path.isdir(path):
                shutil.rmtree(path)
                print("Removed directory: %s" % path)
    return None


def build_parser() -> argparse.ArgumentParser:
    """Command line parser for main."""
    parser = argparse.ArgumentParser(
        prog="python3 main.py",
        description="This program computes metrics and visualizations for fish trajectories, the results are saved in the directory %s" % config.DIR_CSV_LOCAL,
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
        help=f"Time interval in seconds (>= {MIN_TIME_INTERVAL}) to compute averages of metrics, or 'hour', or 'day' (one row per camera-day in day_<metric>.csv). Default: one row per camera-day in <metric>.csv",
        type=parse_time_interval,
        default=None,
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
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    tstart = time.time()
    config.create_directories()
    main(**vars(args))
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
