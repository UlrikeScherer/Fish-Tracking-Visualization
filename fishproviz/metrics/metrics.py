import fishproviz.config as config
from fishproviz.utils import (
    csv_of_the_day,
    get_days_in_order,
    get_fish2camera_map,
    all_error_filters,
)
from fishproviz.utils.tank_area_config import get_area_functions
from fishproviz.utils.transformation import pixel_to_cm, px2cm
from fishproviz.methods import tortuosity_of_chunk, distance_to_wall_chunk, mean_std, distance_to_object_chunk
from .results_to_csv import metric_result_to_csv
from .compute_metrics import (
    compute_step_lengths,
    compute_turning_angles,
    entropy_for_chunk,
)
import pandas as pd
import numpy as np
import bisect

from fishproviz.utils.utile import start_time_of_day_to_seconds
from fishproviz.utils.object_config import read_object_data_from_json

NDIM = 3


def num_of_spikes(steps):
    spike_places = get_spikes_filter(steps)
    return np.sum(spike_places), spike_places


def get_spikes_filter(steps):
    return steps > config.SPIKE_THRESHOLD


def update_filter_two_points(steps, filter_index):
    return filter_index[:-1] | filter_index[1:] | get_spikes_filter(steps)


def update_filter_three_points(steps, filter_index):
    filter_index = update_filter_two_points(steps, filter_index)
    return filter_index[:-1] | filter_index[1:]


def activity_mean_sd(steps, error_index):
    steps = steps[~error_index]
    if len(steps) == 0:
        return np.nan, np.nan
    return mean_std(steps)


def get_gaps_in_dataframes(frames):
    gaps_select = (frames[1:] - frames[:-1]) > 1
    return np.where(gaps_select)[0], gaps_select


def calculate_result_for_interval(data, split_index, avg_metric_f, error_index, NDIM=3):
    len_out = len(split_index) + 1
    mu_sd = np.zeros([len_out, NDIM], dtype=float)
    for i, (chunk, err_flt) in enumerate(
        zip(np.split(data, split_index), np.split(error_index, split_index))
    ):
        chunk = chunk[~err_flt]
        mu_sd[i, :-1] = avg_metric_f(chunk)
        mu_sd[i, -1] = len(chunk)
    return mu_sd


def entropy(data, frame_interval, error_index, area):
    return calculate_result_for_interval(
        data,
        frame_interval,
        lambda chunk: entropy_for_chunk(chunk, area),
        error_index,
        NDIM=2,
    )


def distance_to_wall(data, frame_interval, error_index, area):
    fish_key = area[0]
    return calculate_result_for_interval(
        data,
        frame_interval,
        lambda chunk: mean_std(
            px2cm(distance_to_wall_chunk(chunk, area[1]), fish_key=fish_key)
        ),
        error_index,
    )

def distance_to_object(data, frame_interval, error_index, area):
    fish_key = area[0]
    return calculate_result_for_interval(
        data,
        frame_interval,
        lambda chunk: mean_std(
            px2cm(distance_to_object_chunk(chunk, area[1], area[2], area[3]), fish_key=fish_key)
        ),
        error_index,
    )

def tortuosity(data, frame_interval, error_index):
    return calculate_result_for_interval(
        data,
        frame_interval,
        lambda chunk: mean_std(tortuosity_of_chunk(chunk)),
        error_index,
    )


def mean_std_median(chunk):
    if len(chunk) == 0:
        return (np.nan, np.nan, np.nan)
    return (*mean_std(chunk), np.percentile(chunk, 50))


def absolute_angles(data, frame_interval, filter_index):
    error_index = update_filter_three_points(compute_step_lengths(data), filter_index)
    return calculate_result_for_interval(
        np.abs(compute_turning_angles(data)), frame_interval, mean_std, error_index
    )


def activity(data, frame_interval, filter_index, include_median=False):
    steps = compute_step_lengths(data)
    filter_index = update_filter_two_points(steps, filter_index)
    return calculate_result_for_interval(
        steps,
        frame_interval,
        mean_std_median if include_median else mean_std,
        filter_index,
        NDIM=4 if include_median else 3,
    )


def turning_angle(data, frame_interval, filter_index):
    error_index = update_filter_three_points(compute_step_lengths(data), filter_index)
    return calculate_result_for_interval(
        compute_turning_angles(data), frame_interval, mean_std, error_index
    )


def metric_per_interval(
    fish_ids=None,
    time_interval=100,
    day_interval=None,
    metric=activity,
    write_to_csv=False,
    drop_out_of_scope=False,
    out_dim=3,
    include_median=False,
    print_logs=False,
):
    """
    Applies a given function to all fishes in fish_ids with the time_interval, for all days in the day_interval interval
    Args:
        fish_ids(list, int):    List of fish ids
        time_interval(int):     Time Interval to apply the metric to
        day_interval(Tuple):    Tuple of the first day to the last day, out of 0 to 29.
        metric(function):       A function to apply to the data, {activity, tortuosity, turning_angle,...}
        write_to_csv(bool):     Indicate weather the results should be written to a csv
    Returns:
        package:                dict of computed results, and meta information
    """
    if isinstance(fish_ids, int):
        fish_ids = [fish_ids]
    fish2camera = get_fish2camera_map()
    if fish_ids is None:
        fish_ids = [i for i in range(len(fish2camera))]
    area_func = get_area_functions()
    results = dict()
    package = dict(
        metric_name=metric.__name__,
        time_interval=time_interval,
        results=results,
    )
    metric_kwargs = dict()
    if include_median:
        out_dim = out_dim + 1
        metric_kwargs.update(include_median=include_median)

    if metric.__name__ in [ distance_to_object.__name__]:
        dict_ellipses = read_object_data_from_json()

    for i, fish in enumerate(fish_ids):
        camera_id, is_back = fish2camera[fish, 0], fish2camera[fish, 1] == config.BACK
        fish_key = "%s_%s" % (camera_id, fish2camera[fish, 1])
        day_dict = dict()
        days = get_days_in_order(
            interval=day_interval,
            camera=camera_id,
            is_back=is_back,
        )
        for j, day in enumerate(days):
            keys, data_in_batches = csv_of_the_day(
                camera_id,
                day,
                is_back=is_back,
                drop_out_of_scope=drop_out_of_scope,
                print_logs=print_logs,
            )  # True or False testing needed
            if len(data_in_batches) > 0:
                daytime_DF = (
                    start_time_of_day_to_seconds(day.split("_")[1])
                    * config.FRAMES_PER_SECOND
                )
                # use index frames to get the precis time of the data when averaging
                for k, dfi in zip(keys, data_in_batches):
                    dfi.index = dfi.FRAME + (int(k) * config.BATCH_SIZE)  # +daytime_DF
                df = pd.concat(data_in_batches)
                step = time_interval * config.FRAMES_PER_SECOND
                time_points = np.arange(0, int(df.index[-1]), step)
                split_by_interval_idx = [
                    bisect.bisect_left(df.index, h) for h in time_points[1:]
                ]
                data = df[["xpx", "ypx"]].to_numpy()
                area_tuple = (fish_key, area_func(fish_key))
                err_filter = all_error_filters(
                    data, area_tuple, fish_key=fish_key, day=day
                )

                if metric.__name__ in [  # metrics in pixels using the area config
                    entropy.__name__,
                    distance_to_wall.__name__,
                ]:
                    # DISTANCE TO WALL METRIC
                    result = metric(
                        data,
                        split_by_interval_idx,
                        err_filter,
                        area_tuple,
                        **metric_kwargs
                    )
                elif metric.__name__ in [ distance_to_object.__name__]:
                    # DISTANCE TO ELLIPSE
                    ellipse = dict_ellipses[fish_key][day]
                    ori_x = (ellipse["end_x"] + ellipse["origin_x"]) / 2
                    ori_y = (ellipse["end_y"] + ellipse["origin_y"]) / 2
                    r_x = (ellipse["end_x"] - ellipse["origin_x"]) / 2
                    r_y = (ellipse["end_y"] - ellipse["origin_y"]) / 2
                    result = metric(data,
                                    split_by_interval_idx,
                                    err_filter,
                                    (fish_key, np.array([ori_x, ori_y]), r_x, r_y),
                                    **metric_kwargs)
                else:
                    data_cm = pixel_to_cm(data, fish_key=fish_key)
                    result = metric(
                        data_cm, split_by_interval_idx, err_filter, **metric_kwargs
                    )
                # concat the results array with the index of df for every time_interval step
                result = pd.DataFrame(result, index=time_points)
                day_dict[day] = result
            else:
                day_dict[day] = pd.DataFrame(np.empty([0, out_dim]))

        results[fish_key] = day_dict
    if write_to_csv:
        metric_result_to_csv(**package)
    return package


def activity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=activity)


def turning_angle_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=turning_angle)


def absolute_angle_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=absolute_angles)


def tortuosity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=tortuosity)


def entropy_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=entropy, out_dim=2)


def distance_to_wall_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=distance_to_wall)

def distance_to_object_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=distance_to_object)
