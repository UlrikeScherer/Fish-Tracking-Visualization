from fishproviz.utils import all_error_filters
from fishproviz.config import (
    BACK,
    THRESHOLD_AREA_PX,
    SPIKE_THRESHOLD,
    FRAMES_PER_SECOND,
)
from fishproviz.utils import (
    csv_of_the_day,
    get_days_in_order,
    get_fish2camera_map,
)
from fishproviz.utils.tank_area_config import get_area_functions
from fishproviz.utils.transformation import pixel_to_cm, px2cm
from fishproviz.metrics.results_to_csv import metric_data_to_csv
from src.methods import (
    tortuosity_of_chunk,
    turning_directions,
    distance_to_wall_chunk,
    mean_std,
    calc_steps,
)  # cython
import pandas as pd
import numpy as np
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt

NDIM = 3


def num_of_spikes(steps):
    spike_places = get_spikes_filter(steps)
    return np.sum(spike_places), spike_places


def get_spikes_filter(steps):
    return steps > SPIKE_THRESHOLD


def update_filter_two_points(steps, filter_index):
    return filter_index[:-1] | filter_index[1:] | get_spikes_filter(steps)


def update_filter_three_points(steps, filter_index):
    filter_index = update_filter_two_points(steps, filter_index)
    return filter_index[:-1] | filter_index[1:]


def calc_length_of_steps(batchxy):
    xsq = (batchxy[1:, 0] - batchxy[:-1, 0]) ** 2
    ysq = (batchxy[1:, 1] - batchxy[:-1, 1]) ** 2
    c = np.sqrt(ysq + xsq)
    return c


def activity_mean_sd(steps, error_index):
    steps = steps[~error_index]
    if len(steps) == 0:
        return np.nan, np.nan
    return mean_std(steps)


def get_gaps_in_dataframes(frames):
    gaps_select = (frames[1:] - frames[:-1]) > 1
    return np.where(gaps_select)[0], gaps_select


def calc_step_per_frame(batchxy, frames):
    """This function calculates the eucleadian step length in centimeters per FRAME, this is useful as a speed measurement after the removal of erroneous data points."""
    frame_dist = frames[1:] - frames[:-1]
    c = calc_length_of_steps(batchxy) / frame_dist
    return c


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def determinant(v, w):
    """Determinant of two vectors."""
    return v[0] * w[1] - v[1] * w[0]


def direction_angle(v, w):
    """Return the angle between v,w anti clockwise from direction v to m."""
    cos = np.dot(v, w)
    r = np.arccos(np.clip(cos, -1, 1))
    det = determinant(v, w)
    if det < 0:
        return r
    else:
        return -r


def angle(v, w):
    cos = np.dot(v, w)
    return np.arccos(np.clip(cos, -1, 1))


def sum_of_angles(df):
    y = df.ypx.array[1:] - df.ypx.array[:-1]
    x = df.xpx.array[1:] - df.xpx.array[:-1]
    if len(x) == 0:
        return 0
    u = unit_vector([x[0], y[0]])
    sum_alpha = 0
    for i in range(1, y.size):
        v = unit_vector([x[i], y[i]])
        if np.any(np.isnan(v)):
            continue
        if np.any(np.isnan(u)):
            u = v
            continue
        alpha = direction_angle(u, v)
        sum_alpha += alpha
        u = v
    return sum_alpha


def entropy_heatmap(chunk, area, bins=(18, 18)):
    """Calculate the 2D histogram of the chunk"""
    th = THRESHOLD_AREA_PX
    xmin, xmax = min(area[:, 0]) - th, max(area[:, 0]) + th
    ymin, ymax = min(area[:, 1]) - th, max(area[:, 1]) + th

    return np.histogram2d(
        chunk[:, 0],
        chunk[:, 1],
        bins=bins,
        density=False,
        range=[[xmin, xmax], [ymin, ymax]],
    )[0]


def entropy_for_chunk(chunk, area_tuple):
    """
    Args: chunk,
    area = tuple(fish_key, data)
    retrun entropy
    """
    if chunk.shape[0] == 0:
        return np.nan
    fish_key, area = area_tuple

    hist = entropy_heatmap(chunk, area)
    l_x, l_y = hist.shape
    if BACK in fish_key:  # if back use take the upper triangle -3
        tri = np.triu_indices(l_y, k=-3)
    else:  # if front the lower triangle +3
        tri = np.tril_indices(l_y, k=3)
    sum_hist = np.sum(hist)
    if sum_hist == 0:  #
        print(chunk[:10])
        print(
            "Warning for %s all %d data points where not in der range of histogram and removed"
            % (fish_key, chunk.shape[0])
        )
        return np.nan
    if chunk.shape[0] > sum_hist:
        # print(chunk[:10])
        print(
            "Warning for %s %d out of %d data points where not in der range of histogram and removed"
            % (fish_key, chunk.shape[0] - sum_hist, chunk.shape[0])
        )
    if sum_hist > np.sum(hist[tri]):
        print(
            "Warning for %s the selected area for entropy has lost some points: "
            % fish_key,
            "sum hist: ",
            np.sum(hist),
            "sum selection: ",
            sum(hist[tri]),
            "\n",
            fish_key,
        )
        print("entropy: ", scipy_stats.entropy(hist[tri]))
        plt.plot(*area.T)
        plt.plot(*chunk.T, "*")
    return scipy_stats.entropy(hist[tri])


def calculate_result_for_interval(
    data, frame_interval, avg_metric_f, error_index, NDIM=3
):
    SIZE = data.shape[0]
    len_out = int(np.ceil(SIZE / frame_interval))
    mu_sd = np.zeros([len_out, NDIM], dtype=float)
    for i, s in enumerate(
        range(frame_interval, data.shape[0] + frame_interval, frame_interval)
    ):
        chunk = data[s - frame_interval : s][~error_index[s - frame_interval : s]]
        mu_sd[i, :-1] = avg_metric_f(chunk)
        mu_sd[i, -1] = len(chunk)
    return mu_sd


def mean_std_for_interval(results, frame_interval, filtered, NDIM=3):
    len_out = int(np.ceil(results.size / frame_interval))
    mu_sd = np.zeros([len_out, NDIM], dtype=float)
    for i, s in enumerate(range(0, results.size, frame_interval)):
        chunk = results[s : s + frame_interval][
            filtered[s : s + frame_interval]
        ]  # select chunk and filter it
        mu_sd[i, :-1] = mean_std(chunk)
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
        lambda chunk: mean_std(px2cm(distance_to_wall_chunk(chunk, area[1]), fish_key=fish_key)),
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
    filter_index = ~update_filter_three_points(calc_steps(data), filter_index)
    return mean_std_for_interval(
        np.abs(turning_directions(data)), frame_interval, filter_index
    )


def activity(data, frame_interval, filter_index, include_median=False):
    steps = calc_steps(data)
    filter_index = update_filter_two_points(steps, filter_index)
    return calculate_result_for_interval(
        steps,
        frame_interval,
        mean_std_median if include_median else mean_std,
        filter_index,
        NDIM=4 if include_median else 3,
    )


def turning_angle(data, frame_interval, filter_index):
    filter_index = ~update_filter_three_points(calc_steps(data), filter_index)
    return mean_std_for_interval(turning_directions(data), frame_interval, filter_index)


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

    for i, fish in enumerate(fish_ids):
        camera_id, is_back = fish2camera[fish, 0], fish2camera[fish, 1] == BACK
        fish_key = "%s_%s" % (camera_id, fish2camera[fish, 1])
        day_dict = dict()
        days = get_days_in_order(
            interval=day_interval,
            camera=camera_id,
            is_back=is_back,
        )
        for j, day in enumerate(days):
            keys, df_day = csv_of_the_day(
                camera_id,
                day,
                is_back=is_back,
                drop_out_of_scope=drop_out_of_scope,
                print_logs=print_logs
            )  # True or False testing needed
            if len(df_day) > 0:
                df = pd.concat(df_day)
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
                        time_interval * FRAMES_PER_SECOND,
                        err_filter,
                        area_tuple,
                        **metric_kwargs
                    )
                else:
                    data = pixel_to_cm(df[["xpx", "ypx"]].to_numpy(), fish_key=fish_key)
                    result = metric(
                        data,
                        time_interval * FRAMES_PER_SECOND,
                        err_filter,
                        **metric_kwargs
                    )
                day_dict[day] = result
            else:
                day_dict[day] = np.empty([0, out_dim])
        results[fish_key] = day_dict
    if write_to_csv:
        metric_data_to_csv(**package)
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


# function that returns a boolean deciding if it is odd or even
# python loop
