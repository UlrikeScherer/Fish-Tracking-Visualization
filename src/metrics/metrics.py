from src.utils import all_error_filters
from src.config import (
    BACK,
    THRESHOLD_AREA_PX,
    SPIKE_THRESHOLD,
    N_DAYS,
    N_FISHES,
    N_SECONDS_OF_DAY,
    FRAMES_PER_SECOND,
    HOURS_PER_DAY,
    BLOCK,
    DATA_results,
    float_format,
    sep,
)
from src.utils import (
    csv_of_the_day,
    get_all_days_of_context,
    get_days_in_order,
    get_fish2camera_map,
    get_seconds_from_day,
)
from src.metrics.tank_area_config import get_area_functions
from src.utils.transformation import pixel_to_cm, px2cm
from src.methods import (
    activity,
    turning_angle,
    tortuosity_of_chunk,
    distance_to_wall_chunk,
    mean_std,
    absolute_angles,
)  # cython
import pandas as pd
import os
import numpy as np
from scipy.stats import entropy
from itertools import product
import matplotlib.pyplot as plt

from src.utils.utile import get_start_time_directory


def num_of_spikes(steps):
    spike_places = steps > SPIKE_THRESHOLD
    return np.sum(spike_places), spike_places


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


def entropy_for_chunk(chunk, area_tuple):
    """
    Args: chunk,
    area = tuple(fish_key, data)
    retrun entropy, std * 100
    """
    if chunk.shape[0] == 0:
        return np.nan, np.nan
    fish_key, area = area_tuple
    th = THRESHOLD_AREA_PX
    xmin, xmax = min(area[:, 0]) - th, max(area[:, 0]) + th
    ymin, ymax = min(area[:, 1]) - th, max(area[:, 1]) + th

    hist = np.histogram2d(
        chunk[:, 0],
        chunk[:, 1],
        bins=(18, 18),
        density=False,
        range=[[xmin, xmax], [ymin, ymax]],
    )[0]

    l_x, l_y = hist.shape
    if BACK in fish_key:  # if back use take the upper triangle -3
        tri = np.triu_indices(l_y, k=-3)
    else:  # if front the lower triangle +3
        tri = np.tril_indices(l_y, k=3)
    sum_hist = np.sum(hist)
    if sum_hist == 0:  #
        print(chunk[:10], xmin, ymin, xmax, ymax)
        print(
            "Warning for %s all %d data points where not in der range of histogram and removed"
            % (fish_key, chunk.shape[0])
        )
        return np.nan, np.nan
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
        print("entropy: ", entropy(hist[tri]))
        plt.plot(*area.T)
        plt.plot(*chunk.T, "*")
    return entropy(hist[tri]), np.std(hist[tri]) * 100


def average_by_metric(data, frame_interval, avg_metric_f, error_index):
    SIZE = data.shape[0]
    len_out = int(np.ceil(SIZE / frame_interval))
    mu_sd = np.zeros([len_out, 3], dtype=float)
    for i, s in enumerate(
        range(frame_interval, data.shape[0] + frame_interval, frame_interval)
    ):
        chunk = data[s - frame_interval : s][~error_index[s - frame_interval : s]]
        mu_sd[i, :2] = avg_metric_f(chunk)
        mu_sd[i, 2] = len(chunk)
    return mu_sd


def entropy_for_data(data, frame_interval, error_index, area):
    return average_by_metric(
        data, frame_interval, lambda chunk: entropy_for_chunk(chunk, area), error_index
    )


def distance_to_wall(data, frame_interval, error_index, area):
    return average_by_metric(
        data,
        frame_interval,
        lambda chunk: mean_std(px2cm(distance_to_wall_chunk(chunk, area[1]))),
        error_index,
    )


def tortuosity(data, frame_interval, error_index):
    return average_by_metric(
        data,
        frame_interval,
        lambda chunk: mean_std(tortuosity_of_chunk(chunk)),
        error_index,
    )


def metric_per_interval(
    fish_ids=[i for i in range(N_FISHES)],
    time_interval=100,
    day_interval=(0, N_DAYS),
    metric=activity,
    is_feeding=False,
    write_to_csv=False,
    drop_out_of_scope=False,
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
    fish2camera = get_fish2camera_map(is_feeding=is_feeding)
    area_func = get_area_functions()
    results = dict()
    package = dict(
        metric_name=metric.__name__,
        time_interval=time_interval,
        is_feeding=is_feeding,
        results=results,
    )
    for i, fish in enumerate(fish_ids):
        camera_id, is_back = fish2camera[fish, 0], fish2camera[fish, 1] == BACK
        fish_key = "%s_%s" % (camera_id, fish2camera[fish, 1])
        day_dict = dict()
        days = get_days_in_order(
            interval=day_interval,
            is_feeding=is_feeding,
            camera=camera_id,
            is_back=is_back,
        )
        for j, day in enumerate(days):
            keys, df_day = csv_of_the_day(
                camera_id,
                day,
                is_back=is_back,
                is_feeding=is_feeding,
                drop_out_of_scope=drop_out_of_scope,
            )  # True or False testing needed
            if len(df_day) > 0:
                df = pd.concat(df_day)
                data = df[["xpx", "ypx"]].to_numpy()
                area_tuple = (fish_key, area_func(fish_key, day=day))
                err_filter = all_error_filters(
                    data, area_tuple, fish_key=fish_key, day=day
                )
                if metric.__name__ in [
                    entropy_for_data.__name__,
                    distance_to_wall.__name__,
                ]:
                    # DISTANCE TO WALL METRIC
                    result = metric(
                        data, time_interval * FRAMES_PER_SECOND, err_filter, area_tuple
                    )
                else:
                    data = pixel_to_cm(df[["xpx", "ypx"]].to_numpy())
                    result = metric(data, time_interval * FRAMES_PER_SECOND, err_filter)
                day_dict[day] = result
            else:
                day_dict[day] = np.empty([0, 3])
        results[fish_key] = day_dict
    if write_to_csv:
        metric_data_to_csv(**package)
    return package


def metric_data_to_csv(
    results=None, metric_name=None, time_interval=None, is_feeding=None
):
    for i, (cam_pos, days) in enumerate(results.items()):
        time = list()
        for j, (day, value) in enumerate(days.items()):
            sec_day = get_seconds_from_day(day)
            time.extend(
                [
                    (day, t * time_interval + j * N_SECONDS_OF_DAY + sec_day)
                    for t in range(1, value.shape[0] + 1)
                ]
            )

        time_np = np.array(time)
        concat_r = np.concatenate(list(days.values()))
        if (
            time_np.ndim != concat_r.ndim
        ):  # if data for day is empty do not write an csv-entry
            print(
                "# WARNING: for %s time and results have unequal dimensions. time: %s results: %s "
                % (cam_pos, time_np, concat_r)
            )
            continue
        data = np.concatenate((time_np, concat_r), axis=1)
        df = pd.DataFrame(
            data, columns=["day", "time", "mean", "std", "number_of_valid_data_points"]
        )
        directory = get_results_directory(metric_name, is_feeding)
        df.to_csv(
            "%s/%s_%s.csv" % (directory, time_interval, cam_pos),
            sep=sep,
            float_format=float_format,
        )


def metric_per_hour_csv(
    results=None, metric_name=None, time_interval=None, is_feeding=None
):
    data_idx = np.array(list(product(results.keys(), range(HOURS_PER_DAY))))
    # initialize table of nan
    data = np.empty((data_idx.shape[0], N_DAYS))
    data.fill(np.nan)
    days = get_all_days_of_context()
    df = pd.DataFrame(data=data_idx, columns=["CAMERA_POSITION", "HOUR"])
    df_d = pd.DataFrame(data=data, columns=days)
    df_mean = pd.concat((df, df_d), axis=1)
    df_std = df_mean.copy()
    df_n_valid = df_mean.copy()
    for i, (cam_pos, fish) in enumerate(results.items()):
        for j, (day, day_data) in enumerate(fish.items()):
            idx = i * HOURS_PER_DAY
            k = idx + day_data.shape[0] - 1
            df_mean.loc[idx:k, day] = day_data[:, 0]
            df_std.loc[idx:k, day] = day_data[:, 1]
            df_n_valid.loc[idx:k, day] = day_data[:, 2]

    directory = get_results_directory(metric_name, is_feeding)
    df_mean.to_csv(
        "%s/%s_mean.csv" % (directory, metric_name),
        sep=sep,
        float_format=float_format,
    )
    df_std.to_csv(
        "%s/%s_std.csv" % (directory, metric_name),
        sep=sep,
        float_format=float_format,
    )
    df_n_valid.to_csv(
        "%s/%s_num_valid_datapoints.csv" % (directory, metric_name),
        sep=sep,
    )


def get_results_directory(metric_name, is_feeding):
    directory = "%s/%s/%s/%s/" % (
        DATA_results,
        BLOCK,
        get_start_time_directory(is_feeding),
        metric_name,
    )
    if os.path.exists(directory):
        return directory
    else:
        os.makedirs(directory)
        return directory


def activity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=activity)


def turning_angle_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=turning_angle)


def absolute_angle_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=absolute_angles)


def tortuosity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=tortuosity)


def entropy_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=entropy_for_data)


def distance_to_wall_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=distance_to_wall)


# function that returns a boolean deciding if it is odd or even
# python loop
