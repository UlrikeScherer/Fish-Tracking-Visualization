from itertools import product
import os
import numpy as np
import pandas as pd
from src.config import (
    CAM_POS,
    N_SECONDS_OF_DAY,
    HOURS_PER_DAY,
    N_SECONDS_PER_HOUR,
    PROJECT_ID,
    RESULTS_PATH,
    float_format,
    sep,
)
from src.utils.utile import (
    get_interval_name_from_seconds,
    get_start_time_directory,
    get_seconds_from_day,
    get_all_days_of_context,
)

csv_columns_time = ["day", "time"]
csv_columns_results = ["mean", "std", "median"]
num_datapoints = "number_of_valid_data_points"


def get_csv_columns_from_results_dim(dimension, metric_name):
    if dimension == 2:
        return [metric_name, num_datapoints]
    elif dimension == 3:
        return [*csv_columns_results[:2], num_datapoints]
    elif dimension == 4:
        return [*csv_columns_results[:3], num_datapoints]
    else:
        raise ValueError("dimension must be either 2, 3 or 4, but was %s" % dimension)


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
            data,
            columns=csv_columns_time
            + get_csv_columns_from_results_dim(concat_r.shape[1], metric_name),
        )
        df.to_csv(
            get_filename_for_metric_csv(
                metric_name, time_interval, is_feeding, cam_pos=cam_pos
            ),
            sep=sep,
            float_format=float_format,
        )


def metric_per_hour_csv(
    results=None, metric_name=None, time_interval=None, is_feeding=None
):
    columns = [CAM_POS]
    entities_per_day = 1
    interval_name = get_interval_name_from_seconds(time_interval)
    if time_interval == N_SECONDS_PER_HOUR * HOURS_PER_DAY:
        data_idx = np.array(list(results.keys()))
    elif time_interval == N_SECONDS_PER_HOUR:
        entities_per_day = HOURS_PER_DAY
        columns.append("HOUR")
        data_idx = np.array(list(product(results.keys(), range(HOURS_PER_DAY))))
    else:
        raise ValueError(
            "time_interval must be either %s or %s, but was %s"
            % (N_SECONDS_PER_HOUR * HOURS_PER_DAY, N_SECONDS_PER_HOUR, time_interval)
        )
    # initialize table of nan
    days = get_all_days_of_context()
    data = np.empty((data_idx.shape[0], len(days)))
    data.fill(np.nan)
    df = pd.DataFrame(data=data_idx, columns=columns)
    df_d = pd.DataFrame(data=data, columns=days)
    first_key = next(iter(results.keys()))
    first_day = next(iter(results[first_key].keys()))
    measures = get_csv_columns_from_results_dim(
        results[first_key][first_day].shape[1], metric_name
    )
    dfs_measures = [pd.concat((df, df_d), axis=1) for m in measures]
    for i, (cam_pos, fish) in enumerate(results.items()):
        for j, (day, day_data) in enumerate(fish.items()):
            idx = i * entities_per_day
            k = idx + day_data.shape[0] - 1
            for mi in range(len(dfs_measures)):
                dfs_measures[mi].loc[idx:k, day] = day_data[:, mi]

    for mi, df_m in enumerate(dfs_measures):
        df_m.to_csv(
            get_filename_for_metric_csv(
                metric_name, interval_name, is_feeding, measure_name=measures[mi]
            ),
            sep=sep,
            float_format=float_format,
        )


# generates csv file name for a metric and a time interval
def get_filename_for_metric_csv(
    metric_name, time_interval, is_feeding, measure_name=None, cam_pos=None
):
    """
    :param metric_name: name of the metric
    :param time_interval: in seconds our string representation of the time interval
    :param is_feeding:
    :param measure_name:
    :param cam_pos:
    :return: filename for csv file
    """
    directory = get_results_directory(metric_name, is_feeding)
    if measure_name:
        if metric_name == measure_name:
            return "%s/%s_%s.csv" % (directory, time_interval, metric_name)
        return "%s/%s_%s_%s.csv" % (directory, time_interval, metric_name, measure_name)
    elif cam_pos:
        return "%s/%s_%s.csv" % (directory, time_interval, cam_pos)
    else:
        return "%s/%s_%s.csv" % (directory, time_interval, metric_name)


def get_results_directory(metric_name, is_feeding):
    directory = "%s/%s/%s" % (
        RESULTS_PATH,
        PROJECT_ID,
        metric_name,
    )
    if os.path.exists(directory):
        return directory
    else:
        os.makedirs(directory)
        return directory
