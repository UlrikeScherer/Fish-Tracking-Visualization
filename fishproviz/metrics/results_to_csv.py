import os
import numpy as np
import pandas as pd

# direct import to allow global config changes
import fishproviz.config as config
from fishproviz.utils.utile import (
    get_interval_name_from_seconds,
    get_seconds_from_day,
)

csv_columns_time = ["day", "time"]
csv_columns_results = ["mean", "std", "median"]
num_datapoints = "num_valid_points"


def get_csv_columns_from_results_dim(dimension, metric_name):
    if dimension == 2:
        return [metric_name, num_datapoints]
    elif dimension == 3:
        return [*csv_columns_results[:2], num_datapoints]
    elif dimension == 4:
        return [*csv_columns_results[:3], num_datapoints]
    else:
        raise ValueError("dimension must be either 2, 3 or 4, but was %s" % dimension)


def metric_result_to_csv(results=None, metric_name=None, time_interval=None):
    columns = ["cam_pos", "day", "df_index"]
    interval_name = get_interval_name_from_seconds(time_interval)
    df_sum = pd.concat(
        [
            pd.concat(
                [pd.DataFrame(day_data) for day_data in fish_data.values()],
                keys=fish_data.keys(),
            )
            if len(fish_data) > 0
            else None
            for fish_data in results.values()
        ],
        keys=results.keys(),
    )
    measures = get_csv_columns_from_results_dim(df_sum.shape[1], metric_name)
    df_sum = df_sum.reset_index()
    df_sum.columns = [*columns, *measures]
    # update the type of the columns
    df_sum[num_datapoints] = df_sum[num_datapoints].astype(int)
    if "hour" == interval_name:
        df_sum["hour"] = df_sum["df_index"] // (
            time_interval * config.FRAMES_PER_SECOND
        )
    df_sum.to_csv(
        get_filename_for_metric_csv(metric_name, interval_name),
        float_format=config.float_format,
        sep=config.sep,
    )


# generates csv file name for a metric and a time interval
def get_filename_for_metric_csv(
    metric_name, time_interval, measure_name=None, cam_pos=None
):
    """
    :param metric_name: name of the metric
    :param time_interval: in seconds our string representation of the time interval
    :param measure_name:
    :param cam_pos:
    :return: filename for csv file
    """
    directory = get_results_directory(metric_name)
    if measure_name:
        if metric_name == measure_name:
            return "%s/%s_%s.csv" % (directory, time_interval, metric_name)
        return "%s/%s_%s_%s.csv" % (directory, time_interval, metric_name, measure_name)
    elif cam_pos:
        return "%s/%s_%s.csv" % (directory, time_interval, cam_pos)
    else:
        return "%s/%s_%s.csv" % (directory, time_interval, metric_name)


def get_results_directory(metric_name):
    directory = "%s/%s/%s" % (
        config.RESULTS_PATH,
        config.PROJECT_ID,
        metric_name,
    )
    if os.path.exists(directory):
        return directory
    else:
        os.makedirs(directory)
        return directory
