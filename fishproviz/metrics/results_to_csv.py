import os
from collections.abc import Sequence
from typing import Literal, Optional

import pandas as pd

# direct import to allow global config changes
import fishproviz.config as config
from fishproviz.utils.utile import HOUR, get_interval_name_from_seconds

csv_columns_time = ["day", "time"]
csv_columns_results = ["mean", "std", "median"]
num_datapoints = "num_valid_points"
num_out_of_range = "num_out_of_range"
ENTROPY = "entropy"
TURNING_ANGLE_STREAK_LENGTH = "turning_angle_streak_length"
RAW_ANGLE_METRICS = ("turning_angle", "absolute_angles")
RAW_STREAK_METRICS = (TURNING_ANGLE_STREAK_LENGTH,)
# metrics whose raw output is one row per path window: frame_start, frame_end, path_cm, straight_cm, <metric>
RAW_WINDOW_METRICS = ("tortuosity",)
TANGLE_METRICS = (*RAW_ANGLE_METRICS, *RAW_STREAK_METRICS)
STEP_METRICS = ("activity", "step_length")
# metrics whose raw output is one row per angle or step: frame, xpx, ypx, <metric>, excluded
RAW_ROW_METRICS = (*RAW_ANGLE_METRICS, *STEP_METRICS)
# metrics whose raw output is per-point values summed over AGGREGATE_EVERY_N windows
AGGREGATED_RAW_METRICS = ("distance_to_wall",)
RAW_METRICS = (*RAW_ROW_METRICS, *RAW_STREAK_METRICS, *RAW_WINDOW_METRICS, *AGGREGATED_RAW_METRICS)


def require_raw_mode(metric_names: Sequence[str]) -> None:
    """Raise naming every metric in metric_names that has no raw (unaveraged) output."""
    without_raw = [name for name in metric_names if name not in RAW_METRICS]
    if without_raw:
        raise ValueError(f"Raw output (UNAVERAGED=1 / every_point) is not available for {without_raw}; it exists for {list(RAW_METRICS)}. Set UNAVERAGED=0 or run another metric")


def get_csv_columns_from_results_dim(dimension: int, metric_name: str) -> list[str]:
    """Names of the averaged result columns of metric_name, whose results have dimension columns."""
    if metric_name == ENTROPY:
        if dimension != 3:
            raise ValueError(f"entropy results must have 3 columns [entropy, n, n_out_of_range], but had {dimension}")
        return [metric_name, num_datapoints, num_out_of_range]
    if dimension == 3:
        return [*csv_columns_results[:2], num_datapoints]
    if dimension == 4:
        return [*csv_columns_results[:3], num_datapoints]
    raise ValueError(f"dimension must be either 3 or 4 for metric '{metric_name}', but was {dimension}")


def _check_results(results: dict[str, dict[str, pd.DataFrame]], metric_name: str) -> None:
    """Raise when no fish, or any single fish, produced no rows of data for the metric."""
    rows_per_fish = {fish_key: sum(len(day_data) for day_data in fish_data.values()) for fish_key, fish_data in results.items()}
    if not results or not any(rows_per_fish.values()):
        raise ValueError(f"No fish produced any data for metric '{metric_name}'; check the data directories and the day folders of every camera")
    for fish_key, n_rows in rows_per_fish.items():
        if n_rows == 0:
            raise ValueError(f"Fish '{fish_key}' produced no data in any day for metric '{metric_name}'; check its day folders or exclude it with -fid")


def metric_result_to_csv(
    results: dict[str, dict[str, pd.DataFrame]],
    metric_name: str,
    time_interval: Optional[int | Literal["day"]] = None,
    all_points: bool = False,
    every_point: bool = False,
) -> None:
    """Writes the averaged results to one csv per metric and interval, or raw results to one csv per camera-day."""
    _check_results(results, metric_name)
    if not every_point:
        columns = ["cam_pos", "day", "df_index"]
        interval_name = None if all_points else get_interval_name_from_seconds(time_interval)
        df_sum = pd.concat(
            [pd.concat([pd.DataFrame(day_data) for day_data in fish_data.values()], keys=fish_data.keys()) for fish_data in results.values()],
            keys=results.keys(),
        )
        measures = get_csv_columns_from_results_dim(df_sum.shape[1], metric_name)
        df_sum = df_sum.reset_index()
        df_sum.columns = [*columns, *measures]
        # update the type of the columns
        count_columns = [column for column in (num_datapoints, num_out_of_range) if column in measures]
        df_sum[count_columns] = df_sum[count_columns].astype(int)

        if interval_name == HOUR:
            # bin ordinal within the day, counted from the day's first loaded frame
            first_frame = df_sum.groupby(["cam_pos", "day"])["df_index"].transform("min")
            df_sum["hour"] = (df_sum["df_index"] - first_frame) // (time_interval * config.FRAMES_PER_SECOND)
        df_sum.to_csv(
            get_filename_for_metric_csv(metric_name, interval_name, all_points=all_points),
            float_format="%.10f",
            sep=config.sep,
        )
    else:
        directory = get_results_directory(metric_name)
        suffix = f"_skip{config.TANGLE_N_SKIP}" if metric_name in TANGLE_METRICS else ""
        no_index = metric_name not in AGGREGATED_RAW_METRICS
        for key in results.keys():
            for subkey in results[key].keys():
                subdir = f"{directory}/raw"
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                if len(results[key][subkey]) > 0:
                    results[key][subkey].to_csv(
                        f"{subdir}/{key}_{subkey}{suffix}.csv",
                        index=not no_index,
                        index_label="i",
                    )


# generates csv file name for a metric and a time interval
def get_filename_for_metric_csv(
    metric_name,
    time_interval,
    measure_name=None,
    cam_pos=None,
    all_points: bool = False,
):
    """
    :param metric_name: name of the metric
    :param time_interval: in seconds our string representation of the time interval
    :param measure_name:
    :param cam_pos:
    :return: filename for csv file
    """
    directory = get_results_directory(metric_name)
    if metric_name in TANGLE_METRICS:
        metric_name = metric_name + f"_skip{config.TANGLE_N_SKIP}"
    if all_points:
        return "%s/%s.csv" % (directory, metric_name)
    else:
        if measure_name:
            if metric_name == measure_name:
                return "%s/%s_%s.csv" % (directory, time_interval, metric_name)
            return "%s/%s_%s_%s.csv" % (
                directory,
                time_interval,
                metric_name,
                measure_name,
            )
        elif cam_pos:
            return "%s/%s_%s.csv" % (directory, time_interval, cam_pos)
        else:
            return "%s/%s_%s.csv" % (directory, time_interval, metric_name)


def get_results_directory(metric_name):
    directory = "%s/%s" % (
        config.RESULTS_PATH,
        metric_name,
    )
    if os.path.exists(directory):
        return directory
    else:
        os.makedirs(directory)
        return directory
