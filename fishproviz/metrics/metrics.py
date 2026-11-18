import inspect
from collections.abc import Sequence
from typing import Callable, Literal, Optional

import fishproviz.config as config
from fishproviz.utils import (
    csv_of_the_day,
    get_days_in_order,
    get_fish2camera_map,
    all_error_filters,
)
from fishproviz.utils.transformation import _area_functions, pixel_to_cm, px2cm
from fishproviz.methods import (
    tortuosity_of_chunk,
    distance_to_wall_chunk,
    mean_std,
    distance_to_object_chunk,
)
from .results_to_csv import metric_result_to_csv, require_raw_mode, AGGREGATED_RAW_METRICS, RAW_ROW_METRICS, RAW_STREAK_METRICS
from .compute_primitives import (
    compute_step_lengths,
    entropy_for_chunk,
)
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from fishproviz.utils.object_config import read_object_data_from_json
from fishproviz.utils.utile import DAY, feeding_times_start_end_dict, get_start_end_index

NDIM = 3


def num_of_spikes(steps: NDArray[np.float64]) -> tuple[int, NDArray[np.bool_]]:
    spike_places = get_spikes_filter(steps)
    return int(np.sum(spike_places)), spike_places


def get_spikes_filter(steps: NDArray[np.float64]) -> NDArray[np.bool_]:
    return steps > config.SPIKE_THRESHOLD


def update_filter_two_points(steps: NDArray[np.float64], filter_index: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return filter_index[:-1] | filter_index[1:] | get_spikes_filter(steps)


def update_filter_three_points(steps: NDArray[np.float64], filter_index: NDArray[np.bool_]) -> NDArray[np.bool_]:
    filter_index = update_filter_two_points(steps, filter_index)
    return filter_index[:-1] | filter_index[1:]


def steps_and_exclusions(data: NDArray[np.float64], filter_index: NDArray[np.bool_]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Step lengths between consecutive positions and the mask of steps excluded by an erroneous endpoint or a spike.

    Step i runs from row i to row i + 1; it is NaN where an endpoint is NaN.
    """
    steps = compute_step_lengths(data).astype(np.float64)
    return steps, update_filter_two_points(steps, filter_index)


def activity_mean_sd(steps: NDArray[np.float64], error_index: NDArray[np.bool_]) -> tuple[float, float]:
    """Mean and std of the steps not flagged in error_index (NaN, NaN when none remain)."""
    mean, std, _ = calculate_result_for_interval(steps, [], mean_std, error_index)[0]
    return mean, std


def calculate_result_for_interval(
    data: NDArray[np.float64],
    split_index: Sequence[int],
    avg_metric_f: Callable,
    error_index: NDArray[np.bool_],
    NDIM: int = 3,
    checkfornans: bool = False,
    averaging: bool = True,
) -> NDArray[np.float64]:
    """Applies avg_metric_f to every chunk of data split at split_index after dropping error_index rows.

    With averaging, returns (len(split_index) + 1, NDIM) rows of [*avg_metric_f(chunk), n]; without,
    returns avg_metric_f of every chunk concatenated into one column.
    """
    chunks = [chunk[~err_flt] for chunk, err_flt in zip(np.split(data, split_index), np.split(error_index, split_index))]
    if checkfornans:
        chunks = [chunk[~np.isnan(chunk)] for chunk in chunks]
    if not averaging:
        return np.concatenate([np.asarray(avg_metric_f(chunk), dtype=float) for chunk in chunks]).reshape((-1, 1))
    mu_sd = np.zeros([len(chunks), NDIM], dtype=float)
    for i, chunk in enumerate(chunks):
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


def distance_to_wall(data, frame_interval, error_index, area, *, unaveraged: bool = False):
    """Mean distance to the tank wall in cm per interval, or every point's distance when unaveraged."""
    fish_key = area[0]

    def dtw(chunk):
        return px2cm(distance_to_wall_chunk(chunk, area[1]), fish_key=fish_key)

    def avg_func(chunk):
        return mean_std(dtw(chunk))

    return calculate_result_for_interval(
        data.astype("double"),
        frame_interval,
        dtw if unaveraged else avg_func,
        error_index,
        averaging=not unaveraged,
    )


def distance_to_object(data, frame_interval, error_index, area):
    fish_key = area[0]
    return calculate_result_for_interval(
        data.astype("double"),
        frame_interval,
        lambda chunk: mean_std(
            px2cm(
                distance_to_object_chunk(chunk, area[1], area[2], area[3]),
                fish_key=fish_key,
            )
        ),
        error_index,
    )


def tortuosity(data, frame_interval, error_index):
    return calculate_result_for_interval(
        data.astype("double"),
        frame_interval,
        lambda chunk: mean_std(tortuosity_of_chunk(chunk)),
        error_index,
    )


def mean_std_median(chunk):
    if len(chunk) == 0:
        return (np.nan, np.nan, np.nan)
    return (*mean_std(chunk.astype("double")), np.percentile(chunk, 50))


def _step_metric(
    data: NDArray[np.float64],
    frame_interval: Sequence[int],
    filter_index: NDArray[np.bool_],
    *,
    include_median: bool,
    unaveraged: bool,
) -> NDArray[np.float64]:
    """Shared implementation of activity and step_length.

    Returns (n_intervals, 3) rows of [mean, std, n] (4 with the median before n), or when unaveraged
    an (n_steps, 3) array of [row, step, excluded] in time order, row being the step's start row.
    """
    if include_median and unaveraged:
        raise ValueError("include_median applies to averaged output only; raw (unaveraged) step output has no median")
    steps, excluded = steps_and_exclusions(data, filter_index)
    if unaveraged:
        return np.column_stack([np.arange(len(steps)), steps, excluded]).astype(np.float64)
    return calculate_result_for_interval(
        steps,
        frame_interval,
        mean_std_median if include_median else mean_std,
        excluded,
        NDIM=4 if include_median else 3,
    )


def activity(
    data: NDArray[np.float64],
    frame_interval: Sequence[int],
    filter_index: NDArray[np.bool_],
    include_median: bool = False,
    *,
    unaveraged: bool = False,
) -> NDArray[np.float64]:
    """Step length statistics per interval, optionally with the median (see _step_metric)."""
    return _step_metric(data, frame_interval, filter_index, include_median=include_median, unaveraged=unaveraged)


def step_length(
    data: NDArray[np.float64],
    frame_interval: Sequence[int],
    filter_index: NDArray[np.bool_],
    *,
    unaveraged: bool = False,
) -> NDArray[np.float64]:
    """Step length statistics per interval (see _step_metric)."""
    return _step_metric(data, frame_interval, filter_index, include_median=False, unaveraged=unaveraged)


def interval_split(frames: pd.Index, step_frames: Optional[int]) -> tuple[NDArray[np.int64], list[int]]:
    """Bin start frames from the first loaded frame, and the data rows where every later bin begins.

    step_frames None gives a single bin covering every loaded frame. The last frame never opens a bin of its own.
    """
    first, last = int(frames[0]), int(frames[-1])
    time_points = np.array([first]) if step_frames is None else np.arange(first, max(last, first + 1), step_frames)
    return time_points, np.searchsorted(frames, time_points[1:], side="left").tolist()


def _compute_metric(
    metric: Callable,
    data: NDArray[np.float64],
    split_index: list[int],
    err_filter: NDArray[np.bool_],
    area_tuple: tuple,
    ellipse: Optional[dict],
    metric_kwargs: dict,
) -> NDArray[np.float64]:
    """Calls metric with the coordinate space and extra inputs it expects."""
    fish_key = area_tuple[0]
    if metric.__name__ in [entropy.__name__, distance_to_wall.__name__]:  # metrics in pixels using the area config
        return metric(data, split_index, err_filter, area_tuple, **metric_kwargs)
    if metric.__name__ == distance_to_object.__name__:
        ori_x = (ellipse["end_x"] + ellipse["origin_x"]) / 2
        ori_y = (ellipse["end_y"] + ellipse["origin_y"]) / 2
        r_x = (ellipse["end_x"] - ellipse["origin_x"]) / 2
        r_y = (ellipse["end_y"] - ellipse["origin_y"]) / 2
        return metric(data, split_index, err_filter, (fish_key, np.array([ori_x, ori_y]), r_x, r_y), **metric_kwargs)
    data_cm = pixel_to_cm(data, fish_key=fish_key)
    if "area" in inspect.signature(metric).parameters:
        return metric(data_cm, split_index, err_filter, area=area_tuple, data_px=data, **metric_kwargs)
    return metric(data_cm, split_index, err_filter, **metric_kwargs)


def _result_table(
    metric_name: str,
    result: NDArray[np.float64],
    frames: pd.Index,
    data: NDArray[np.float64],
    time_points: NDArray[np.int64],
    every_point: bool,
) -> pd.DataFrame:
    """Turns a metric result into the day's table, attaching frame numbers and tracker coordinates to raw rows."""
    if every_point and metric_name in RAW_ROW_METRICS:
        # result columns: [row, value, excluded]
        rows = result[:, 0].astype(int)
        return pd.DataFrame(
            {
                "frame": frames[rows],
                "xpx": data[rows, 0],
                "ypx": data[rows, 1],
                metric_name: result[:, 1],
                "excluded": result[:, 2].astype(bool),
            }
        )
    if every_point and metric_name in RAW_STREAK_METRICS:
        # result columns: [row_start, row_end, side, length]; rows become day-level frame numbers
        return pd.DataFrame(
            {
                "frame_start": frames[result[:, 0].astype(int)],
                "frame_end": frames[result[:, 1].astype(int)],
                "side": result[:, 2].astype(int),
                "length": result[:, 3].astype(int),
            }
        )
    if every_point and metric_name in AGGREGATED_RAW_METRICS:
        return pd.DataFrame({metric_name: np.add.reduceat(result, np.arange(0, len(result), config.AGGREGATE_EVERY_N)).reshape((-1,))})
    return pd.DataFrame(result, index=time_points)


def _day_frame(keys: list[str], batches: list[pd.DataFrame], start_end_times: Optional[dict], day: str, fish_key: str) -> pd.DataFrame:
    """Concatenates the day's batches, keeping only the frames inside each batch's feeding/trial window."""
    windowed = []
    for key, batch in zip(keys, batches):
        start_idx, end_idx = get_start_end_index(start_end_times, day, key, fish_key)
        windowed.append(batch[batch.FRAME.between(start_idx, end_idx)])
    return pd.concat(windowed) if windowed else pd.DataFrame(columns=["FRAME", "xpx", "ypx"])


def _metric_for_day(
    metric: Callable,
    df: pd.DataFrame,
    area_tuple: tuple,
    day: str,
    *,
    step_frames: Optional[int],
    ellipse: Optional[dict],
    metric_kwargs: dict,
    every_point: bool,
    out_dim: int,
) -> pd.DataFrame:
    """The day's result table; an empty frame of out_dim columns when no frame of the day was loaded."""
    if df.empty:
        return pd.DataFrame(np.empty([0, out_dim]))
    # df is indexed by absolute day frame, used for the precise time when averaging
    time_points, split_by_interval_idx = interval_split(df.index, step_frames)
    data = df[["xpx", "ypx"]].to_numpy()
    err_filter = all_error_filters(data, area_tuple, fish_key=area_tuple[0], day=day)
    result = _compute_metric(metric, data, split_by_interval_idx, err_filter, area_tuple, ellipse, metric_kwargs)
    return _result_table(metric.__name__, result, df.index, data, time_points, every_point)


def metric_per_interval(
    fish_ids: Optional[list[int]] = None,
    time_interval: int | Literal["day"] = 100,
    day_interval: Optional[tuple[int, int]] = None,
    metric: Callable = activity,
    write_to_csv: bool = False,
    include_median: bool = False,
    print_logs: bool = False,
    is_feeding: bool = False,
    is_novel_object: bool = False,
    is_sociability: bool = False,
    all_points: bool = False,
    every_point: bool = False,
) -> dict:
    """
    Applies a given function to all fishes in fish_ids with the time_interval, for all days in the day_interval interval
    Args:
        fish_ids(list, int):    List of fish ids
        time_interval:          Bin width in seconds, or DAY for one bin per day written as day_<metric>.csv
        day_interval(Tuple):    Tuple of the first day to the last day, out of 0 to 29.
        metric(function):       A function to apply to the data, {activity, tortuosity, turning_angle,...}
        write_to_csv(bool):     Indicate weather the results should be written to a csv
        all_points(bool):       One bin per day written as <metric>.csv; time_interval is then ignored
        every_point(bool):      Raw output (rows, steps, angles, streaks); raises for metrics without one
    Returns:
        package:                dict of computed results, and meta information
    """
    if isinstance(fish_ids, int):
        fish_ids = [fish_ids]
    fish2camera = get_fish2camera_map()
    if fish_ids is None:
        fish_ids = [i for i in range(len(fish2camera))]
    if every_point:
        require_raw_mode([metric.__name__])
    area_func = _area_functions()
    results = dict()
    package = dict(
        metric_name=metric.__name__,
        time_interval=time_interval,
        results=results,
    )
    metric_kwargs = dict()
    if "unaveraged" in inspect.signature(metric).parameters:
        # explicit, so raw mode follows every_point only and never the global UNAVERAGED flag
        metric_kwargs.update(unaveraged=every_point)
    # columns of an empty day: [entropy, n] for entropy, [mean, std, (median,) n] otherwise
    out_dim = (2 if metric.__name__ == entropy.__name__ else 3) + int(include_median)
    if include_median:
        metric_kwargs.update(include_median=include_median)
    step_frames = None if all_points or time_interval == DAY else time_interval * config.FRAMES_PER_SECOND

    dict_ellipses = read_object_data_from_json() if metric.__name__ == distance_to_object.__name__ else None

    for _, fish in enumerate(fish_ids):
        camera_id, is_back = fish2camera[fish, 0], fish2camera[fish, 1] == config.BACK
        fish_key = "%s_%s" % (camera_id, fish2camera[fish, 1])
        day_dict = dict()
        days = get_days_in_order(
            interval=day_interval,
            camera=camera_id,
            is_back=is_back,
        )
        start_end_times = feeding_times_start_end_dict(is_feeding, is_novel_object, is_sociability)

        for _, day in enumerate(days):
            keys, data_in_batches = csv_of_the_day(
                camera_id,
                day,
                is_back=is_back,
                print_logs=print_logs,
            )
            day_dict[day] = _metric_for_day(
                metric,
                _day_frame(keys, data_in_batches, start_end_times, day, fish_key),
                (fish_key, area_func(fish_key)),
                day,
                step_frames=step_frames,
                ellipse=dict_ellipses[fish_key][day] if dict_ellipses is not None else None,
                metric_kwargs=metric_kwargs,
                every_point=every_point,
                out_dim=out_dim,
            )

        results[fish_key] = day_dict
    if write_to_csv:
        metric_result_to_csv(**package, all_points=all_points, every_point=every_point)
    return package


def activity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=activity)


def step_length_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=step_length)


def tortuosity_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=tortuosity)


def entropy_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=entropy)


def distance_to_wall_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=distance_to_wall)


def distance_to_object_per_interval(*args, **kwargs):
    return metric_per_interval(*args, **kwargs, metric=distance_to_object)
