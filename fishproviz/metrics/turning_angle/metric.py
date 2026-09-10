from typing import Optional

import numpy as np
from numpy.typing import NDArray

import fishproviz.config as config
from fishproviz.methods import distance_to_wall_chunk, mean_std
from fishproviz.utils.transformation import px2cm
from ..compute_metrics import compute_step_lengths
from ..metrics import update_filter_three_points, calculate_result_for_interval, metric_per_interval
from .compute import compute_turning_angles, compute_turning_angle_streak_lengths


def _prepare_angle_inputs(
    data: NDArray[np.float64],
    filter_index: NDArray[np.bool_],
    frame_interval: list[int],
    data_px: Optional[NDArray[np.float64]],
    area: Optional[tuple[str, NDArray[np.float64]]],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], list[int], float]:
    """Subsample, apply wall exclusion, and build the error index.

    Reads TANGLE_N_SKIP, DIST_FROM_WALL_TANGLE_IGNORED, and REMOVE_0_VECS from config.
    When skip > 0, data, filter_index, and frame_interval are all rescaled to M-space
    so that error_index[j] covers exactly the subsampled triplet producing angle[j].
    Wall NaN masking is applied to a copy of the subsampled array after error_index
    construction, preserving existing error_index semantics.

    Returns:
        subsampled: (M, 2) positions, NaN'd at wall-excluded points.
        error_index: (M-2,) boolean mask aligned to angle-space.
        frame_interval: rescaled to M-2 space.
        fill_value: 0.0 or np.nan per REMOVE_0_VECS.
    """
    skip = config.TANGLE_N_SKIP
    fill_value = np.nan if config.REMOVE_0_VECS else 0.0

    if skip > 0:
        data_eff = data[:: skip + 1]
        filter_eff = filter_index[:: skip + 1]
        frame_interval = [i // (skip + 1) for i in frame_interval]
        data_px_eff = data_px[:: skip + 1] if data_px is not None else None
    else:
        data_eff, filter_eff = data, filter_index
        data_px_eff = data_px

    error_index = update_filter_three_points(compute_step_lengths(data_eff), filter_eff)

    if config.DIST_FROM_WALL_TANGLE_IGNORED > 0 and area is not None and data_px_eff is not None:
        dtw = px2cm(distance_to_wall_chunk(data_px_eff, area[1]), fish_key=area[0]).astype("double")
        subsampled = data_eff.copy()
        subsampled[dtw < config.DIST_FROM_WALL_TANGLE_IGNORED] = np.array([np.nan, np.nan])
    else:
        subsampled = data_eff

    return subsampled, error_index, frame_interval, fill_value


def turning_angle(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
    unaveraged: bool = config.UNAVERAGED,
) -> NDArray[np.float64]:
    """Mean signed turning angle per time interval.

    Returns:
        (n_intervals, 3) array of [mean, std, n], or flat angle array when unaveraged=True.
    """
    subsampled, error_index, frame_interval, fill_value = _prepare_angle_inputs(data, filter_index, frame_interval, data_px, area)
    return calculate_result_for_interval(
        compute_turning_angles(subsampled, fill_value).astype("double"),
        frame_interval,
        mean_std if not unaveraged else None,
        error_index,
        checkfornans=True,
    )


def absolute_angles(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Mean absolute turning angle per time interval.

    Returns:
        (n_intervals, 3) array of [mean, std, n].
    """
    subsampled, error_index, frame_interval, fill_value = _prepare_angle_inputs(data, filter_index, frame_interval, data_px, area)
    return calculate_result_for_interval(
        np.abs(compute_turning_angles(subsampled, fill_value)).astype("double"),
        frame_interval,
        mean_std,
        error_index,
        checkfornans=True,
    )


def turning_angle_streak_length(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Mean streak length of consecutive same-sign turning angles.

    Returns:
        [mean, std, n] when UNAVERAGED=0, or flat streak length array when UNAVERAGED=1.
    """
    trs = turning_angle(data, frame_interval, filter_index, area, data_px, unaveraged=True)
    streak_lengths = compute_turning_angle_streak_lengths(trs).astype("double")
    return streak_lengths if config.UNAVERAGED else np.concatenate([np.array(mean_std(streak_lengths)), [len(streak_lengths)]])


def turning_angle_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=turning_angle."""
    return metric_per_interval(*args, **kwargs, metric=turning_angle)


def absolute_angle_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=absolute_angles."""
    return metric_per_interval(*args, **kwargs, metric=absolute_angles)


def turning_angle_streak_length_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=turning_angle_streak_length."""
    return metric_per_interval(*args, **kwargs, metric=turning_angle_streak_length)
