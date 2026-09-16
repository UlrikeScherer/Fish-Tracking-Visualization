from typing import Optional

import numpy as np
from numpy.typing import NDArray

import fishproviz.config as config
from fishproviz.methods import distance_to_wall_chunk, mean_std
from fishproviz.utils.transformation import px2cm
from ..compute_primitives import compute_step_lengths
from ..metrics import update_filter_three_points, calculate_result_for_interval, metric_per_interval
from .compute import compute_turning_angles, compute_turning_angle_streak_lengths


def _prepare_angle_inputs(
    data: NDArray[np.float64],
    filter_index: NDArray[np.bool_],
    frame_interval: list[int],
    data_px: Optional[NDArray[np.float64]],
    area: Optional[tuple[str, NDArray[np.float64]]],
    *,
    skip: int,
    fill_value: float,
    wall_threshold_cm: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], list[int]]:
    """Subsample, apply wall exclusion, and build the error index.

    When skip > 0, data, filter_index, and frame_interval are rescaled to M-space
    so that error_index[j] covers exactly the subsampled triplet producing angle[j].
    Wall NaN masking is applied after error_index construction.
    """
    if skip > 0:
        data_eff = data[:: skip + 1]
        filter_eff = filter_index[:: skip + 1]
        frame_interval = [i // (skip + 1) for i in frame_interval]
        data_px_eff = data_px[:: skip + 1] if data_px is not None else None
    else:
        data_eff, filter_eff = data, filter_index
        data_px_eff = data_px

    error_index = update_filter_three_points(compute_step_lengths(data_eff), filter_eff)

    if wall_threshold_cm > 0 and area is not None and data_px_eff is not None:
        dtw = px2cm(distance_to_wall_chunk(data_px_eff, area[1]), fish_key=area[0]).astype("double")
        subsampled = data_eff.copy()
        subsampled[dtw < wall_threshold_cm] = np.array([np.nan, np.nan])
    else:
        subsampled = data_eff

    return subsampled, error_index, frame_interval


def _angle_metric(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]],
    data_px: Optional[NDArray[np.float64]],
    *,
    skip: int,
    fill_value: float,
    wall_threshold_cm: float,
    absolute: bool,
) -> NDArray[np.float64]:
    """Shared implementation for turning_angle and absolute_angles.

    Args:
        absolute: if True, takes np.abs() of the angle array before averaging.

    Returns:
        (n_intervals, 3) array of [mean, std, n] per interval.
    """
    subsampled, error_index, frame_interval = _prepare_angle_inputs(
        data,
        filter_index,
        frame_interval,
        data_px,
        area,
        skip=skip,
        fill_value=fill_value,
        wall_threshold_cm=wall_threshold_cm,
    )
    angles = compute_turning_angles(subsampled, fill_value)
    if absolute:
        angles = np.abs(angles)
    return calculate_result_for_interval(angles, frame_interval, mean_std, error_index, checkfornans=True)


def turning_angle(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
    *,
    skip: Optional[int] = None,
    remove_zero_vecs: Optional[bool] = None,
    wall_threshold_cm: Optional[float] = None,
) -> NDArray[np.float64]:
    """Mean signed turning angle per time interval.

    Args:
        skip: frames skipped between sampled points; None reads config.TANGLE_N_SKIP.
        remove_zero_vecs: if True, undefined angles fill with NaN instead of 0.0;
            None reads config.REMOVE_0_VECS.
        wall_threshold_cm: wall-proximity exclusion threshold in cm;
            None reads config.DIST_FROM_WALL_TANGLE_IGNORED.

    Returns:
        (n_intervals, 3) array of [mean, std, n].
    """
    _skip = config.TANGLE_N_SKIP if skip is None else skip
    _rzv = bool(config.REMOVE_0_VECS) if remove_zero_vecs is None else remove_zero_vecs
    _wall = config.DIST_FROM_WALL_TANGLE_IGNORED if wall_threshold_cm is None else wall_threshold_cm
    return _angle_metric(
        data,
        frame_interval,
        filter_index,
        area,
        data_px,
        skip=_skip,
        fill_value=np.nan if _rzv else 0.0,
        wall_threshold_cm=_wall,
        absolute=False,
    )


def absolute_angles(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
    *,
    skip: Optional[int] = None,
    remove_zero_vecs: Optional[bool] = None,
    wall_threshold_cm: Optional[float] = None,
) -> NDArray[np.float64]:
    """Mean absolute turning angle per time interval.

    Identical to turning_angle but takes np.abs() of angles before averaging.

    Returns:
        (n_intervals, 3) array of [mean, std, n].
    """
    _skip = config.TANGLE_N_SKIP if skip is None else skip
    _rzv = bool(config.REMOVE_0_VECS) if remove_zero_vecs is None else remove_zero_vecs
    _wall = config.DIST_FROM_WALL_TANGLE_IGNORED if wall_threshold_cm is None else wall_threshold_cm
    return _angle_metric(
        data,
        frame_interval,
        filter_index,
        area,
        data_px,
        skip=_skip,
        fill_value=np.nan if _rzv else 0.0,
        wall_threshold_cm=_wall,
        absolute=True,
    )


def turning_angle_streak_length(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
    *,
    skip: Optional[int] = None,
    remove_zero_vecs: Optional[bool] = None,
    wall_threshold_cm: Optional[float] = None,
    unaveraged: Optional[bool] = None,
) -> NDArray[np.float64]:
    """Mean run length of consecutive same-sign turning angles.

    Args:
        unaveraged: if True, returns raw streak length array; if False, returns
            [mean, std, n]. None reads config.UNAVERAGED.

    Returns:
        [mean, std, n] when unaveraged=False, or flat streak length array when True.
    """
    _skip = config.TANGLE_N_SKIP if skip is None else skip
    _rzv = bool(config.REMOVE_0_VECS) if remove_zero_vecs is None else remove_zero_vecs
    _wall = config.DIST_FROM_WALL_TANGLE_IGNORED if wall_threshold_cm is None else wall_threshold_cm
    _unaveraged = bool(config.UNAVERAGED) if unaveraged is None else unaveraged
    fill_value = np.nan if _rzv else 0.0

    subsampled, error_index, _ = _prepare_angle_inputs(
        data,
        filter_index,
        frame_interval,
        data_px,
        area,
        skip=_skip,
        fill_value=fill_value,
        wall_threshold_cm=_wall,
    )
    angles = compute_turning_angles(subsampled, fill_value)
    valid = ~error_index & ~np.isnan(angles)
    streak_lengths = compute_turning_angle_streak_lengths(angles[valid])
    return streak_lengths if _unaveraged else np.concatenate([np.array(mean_std(streak_lengths)), [len(streak_lengths)]])


def turning_angle_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=turning_angle."""
    return metric_per_interval(*args, **kwargs, metric=turning_angle)


def absolute_angle_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=absolute_angles."""
    return metric_per_interval(*args, **kwargs, metric=absolute_angles)


def turning_angle_streak_length_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=turning_angle_streak_length."""
    return metric_per_interval(*args, **kwargs, metric=turning_angle_streak_length, is_summary=True)
