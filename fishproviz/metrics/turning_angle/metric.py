from typing import Optional

import numpy as np
from numpy.typing import NDArray

import fishproviz.config as config
from fishproviz.methods import distance_to_wall_chunk, mean_std
from fishproviz.utils.transformation import px2cm
from ..compute_primitives import compute_step_lengths
from ..metrics import update_filter_three_points, calculate_result_for_interval, metric_per_interval
from .compute import compute_turning_angles, compute_turning_angle_streaks, drop_reversals, straight_turns


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
    min_step_px: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], list[int], Optional[NDArray[np.bool_]], Optional[NDArray[np.float64]]]:
    """Subsample, build the error index (flagging wall-adjacent points) and the stationary-step mask.

    When skip > 0, data and filter_index are subsampled to M-space so that error_index[j]
    covers exactly the subsampled triplet producing angle[j]. frame_interval (data-row split
    points) is mapped to angle indices, so each angle lands in the interval of its middle point.
    Points closer to the wall than wall_threshold_cm are added to the per-point filter, so
    every angle whose triplet touches one is dropped from the statistics regardless of fill_value.
    Subsampled steps shorter than min_step_px (measured on the integer tracker pixels) are marked
    stationary, so their angles receive fill_value like zero-length steps.
    Returns the subsampled data, error index, angle split points, stationary mask and subsampled data_px.
    """
    if min_step_px > 0 and data_px is None:
        raise ValueError(f"min_step_px={min_step_px} needs data_px (tracker pixel positions); pass data_px or set min_step_px=0")
    if skip > 0:
        data_eff = data[:: skip + 1]
        filter_eff = filter_index[:: skip + 1]
        data_px_eff = data_px[:: skip + 1] if data_px is not None else None
    else:
        data_eff, filter_eff = data, filter_index
        data_px_eff = data_px
    # angle j has its middle point on data row (j + 1) * (skip + 1); the first angle at or after row i is ceil(i / (skip + 1)) - 1
    angle_interval = [max(-(-i // (skip + 1)) - 1, 0) for i in frame_interval]

    if wall_threshold_cm > 0 and area is not None and data_px_eff is not None:
        dtw = px2cm(distance_to_wall_chunk(data_px_eff, area[1]), fish_key=area[0]).astype("double")
        filter_eff = filter_eff | (dtw < wall_threshold_cm)

    error_index = update_filter_three_points(compute_step_lengths(data_eff), filter_eff)
    stationary = np.linalg.norm(np.diff(data_px_eff, axis=0), axis=1) < min_step_px if min_step_px > 0 else None

    return data_eff, error_index, angle_interval, stationary, data_px_eff


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
    min_step_px: float,
    absolute: bool,
    unaveraged: bool = False,
) -> NDArray[np.float64]:
    """Shared implementation for turning_angle and absolute_angles.

    Args:
        absolute: if True, takes np.abs() of the angle array before averaging. Exact
            reversals are excluded either way (NaN), so both metrics average the same turns.
        unaveraged: if True, skips interval averaging and returns every angle.

    Returns:
        (n_intervals, 3) array of [mean, std, n] per interval, or when unaveraged
        an (M-2, 3) array of [row, angle, excluded] where row is the index into the
        input data of the middle point of the triplet the angle belongs to, and
        excluded is 1.0 where the averaged metric drops the angle: an error, spike
        or wall filter touching one of its three points, or a NaN angle (gap, exact
        reversal, or stationary step under NaN fill).
    """
    subsampled, error_index, frame_interval, stationary, _ = _prepare_angle_inputs(
        data,
        filter_index,
        frame_interval,
        data_px,
        area,
        skip=skip,
        fill_value=fill_value,
        wall_threshold_cm=wall_threshold_cm,
        min_step_px=min_step_px,
    )
    angles = compute_turning_angles(subsampled, fill_value, stationary)
    angles = drop_reversals(angles)
    if absolute:
        angles = np.abs(angles)
    if unaveraged:
        rows = (np.arange(len(angles)) + 1) * (skip + 1)
        return np.column_stack([rows, angles, error_index | np.isnan(angles)]).astype(np.float64)
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
    min_step_px: Optional[float] = None,
    unaveraged: Optional[bool] = None,
) -> NDArray[np.float64]:
    """Mean signed turning angle per time interval.

    Sign convention: for tracker pixel coordinates (origin top-left, y down) and an
    unmirrored top-down camera, positive = clockwise on the video image = right turn.

    Sign convention verified (Oct 2026): the sign was flipped in the git history on
    2023-06-16, so results from before that date have the opposite sign. The project
    scientist confirmed that BioTracker uses the top-left video corner as (0, 0) with x
    to the right and y downwards, and checked raw turning angles against the observed
    fish movement. Right turns are positive and left turns negative; the current
    convention is correct. Do not flip it.

    Exact reversals have no turn side and are excluded (NaN, not counted in n).

    Args:
        skip: frames skipped between sampled points; None reads config.TANGLE_N_SKIP.
        remove_zero_vecs: if True, undefined angles fill with NaN instead of 0.0;
            None reads config.REMOVE_0_VECS.
        wall_threshold_cm: wall-proximity exclusion threshold in cm;
            None reads config.DIST_FROM_WALL_TANGLE_IGNORED.
        min_step_px: steps shorter than this (in tracker pixels) count as stationary;
            None reads config.TANGLE_MIN_STEP_PX.
        unaveraged: if True, return every angle instead of interval averages;
            None reads config.UNAVERAGED.

    Returns:
        (n_intervals, 3) array of [mean, std, n], or when unaveraged an (M-2, 3)
        array of [row, angle, excluded] (see _angle_metric).
    """
    _skip = config.TANGLE_N_SKIP if skip is None else skip
    _rzv = bool(config.REMOVE_0_VECS) if remove_zero_vecs is None else remove_zero_vecs
    _wall = config.DIST_FROM_WALL_TANGLE_IGNORED if wall_threshold_cm is None else wall_threshold_cm
    _min_step = config.TANGLE_MIN_STEP_PX if min_step_px is None else min_step_px
    return _angle_metric(
        data,
        frame_interval,
        filter_index,
        area,
        data_px,
        skip=_skip,
        fill_value=np.nan if _rzv else 0.0,
        wall_threshold_cm=_wall,
        min_step_px=_min_step,
        absolute=False,
        unaveraged=bool(config.UNAVERAGED) if unaveraged is None else unaveraged,
    )


def absolute_angles(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
    *,
    skip: Optional[int] = None,
    wall_threshold_cm: Optional[float] = None,
    min_step_px: Optional[float] = None,
    unaveraged: Optional[bool] = None,
) -> NDArray[np.float64]:
    """Mean absolute turning angle per time interval.

    Like turning_angle but averages np.abs() of angles. Stationary angles (a step of length 0
    or under min_step_px) are always excluded, since a 0 rad fill would make |angle| partly
    measure inactivity. Exact reversals are excluded too, although their size (pi) is known,
    so that both metrics average the same turns (complete-case; see documentation/TODO.md).

    Returns:
        (n_intervals, 3) array of [mean, std, n], or when unaveraged an (M-2, 3)
        array of [row, |angle|, excluded] (see _angle_metric).
    """
    _skip = config.TANGLE_N_SKIP if skip is None else skip
    _wall = config.DIST_FROM_WALL_TANGLE_IGNORED if wall_threshold_cm is None else wall_threshold_cm
    _min_step = config.TANGLE_MIN_STEP_PX if min_step_px is None else min_step_px
    return _angle_metric(
        data,
        frame_interval,
        filter_index,
        area,
        data_px,
        skip=_skip,
        fill_value=np.nan,
        wall_threshold_cm=_wall,
        min_step_px=_min_step,
        absolute=True,
        unaveraged=bool(config.UNAVERAGED) if unaveraged is None else unaveraged,
    )


def turning_angle_streak_length(
    data: NDArray[np.float64],
    frame_interval: list[int],
    filter_index: NDArray[np.bool_],
    area: Optional[tuple[str, NDArray[np.float64]]] = None,
    data_px: Optional[NDArray[np.float64]] = None,
    *,
    skip: Optional[int] = None,
    wall_threshold_cm: Optional[float] = None,
    min_step_px: Optional[float] = None,
    straight_px: Optional[float] = None,
    unaveraged: Optional[bool] = None,
) -> NDArray[np.float64]:
    """Mean number of consecutive turns made to the same side, per time interval.

    A streak breaks at a removed angle (error, spike or wall filter on one of its three points, or a
    gap), a stationary angle (a step of 0 px or under min_step_px) and an exact reversal. A straight
    angle (outgoing step at most straight_px tracker pixels sideways of the incoming line, going
    forward) is neutral: it neither breaks the streak nor adds to its length. Otherwise positive angles
    (right turns) and negative angles (left turns) form separate streaks. Each streak is assigned to
    the interval of its first counted angle and is not split at interval boundaries.

    Args:
        data_px: tracker pixel positions matching data; required, since straightness is judged on them.
        skip: frames skipped between sampled points; None reads config.TANGLE_N_SKIP.
        wall_threshold_cm: wall-proximity exclusion threshold in cm;
            None reads config.DIST_FROM_WALL_TANGLE_IGNORED.
        min_step_px: steps shorter than this (in tracker pixels) count as stationary;
            None reads config.TANGLE_MIN_STEP_PX.
        straight_px: sideways deviation in tracker pixels up to which an angle counts as straight;
            0 means exactly straight only, negative raises. None reads config.STREAK_STRAIGHT_PX.
        unaveraged: if True, return every streak instead of interval averages;
            None reads config.UNAVERAGED.

    Returns:
        (n_intervals, 3) array of [mean, std, n] over streak lengths (left and right pooled, n =
        number of streaks, [nan, nan, 0] for an interval without streaks), or when unaveraged an
        (n_streaks, 4) array of [row_start, row_end, side, length] in time order, where the rows are
        the input data rows of the middle points of the first and last counted angles, side is +1
        (right) or -1 (left) and length is the number of counted angles.
    """
    _skip = config.TANGLE_N_SKIP if skip is None else skip
    _wall = config.DIST_FROM_WALL_TANGLE_IGNORED if wall_threshold_cm is None else wall_threshold_cm
    _min_step = config.TANGLE_MIN_STEP_PX if min_step_px is None else min_step_px
    _straight = config.STREAK_STRAIGHT_PX if straight_px is None else straight_px
    _unaveraged = bool(config.UNAVERAGED) if unaveraged is None else unaveraged
    if data_px is None:
        raise ValueError("turning_angle_streak_length needs data_px (tracker pixel positions): straight angles are judged on tracker pixels")
    if _straight < 0:
        raise ValueError(f"straight_px={_straight} must be >= 0 (sideways deviation in tracker pixels); set straight_px=0 for exactly straight only")

    subsampled, error_index, angle_interval, stationary, subsampled_px = _prepare_angle_inputs(
        data,
        filter_index,
        frame_interval,
        data_px,
        area,
        skip=_skip,
        fill_value=np.nan,
        wall_threshold_cm=_wall,
        min_step_px=_min_step,
    )
    angles = compute_turning_angles(subsampled, np.nan, stationary)
    straight = straight_turns(subsampled_px, _straight)
    streaks = compute_turning_angle_streaks(angles, error_index, straight)
    if _unaveraged:
        rows = (streaks[:, :2] + 1) * (_skip + 1)
        return np.column_stack([rows, streaks[:, 2:]]).astype(np.float64)
    lengths = streaks[:, 3].astype(np.float64)
    split_index = np.searchsorted(streaks[:, 0], angle_interval, side="left")
    return calculate_result_for_interval(lengths, split_index, mean_std, np.zeros(len(lengths), dtype=bool))


def turning_angle_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=turning_angle."""
    return metric_per_interval(*args, **kwargs, metric=turning_angle)


def absolute_angle_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=absolute_angles."""
    return metric_per_interval(*args, **kwargs, metric=absolute_angles)


def turning_angle_streak_length_per_interval(*args, **kwargs) -> dict:
    """Entry point: delegates to metric_per_interval with metric=turning_angle_streak_length."""
    return metric_per_interval(*args, **kwargs, metric=turning_angle_streak_length)
