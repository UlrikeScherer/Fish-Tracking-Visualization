from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Relative |sin| below which an anti-parallel step pair counts as an exact reversal. For integer pixel
# positions a non-reversal has |sin| >= 1/(n1*n2) (step lengths in px), about 4e-7 even for 1500 px steps.
REVERSAL_SIN_TOL: float = 1e-9


def compute_turning_angles(
    points: NDArray[np.float64],
    fill_value: float = 0.0,
    stationary: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float64]:
    """Compute signed turning angles for a sequence of 2D positions.

    For each interior position j, returns the signed angle in (-pi, pi] between
    the incoming and outgoing displacement vectors. Positive = counterclockwise in
    a y-up frame, which is clockwise (a right turn) on tracker pixel coordinates (y down).
    An angle with a non-finite adjacent displacement (a NaN or inf position, i.e. a gap) is always
    NaN. An angle with finite displacements where either has zero length (the fish did not move)
    receives fill_value, as does one adjacent to a step marked in stationary.
    An exact reversal (outgoing step anti-parallel to the incoming one) returns +pi, since its
    turn side is undefined; callers needing a signed value map +pi to their own convention.
    The tolerance relies on integer pixel positions: any non-reversal has |cross| >= 1 px^2, so
    |sin| >= 1/(n1*n2), far above REVERSAL_SIN_TOL for any realistic step length.

    Args:
        points: (M, 2) array; NaN entries mark missing positions.
        fill_value: written where an adjacent displacement has zero length or is stationary; 0.0 or np.nan.
        stationary: optional (M-1,) mask of steps to treat as zero length (e.g. below a pixel cut-off).

    Returns:
        Array of shape (M-2,), empty for M < 3. Entry j covers triplet (points[j], points[j+1], points[j+2]).
    """
    if len(points) < 3:
        return np.zeros(0, dtype=float)
    diffs = np.diff(points, axis=0)
    norms = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    finite = np.isfinite(norms)
    moving = finite & (norms > 0)
    if stationary is not None:
        moving &= ~stationary
    computable = moving[:-1] & moving[1:]
    gap = ~(finite[:-1] & finite[1:])
    in_vecs = diffs[:-1][computable]
    out_vecs = diffs[1:][computable]
    dot_products = np.einsum("ij,ij->i", in_vecs, out_vecs)
    cross_products = in_vecs[:, 0] * out_vecs[:, 1] - in_vecs[:, 1] * out_vecs[:, 0]
    angles = np.arctan2(cross_products, dot_products)
    reversal = (np.abs(cross_products) <= REVERSAL_SIN_TOL * norms[:-1][computable] * norms[1:][computable]) & (dot_products < 0)
    angles[reversal] = np.pi
    result = np.full(len(points) - 2, fill_value, dtype=float)
    result[computable] = angles
    result[gap] = np.nan
    return result


def score_reversals_as_zero(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed callers score the canonical +pi of an exact reversal as 0.0."""
    return np.where(angles == np.pi, 0.0, angles)


def straight_turns(points: NDArray[np.float64], max_lateral: float) -> NDArray[np.bool_]:
    """Mark angles whose outgoing step deviates sideways by at most max_lateral from the incoming direction.

    Entry j is True where |cross| / |incoming step| <= max_lateral and dot > 0, i.e. the outgoing
    step leaves at most max_lateral (in the units of points) to either side of the incoming line
    and keeps going forward. Angles with a non-finite or zero-length adjacent step are False.

    Returns:
        Boolean array of shape (M-2,), empty for M < 3.
    """
    if len(points) < 3:
        return np.zeros(0, dtype=bool)
    diffs = np.diff(points, axis=0)
    norms = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    moving = np.isfinite(norms) & (norms > 0)
    computable = moving[:-1] & moving[1:]
    in_vecs = diffs[:-1][computable]
    out_vecs = diffs[1:][computable]
    cross_products = in_vecs[:, 0] * out_vecs[:, 1] - in_vecs[:, 1] * out_vecs[:, 0]
    dot_products = np.einsum("ij,ij->i", in_vecs, out_vecs)
    result = np.zeros(len(points) - 2, dtype=bool)
    result[computable] = (np.abs(cross_products) <= max_lateral * norms[:-1][computable]) & (dot_products > 0)
    return result


def compute_turning_angle_streaks(
    angles: NDArray[np.float64],
    breaks: NDArray[np.bool_],
    straight: NDArray[np.bool_],
) -> NDArray[np.int64]:
    """Find runs of consecutive turns to the same side.

    Per angle, in order of precedence: a break (breaks, NaN, or the exact reversal +pi) ends the
    current streak; a neutral angle (straight, or exactly 0.0) neither ends a streak nor adds to it;
    otherwise the angle counts towards side +1 (> 0, right turn on tracker pixels) or -1 (< 0, left).
    An angle of exactly 0.0 has no side, so it is neutral even when not marked straight.

    Returns:
        (n_streaks, 4) int64 array of [first_angle_idx, last_angle_idx, side, length] in time order,
        where the indices are of the first and last counted angles and length counts counted angles.
    """
    is_break = breaks | np.isnan(angles) | (angles == np.pi)
    side = np.where(is_break | straight, 0, np.sign(np.nan_to_num(angles))).astype(np.int64)
    counted = np.flatnonzero(side)
    if len(counted) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    counted_side = side[counted]
    break_count = np.cumsum(is_break)[counted]
    starts_new = np.ones(len(counted), dtype=bool)
    starts_new[1:] = (counted_side[1:] != counted_side[:-1]) | (break_count[1:] != break_count[:-1])
    starts = np.flatnonzero(starts_new)
    ends = np.append(starts[1:], len(counted)) - 1
    return np.column_stack([counted[starts], counted[ends], counted_side[starts], ends - starts + 1]).astype(np.int64)
