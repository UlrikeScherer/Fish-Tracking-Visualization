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
        Array of shape (M-2,). Entry j covers triplet (points[j], points[j+1], points[j+2]).
    """
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
    cross_products = np.cross(in_vecs, out_vecs)
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


def compute_turning_angle_streak_lengths(
    turning_angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract run lengths of consecutive same-sign turning angles.

    NaN values break streaks. Returns negative and positive run lengths concatenated.
    """
    n = len(turning_angles)
    if n == 0:
        return np.array([], dtype=np.float64)
    # NaN → 0 (separator), >0 → 1, <=0 → -1; matches original loop's else-branch treatment of 0.0
    signs = np.where(np.isnan(turning_angles), 0, np.where(turning_angles > 0, 1, -1)).astype(np.int8)
    changes = np.flatnonzero(np.diff(signs, prepend=signs[0] - 1))
    run_lengths = np.diff(np.append(changes, n))
    run_values = signs[changes]
    return np.concatenate([run_lengths[run_values == -1], run_lengths[run_values == 1]]).astype(np.float64)
