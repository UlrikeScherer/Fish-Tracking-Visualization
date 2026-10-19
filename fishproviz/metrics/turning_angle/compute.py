import numpy as np
from numpy.typing import NDArray


def compute_turning_angles(
    points: NDArray[np.float64],
    fill_value: float = 0.0,
) -> NDArray[np.float64]:
    """Compute signed turning angles for a sequence of 2D positions.

    For each interior position j, returns the signed angle in (-pi, pi] between
    the incoming and outgoing displacement vectors. Positive = counterclockwise in
    a y-up frame, which is clockwise (a right turn) on tracker pixel coordinates (y down).
    An angle with a non-finite adjacent displacement (a NaN or inf position, i.e. a gap) is always
    NaN. An angle with finite displacements where either has zero length (the fish did not move)
    receives fill_value.

    Args:
        points: (M, 2) array; NaN entries mark missing positions.
        fill_value: written where an adjacent displacement has zero length; 0.0 or np.nan.

    Returns:
        Array of shape (M-2,). Entry j covers triplet (points[j], points[j+1], points[j+2]).
    """
    diffs = np.diff(points, axis=0)
    norms = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    finite = np.isfinite(norms)
    moving = finite & (norms > 0)
    computable = moving[:-1] & moving[1:]
    gap = ~(finite[:-1] & finite[1:])
    in_vecs = diffs[:-1][computable]
    out_vecs = diffs[1:][computable]
    dot_products = np.einsum("ij,ij->i", in_vecs, out_vecs)
    angles = np.arctan2(np.cross(in_vecs, out_vecs), dot_products)
    result = np.full(len(points) - 2, fill_value, dtype=float)
    result[computable] = angles
    result[gap] = np.nan
    return result


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
