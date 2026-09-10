import numpy as np
from numpy.typing import NDArray


def compute_turning_angles(
    points: NDArray[np.float64],
    fill_value: float = 0.0,
) -> NDArray[np.float64]:
    """Compute signed turning angles for a sequence of 2D positions.

    For each interior position j, returns the signed angle in (-pi, pi] between
    the incoming and outgoing displacement vectors. Positive = counterclockwise.
    Positions where either adjacent displacement is zero or non-finite receive
    fill_value — no angle is computed across a gap or stationary step.

    Args:
        points: (M, 2) array; NaN entries mark wall-excluded or missing positions.
        fill_value: written at non-computable positions; 0.0 or np.nan.

    Returns:
        Array of shape (M-2,). Entry j covers triplet (points[j], points[j+1], points[j+2]).
    """
    orientations = np.vstack([[np.nan, np.nan], np.diff(points, axis=0)])
    is_valid_vec = np.isfinite(orientations).all(axis=1) & (orientations != 0).any(axis=1)
    computable = is_valid_vec[1:-1] & is_valid_vec[2:]
    in_vecs = orientations[1:-1][computable]
    out_vecs = orientations[2:][computable]
    dot_products = np.einsum("ij,ij->i", in_vecs, out_vecs)
    angles = np.arctan2(np.cross(in_vecs, out_vecs), dot_products)
    result = np.full(len(points) - 2, fill_value, dtype=float)
    result[computable] = angles
    return result


def compute_turning_angle_streak_lengths(
    turning_angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract run lengths of consecutive same-sign turning angles.

    NaN values break streaks. Returns negative and positive run lengths concatenated.
    """
    pos_streak = 0
    neg_streak = 0
    neg_streaks: list[int] = []
    pos_streaks: list[int] = []
    for tr in turning_angles:
        if np.isnan(tr):
            if pos_streak > 0:
                pos_streaks.append(pos_streak)
                pos_streak = 0
            if neg_streak > 0:
                neg_streaks.append(neg_streak)
                neg_streak = 0
        else:
            if tr > 0:
                pos_streak += 1
                if neg_streak > 0:
                    neg_streaks.append(neg_streak)
                    neg_streak = 0
            else:
                neg_streak += 1
                if pos_streak > 0:
                    pos_streaks.append(pos_streak)
                    pos_streak = 0
    if pos_streak > 0:
        pos_streaks.append(pos_streak)
    if neg_streak > 0:
        neg_streaks.append(neg_streak)
    return np.concatenate([neg_streaks, pos_streaks])
