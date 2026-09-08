import numpy as np
from numpy._typing import NDArray
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt
import fishproviz.config as config


def compute_step_lengths(points: np.ndarray) -> np.ndarray:
    vectors = np.diff(points, axis=0)
    distances = np.linalg.norm(vectors, axis=1)
    return distances


def calc_step_per_frame(batchxy, frames):
    """This function calculates the eucleadian step length in centimeters per FRAME, this is useful as a speed measurement after the removal of erroneous data points."""
    frame_dist = frames[1:] - frames[:-1]
    c = compute_step_lengths(batchxy) / frame_dist
    return c


def compute_turning_angles(
    points: np.ndarray,
    skip: int = config.TANGLE_N_SKIP,
    remove_zero_vectors: bool = config.REMOVE_0_VECS,
    distance_from_wall_to_ignore: float = config.DIST_FROM_WALL_TANGLE_IGNORED,
    distance_to_wall: NDArray[float] = None,
) -> np.ndarray:
    if distance_from_wall_to_ignore > 0 and distance_to_wall is not None:
        points[distance_to_wall < distance_from_wall_to_ignore] = np.array([np.nan, np.nan])

    subsampled = points[:: skip + 1]
    orientations = np.vstack([[np.nan, np.nan], np.diff(subsampled, axis=0)])
    is_valid_vec = np.isfinite(orientations).all(axis=1) & (orientations != 0).any(axis=1)

    # result[j] = angle at triplet (sub[j], sub[j+1], sub[j+2]),
    # requiring orientations[j+1] and orientations[j+2] both valid
    computable = is_valid_vec[1:-1] & is_valid_vec[2:]

    in_vecs = orientations[1:-1][computable]
    out_vecs = orientations[2:][computable]
    dot_products = np.einsum("ij,ij->i", in_vecs, out_vecs)
    turning_angles = np.arctan2(np.cross(in_vecs, out_vecs), dot_products)

    result = np.full(len(subsampled) - 2, np.nan if remove_zero_vectors else 0, dtype=float)
    result[computable] = turning_angles
    return result


def compute_turning_angle_streak_lengths(turning_angles):
    pos_streak = 0
    neg_streak = 0
    neg_streaks = []
    pos_streaks = []
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


def entropy_heatmap(chunk, area, bins=(18, 18)):
    """Calculate the 2D histogram of the chunk"""
    th = config.THRESHOLD_AREA_PX
    xmin, xmax = min(area[:, 0]) - th, max(area[:, 0]) + th
    ymin, ymax = min(area[:, 1]) - th, max(area[:, 1]) + th

    return np.histogram2d(
        chunk[:, 0],
        chunk[:, 1],
        bins=bins,
        density=False,
        range=[[xmin, xmax], [ymin, ymax]],
    )[0]


def entropy_for_chunk(chunk, area_tuple):
    """
    Args: chunk,
    area = tuple(fish_key, data)
    retrun entropy
    """
    if chunk.shape[0] == 0:
        return np.nan
    fish_key, area = area_tuple

    hist = entropy_heatmap(chunk, area)
    l_x, l_y = hist.shape
    if config.BACK in fish_key:  # if back use take the upper triangle -3
        tri = np.triu_indices(l_y, k=-3)
    else:  # if front the lower triangle +3
        tri = np.tril_indices(l_y, k=3)
    sum_hist = np.sum(hist)
    if sum_hist == 0:  #
        # print(chunk[:10])
        print("Warning for %s all %d data points were not in der range of histogram and removed" % (fish_key, chunk.shape[0]))
        return np.nan
    if chunk.shape[0] > sum_hist:
        # print(chunk[:10])
        print("Warning for %s %d out of %d data points were not in der range of histogram and removed" % (fish_key, chunk.shape[0] - sum_hist, chunk.shape[0]))
    if sum_hist > np.sum(hist[tri]):
        print(
            "Warning for %s the selected area for entropy has lost some points: " % fish_key,
            "sum hist: ",
            np.sum(hist),
            "sum selection: ",
            sum(hist[tri]),
            "\n",
            fish_key,
        )
        print("entropy: ", scipy_stats.entropy(hist[tri]))
        plt.plot(*area.T)
        plt.plot(*chunk.T, "*")
    return scipy_stats.entropy(hist[tri])
