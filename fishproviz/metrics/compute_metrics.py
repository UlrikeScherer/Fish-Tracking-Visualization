import math

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
    # Compute the differences between adjacent points
    if distance_from_wall_to_ignore > 0:
        points[distance_to_wall < distance_from_wall_to_ignore] = np.array([np.nan, np.nan])
        point_chunks = []
        current_valid_chunk = []
        current_nan_chunk = []
        in_nan_chunk = False
        for point in points:
            if np.any(np.isnan(point)):
                in_nan_chunk = True
                if len(current_valid_chunk) > 2:
                    point_chunks.append(np.array(current_valid_chunk))
                elif 3 > len(current_valid_chunk) > 0:
                    current_nan_chunk += current_valid_chunk
                current_valid_chunk = []

                current_nan_chunk.append(point)
            else:
                in_nan_chunk = False
                if len(current_nan_chunk) > 2:
                    point_chunks.append(np.array(current_nan_chunk))
                elif 3 > len(current_nan_chunk) > 0:
                    current_valid_chunk += current_nan_chunk
                current_nan_chunk = []
                current_valid_chunk.append(point)
        if in_nan_chunk:
            if len(current_nan_chunk) > 2:
                point_chunks.append(np.array(current_nan_chunk))
            elif 2 >= len(current_nan_chunk) > 0:
                point_chunks[-1] = np.concatenate([point_chunks[-1], np.array(current_nan_chunk)])
        else:
            if len(current_valid_chunk) > 2:
                point_chunks.append(np.array(current_valid_chunk))
            elif 2 >= len(current_valid_chunk) > 0:
                point_chunks[-1] = np.concatenate([point_chunks[-1], np.array(current_valid_chunk)])

    else:
        point_chunks = [points]
    turning_angles_final = np.array([])
    for i, point_chunk in enumerate(point_chunks):
        if skip < math.ceil((len(point_chunk) - 3) / 3):
            vectors = np.diff(point_chunk[:: skip + 1], axis=0)

            # Find the indices where the difference vector is non-zero and finite
            wanted_indices = np.any(vectors != 0, axis=1) & np.all(np.isfinite(vectors), axis=1)
            vectors = vectors[wanted_indices]
            # Compute the dot products and determinants between pairs of vectors
            dot_products = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
            determinants = np.cross(vectors[:-1], vectors[1:])

            # Compute the turning angles
            turning_angles = np.arctan2(determinants, dot_products)
            if skip == 0:
                turning_angles_result = np.full(point_chunk.shape[0] - 2, np.nan if remove_zero_vectors else 0, dtype=float)
                # the last one is buried in the angle if not False anyways
                wanted_angles = np.where(wanted_indices)[0][1:] - 1
                # Set the turning angles to 0 for equal consecutive points
                turning_angles_result[wanted_angles] = turning_angles
            else:  # in case of skip > 0, make sure to modify array to maintain same length as skip = 0 but with nan values placed accordingly (for compatibility with further processing)
                # Creating a new array 'new_nums' of length len(nums) + (len(nums) - 1) * p filled with zeros
                zero_arr = np.full(len(np.diff(point_chunk[:: skip + 1], axis=0)) - 1, np.nan if remove_zero_vectors else 0, dtype=float)
                zero_arr[np.where(wanted_indices)[0][1:] - 1] = turning_angles
                turning_angles = zero_arr  # take into consideration zero vectors that were discarded by wanted_indeces and put these 0 instead of NaN
                turning_angles_result = np.full(len(turning_angles) + (len(turning_angles) - 1) * (skip), np.nan)

                # Filling the 'new_nums' array with elements from 'nums' at intervals of (p + 1)
                turning_angles_result[:: skip + 1] = turning_angles
                turning_angles_result = np.concatenate([np.full(skip, np.nan), turning_angles_result, np.full(skip, np.nan)])
                turning_angles_result = np.concatenate([turning_angles_result, np.full(point_chunk.shape[0] - 2 - len(turning_angles_result), np.nan)])

        else:
            turning_angles_result = np.full(len(point_chunk) - 2, np.nan)
        turning_angles_final = np.concatenate([turning_angles_final, np.concatenate([np.array([np.nan, np.nan]), turning_angles_result])])

    return turning_angles_final[2:]


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
