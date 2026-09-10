import numpy as np
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
