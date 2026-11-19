import numpy as np
import scipy.stats as scipy_stats
from numpy.typing import NDArray
import fishproviz.config as config


def compute_step_lengths(points: np.ndarray) -> np.ndarray:
    vectors = np.diff(points, axis=0)
    return np.sqrt(np.einsum("ij,ij->i", vectors, vectors))


def calc_step_per_frame(batchxy, frames):
    """This function calculates the eucleadian step length in centimeters per FRAME, this is useful as a speed measurement after the removal of erroneous data points."""
    frame_dist = frames[1:] - frames[:-1]
    c = compute_step_lengths(batchxy) / frame_dist
    return c


def entropy_heatmap(points: NDArray[np.float64], area: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
    """ENTROPY_BINS x ENTROPY_BINS histogram over the area polygon's bounding box +- ENTROPY_MARGIN_PX, and the number of points outside it."""
    margin = config.ENTROPY_MARGIN_PX
    (xmin, ymin), (xmax, ymax) = area.min(axis=0) - margin, area.max(axis=0) + margin
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"Entropy grid range must have positive width and height, got x [{xmin}, {xmax}], y [{ymin}, {ymax}] from the area polygon and ENTROPY_MARGIN_PX={margin}")
    # inclusive on both ends, like np.histogram2d's range (its last bin is closed); NaN counts as out of range
    x, y = points[:, 0], points[:, 1]
    in_range = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    hist = np.histogram2d(x[in_range], y[in_range], bins=config.ENTROPY_BINS, range=[[xmin, xmax], [ymin, ymax]])[0]
    return hist, int(np.count_nonzero(~in_range))


def entropy_for_chunk(points: NDArray[np.float64], area: NDArray[np.float64]) -> tuple[float, int, int]:
    """(Shannon entropy in nats of the occupancy histogram, points histogrammed, points out of range); entropy NaN when none is histogrammed."""
    if len(points) == 0:
        return np.nan, 0, 0
    hist, n_out_of_range = entropy_heatmap(points, area)
    n_histogrammed = int(hist.sum())
    if n_histogrammed == 0:
        return np.nan, 0, n_out_of_range
    return float(scipy_stats.entropy(hist.ravel())), n_histogrammed, n_out_of_range
