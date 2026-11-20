#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cython
# tag: numpy
# You can ignore the previous line.
# It's for internal testing of the cython documentation.
from libc.math cimport acos, sqrt, ceil, isfinite

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

cdef double norm(double v0, double v1):
    return sqrt(v0**2 + v1**2)

cpdef np.ndarray[double, ndim=1] calc_steps(np.ndarray[double, ndim=2] data):
    sq = (data[1:] - data[:-1])**2
    c=np.sqrt(sq[:,0] + sq[:,1])
    return c

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] tortuosity_windows(const double[:, :] run, double window_cm):
    """(n_windows, 4) rows of [start_row, end_row, C, L] of the consecutive full path windows of one finite run.

    A window starts at row s and ends at the first row e where its path length C reaches window_cm; the next starts at e.
    L is the straight start-to-end distance; the trailing remainder shorter than window_cm is dropped.
    """
    if run.shape[1] != 2:
        raise ValueError(f"run must have shape (n, 2), got (n, {run.shape[1]})")
    if not window_cm > 0:
        raise ValueError(f"window_cm must be > 0, got {window_cm}")
    cdef Py_ssize_t n = run.shape[0]
    cdef Py_ssize_t i
    for i in range(n):
        if not (isfinite(run[i, 0]) and isfinite(run[i, 1])):
            raise ValueError(f"run row {i} is not finite ({run[i, 0]}, {run[i, 1]}); split runs at excluded points first")
    cdef np.ndarray[double, ndim=2] windows = np.empty((max(n - 1, 0), 4), dtype=np.float64)
    cdef double[:, ::1] out = windows
    cdef Py_ssize_t start = 0
    cdef Py_ssize_t k = 0
    cdef double path = 0.0
    for i in range(1, n):
        path += norm(run[i, 0] - run[i - 1, 0], run[i, 1] - run[i - 1, 1])
        if path >= window_cm:
            out[k, 0] = start
            out[k, 1] = i
            out[k, 2] = path
            out[k, 3] = norm(run[i, 0] - run[start, 0], run[i, 1] - run[start, 1])
            k += 1
            start = i
            path = 0.0
    return windows[:k]

cpdef (double, double) mean_std(np.ndarray[double, ndim=1] data):
    if data.size == 0:
        return (np.nan, np.nan)
    cdef double mean, std
    mean = data.sum()/data.size
    std = sqrt(((data-mean)**2).sum()/data.size)
    return (mean, std)

#### DISTANCE TO THE WALL --------------

cdef int _validate_wall_inputs(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area) except -1:
    """Raise ValueError unless data is (n, 2) and area is a closed polygon of >= 3 finite vertices without zero-length edges."""
    if data.shape[1] != 2:
        raise ValueError(f"data must have shape (n, 2), got (n, {data.shape[1]})")
    if area.shape[1] != 2:
        raise ValueError(f"area must have shape (n_vertices, 2), got (n_vertices, {area.shape[1]})")
    cdef Py_ssize_t e = area.shape[0]
    if e < 3:
        raise ValueError(f"area needs at least 3 vertices, got {e}")
    cdef Py_ssize_t i, k
    for i in range(e):
        if not (isfinite(area[i, 0]) and isfinite(area[i, 1])):
            raise ValueError(f"area vertex {i} is not finite ({area[i, 0]}, {area[i, 1]})")
    for i in range(e):
        k = (i + 1) % e
        if area[i, 0] == area[k, 0] and area[i, 1] == area[k, 1]:
            raise ValueError(f"area has a zero-length edge: vertex {i} equals vertex {k} ({area[i, 0]}, {area[i, 1]})")
    return 0

cpdef np.ndarray[double, ndim=1] distance_to_wall_chunk(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area):
    """Unsigned minimum distance from each point to the edges of the closed polygon area."""
    _validate_wall_inputs(data, area)
    if data.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return min_distance_to_segment(data, area)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] min_distance_to_segment(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area):
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t e = area.shape[0]
    cdef np.ndarray[double, ndim=2] dists = np.zeros((e, n), dtype=np.float64)
    cdef double ax, ay, abx, aby, ab_sq, px, py, t, fx, fy
    cdef Py_ssize_t i, j
    for i in range(e):
        ax = area[i, 0]
        ay = area[i, 1]
        abx = area[(i + 1) % e, 0] - ax
        aby = area[(i + 1) % e, 1] - ay
        ab_sq = abx * abx + aby * aby
        for j in range(n):
            px = data[j, 0]
            py = data[j, 1]
            t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            fx = ax + t * abx
            fy = ay + t * aby
            dists[i, j] = norm(px - fx, py - fy)
    return np.min(dists, axis=0)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray points_inside_polygon(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area):
    """Bool per point: inside the closed polygon area (even-odd rule) or exactly on its boundary; non-finite points are False."""
    _validate_wall_inputs(data, area)
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t e = area.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] inside = np.zeros(n, dtype=bool)
    cdef double px, py, ax, ay, bx, by
    cdef bint crossed_odd
    cdef Py_ssize_t i, j
    for j in range(n):
        px = data[j, 0]
        py = data[j, 1]
        if not (isfinite(px) and isfinite(py)):
            continue
        crossed_odd = False
        for i in range(e):
            ax = area[i, 0]
            ay = area[i, 1]
            bx = area[(i + 1) % e, 0]
            by = area[(i + 1) % e, 1]
            # exact on-segment test: collinear and within the edge's bounding box
            if (bx - ax) * (py - ay) == (by - ay) * (px - ax) and min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by):
                inside[j] = True
                break
            if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
                crossed_odd = not crossed_odd
        else:
            inside[j] = crossed_odd
    return inside

cpdef np.ndarray[double, ndim=1] distance_to_object_chunk(np.ndarray[double, ndim=2] data,
                                                          np.ndarray[double, ndim=1] ellipse_center,
                                                          double ellipse_rx, double ellipse_ry):
    cdef np.ndarray[double, ndim=1] dists
    #for i in range(size):
    dists = min_distance_to_ellipse(data[:,0], data[:,1], ellipse_center[0], ellipse_center[1], ellipse_rx, ellipse_ry)
    return dists

cdef np.ndarray[double, ndim=2] min_distance_to_ellipse(
    np.ndarray[double, ndim=1] x,
    np.ndarray[double, ndim=1] y,
    double c_x,
    double c_y,
    double r_x,
    double r_y):
    cdef double a
    cdef double b
    cdef np.ndarray[double, ndim=1] min_dists
    min_dists = np.zeros_like(x)
    for i in range(len(x)):
        if (x[i] - c_x) != 0:
            a = (y[i] - c_y) / (x[i] - c_x)
            b = y[i] - a * x[i]
            roots_x = np.roots([(r_y**2) + (r_x**2) * (a**2),
                                -2*c_x*(r_y**2) + 2*a*(b-c_y)*(r_x**2),
                                (r_y**2)*(c_x**2) + (r_x**2)*((b-c_y)**2) - (r_x**2)*(r_y**2)])
            roots_y = a * roots_x + b
        else:
            roots_x = np.array([c_x, c_x])
            roots_y = np.array([r_y + c_y, -r_y + c_y])
        min_dists[i] = np.min(np.sqrt((roots_x - x[i])**2 + (roots_y - y[i])**2))

    return min_dists
