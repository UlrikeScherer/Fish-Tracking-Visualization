#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cython
# tag: numpy
# You can ignore the previous line.
# It's for internal testing of the cython documentation.
from libc.math cimport acos, sqrt, ceil

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

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.float64_t double
cdef int NDIM = 3

cdef double norm(double v0, double v1):
    return sqrt(v0**2 + v1**2)

cpdef np.ndarray[double, ndim=1] calc_steps(np.ndarray[double, ndim=2] data):
    sq = (data[1:] - data[:-1])**2
    c=np.sqrt(sq[:,0] + sq[:,1])
    return c

cpdef np.ndarray[double, ndim=1] tortuosity_of_chunk(np.ndarray[double, ndim=2] data):
    cdef int dist_length = 10 # normed by 10cm of distance traveled
    cdef np.ndarray[double, ndim=1] steps = calc_steps(data)
    cdef np.ndarray[double, ndim=1] c_steps = np.cumsum(steps)  # cumulative sum
    cdef list t_result = []
    cdef double L, C, curr_c
    cdef int i = 0
    cdef int j = 0
    cdef double min_L = 0.1
    curr_c = 0 # start with 0
    while i < c_steps.size-2:
        while j < c_steps.size-1 and c_steps[j]-curr_c < dist_length:
            j+=1
        L = np.sqrt(sum((data[j+1]-data[i])**2))
        C = c_steps[j] - curr_c
        if L < min_L: L=min_L
        if C < min_L: C=min_L
        if L > C: L=C
        t_result.append(C/L)
        curr_c = c_steps[j]
        i=j+1
        j=i

    return np.array(t_result, dtype=float)

cpdef (double, double) mean_std(np.ndarray[double, ndim=1] data):
    if data.size == 0:
        return (np.nan, np.nan)
    cdef double mean, std
    mean = data.sum()/data.size
    std = sqrt(((data-mean)**2).sum()/data.size)
    return (mean, std)

#### DINSTANCE TO THE WALL --------------

cpdef np.ndarray[double, ndim=1] distance_to_wall_chunk(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area):
    return min_distance_to_segment(data, area)

cdef np.ndarray[double, ndim=1] min_distance_to_segment(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] area):
    cdef int n = data.shape[0]
    cdef int e = area.shape[0]
    cdef np.ndarray[double, ndim=2] dists = np.zeros((e, n), dtype=np.float64)
    cdef double ax, ay, abx, aby, ab_sq, px, py, t, fx, fy
    cdef int i, j
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
