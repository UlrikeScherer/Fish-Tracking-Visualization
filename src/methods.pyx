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

ctypedef np.ndarray[float, ndim=1] (*f_type)(np.ndarray[double, ndim=2])

cdef double arccos(double v):
    return acos(v)

cdef double clip(double a, int min_value, int max_value):
    return min(max(a, min_value), max_value)

cdef double dot(double v0, double v1, double w0, double w1):
    return v0*w0+v1*w1

cdef double norm(double v0, double v1):
    return sqrt(v0**2 + v1**2)

cdef (double, double) unit_vector(v0, v1):
    """ Returns the unit vector of the vector.  """
    cdef double n
    n = norm(v0, v1)
    if n == 0:
        return (v0, v1)
    return (v0/n, v1/n)
    
cdef double determinant(double v0, double v1, double w0, double w1):
    """ Determinant of two vectors. """
    return v0*w1-v1*w0

cdef double direction_angle(double v0, double v1, double w0, double w1):
    """ Returns the angle between v,w anti clockwise from direction v to m. """
    cdef double cos, r, det
    cos = dot(v0,v1,w0,w1)
    r = arccos(clip(cos, -1, 1))
    det = determinant(v0,v1,w0,w1)
    if det < 0: 
        return r
    else:
        return -r
    
cdef double angle(double v0, double v1, double w0, double w1):
    cos = dot(v0,v1,w0,w1)
    return arccos(clip(cos, -1, 1))

cpdef (double, double) avg_and_sum_angles(np.ndarray[double, ndim=2] data):
    cdef np.ndarray[double, ndim=2] vecs
    cdef double v0, v1, u0, u1
    vecs = data[1:]-data[:-1]
    vecs = vecs[np.all(vecs!=0, axis=1)]
    cdef double sum_avg, sum_ang
    cdef int N_alpha
    sum_avg, sum_ang, N_alpha = 0.,0.,(len(vecs)-1)

    if N_alpha <= 0: 
        return (sum_avg, sum_ang)
    
    (u0, u1) = unit_vector(vecs[0,0], vecs[0,1])
    cdef int i
    for i in range(1,N_alpha):
        (v0, v1) = unit_vector(vecs[i,0], vecs[i,1])
        sum_avg += angle(u0,u1,v0,v1)
        sum_ang += direction_angle(u0,u1,v0,v1)
        u0, u1 = v0, v1
    return (sum_avg/N_alpha, sum_ang)

cpdef np.ndarray[double, ndim=1] calc_steps(np.ndarray[double, ndim=2] data):
    sq = (data[1:] - data[:-1])**2
    c=np.sqrt(sq[:,0] + sq[:,1])
    return c

cpdef np.ndarray[double, ndim=1] tortuosity_of_chunk(np.ndarray[double, ndim=2] data):
    cdef int dist_length = 10 # normed by 10cm of distance traveled
    cdef np.ndarray[double, ndim=1] steps = calc_steps(data)
    cdef np.ndarray[double, ndim=1] c_steps = np.cumsum(steps)  # cumulative sum
    cdef int length = int(c_steps[-1]/dist_length)
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

cpdef np.ndarray[float, ndim=1] avg_turning_direction(np.ndarray[double, ndim=2] data):
    cdef np.ndarray[double, ndim=2] vecs
    cdef double v0, v1, u0, u1
    vecs_in = data[1:]-data[:-1]
    indices = np.all(vecs_in!=0, axis=1)
    vecs = vecs_in[indices]
    cdef int N_alpha
    N_alpha = len(vecs)
    cdef np.ndarray sum_ang = np.zeros([N_alpha], dtype=float)
    if N_alpha == 0: 
        return sum_ang
    (u0, u1) = unit_vector(vecs[0,0], vecs[0,1])
    cdef int i
    for i in range(1,N_alpha):
        (v0, v1) = unit_vector(vecs[i,0], vecs[i,1])
        sum_ang[i-1] = direction_angle(u0,u1,v0,v1)
        u0, u1 = v0, v1

    results = np.zeros(len(vecs_in), dtype=float)
    results[indices] = sum_ang
    return results


cpdef np.ndarray[double, ndim=2] activity(np.ndarray[double, ndim=2] data, int frame_interval):
    cdef int len_out, i, s
    cdef np.ndarray[double, ndim=2] chunk
    cdef np.ndarray[double, ndim=1] steps
    cdef double SIZE = data.shape[0]
    len_out = int(ceil(SIZE/frame_interval))
    cdef np.ndarray mu_sd = np.zeros([len_out,2], dtype=float)
    for i,s in enumerate(range(0, data.shape[0], frame_interval)):
        chunk = data[s:s+frame_interval]
        chunk = chunk[chunk[:,0] > -1]
        steps = calc_steps(chunk)
        mu_sd[i, 0] = sum(steps)/frame_interval
        mu_sd[i, 1] = sqrt(sum((steps-mu_sd[i, 0])**2)/frame_interval)
    return mu_sd

cpdef np.ndarray[double, ndim=2] turning_angle(np.ndarray[double, ndim=2] data, int frame_interval):
    cdef int len_out, i, s
    cdef np.ndarray[double, ndim=2] chunk
    cdef np.ndarray[double, ndim=1] avg_turning
    cdef double SIZE = data.shape[0]
    len_out = int(ceil(SIZE/frame_interval))
    cdef np.ndarray mu_sd = np.zeros([len_out,2], dtype=float)
    for i,s in enumerate(range(0, data.shape[0], frame_interval)):
        chunk = data[s:s+frame_interval+1]
        chunk = chunk[chunk[:,0] > -1]
        avg_turning = avg_turning_direction(chunk)
        mu_sd[i, 0] = sum(avg_turning)/frame_interval
        mu_sd[i, 1] = sqrt(sum((avg_turning-mu_sd[i, 0])**2)/frame_interval)
    return mu_sd
