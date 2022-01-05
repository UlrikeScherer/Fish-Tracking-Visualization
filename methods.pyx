import cython
# tag: numpy
# You can ignore the previous line.
# It's for internal testing of the cython documentation.
from libc.math cimport acos, sqrt

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

cpdef (double, double) avg_and_sum_angles(df):
    cdef np.ndarray[double, ndim=2] npdf, vecs
    cdef double v0, v1, u0, u1
    npdf = df[["xpx", "ypx"]].to_numpy()
    vecs = npdf[1:]-npdf[:-1]
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