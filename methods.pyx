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

cdef double dot(np.ndarray[double, ndim=1] v, np.ndarray[double, ndim=1] w):
    return v[0]*w[0]+v[1]*w[1]

cdef float norm(np.ndarray[double, ndim=1] vec):
    return sqrt(vec[0]**2 + vec[1]**2)

cdef np.ndarray[double, ndim=1] unit_vector(np.ndarray[double, ndim=1] vector):
    """ Returns the unit vector of the vector.  """
    cdef float n = norm(vector)
    if n == 0:
        return vector
    return vector / n
    
cdef double determinant(np.ndarray[double, ndim=1] v, np.ndarray[double, ndim=1] w):
    """ Determinant of two vectors. """
    return v[0]*w[1]-v[1]*w[0]

cdef double direction_angle(np.ndarray[double, ndim=1] v,np.ndarray[double, ndim=1] w):
    """ Return the angle between v,w anti clockwise from direction v to m. """
    cdef double cos, r, det
    cos = dot(v,w)
    r = arccos(clip(cos, -1, 1))
    det = determinant(v,w)
    if det < 0: 
        return r
    else:
        return -r
    
cdef double angle(np.ndarray[double, ndim=1] v, np.ndarray[double, ndim=1] w):
    cos = dot(v,w)
    return arccos(clip(cos, -1, 1))

cpdef tuple avg_and_sum_angles(df):
    cdef np.ndarray[double, ndim=2] npdf, vecs
    cdef np.ndarray[double, ndim=1] u, v
    npdf = df[["xpx", "ypx"]].to_numpy()
    vecs = npdf[1:]-npdf[:-1]
    vecs = vecs[np.all(vecs!=0, axis=1)]
    cdef double sum_avg, sum_ang
    cdef int N_alpha
    sum_avg, sum_ang, N_alpha = 0.,0.,(len(vecs)-1)

    if N_alpha <= 0: 
        return sum_avg, sum_ang
    
    u = unit_vector(vecs[0])
    cdef int i
    for i in range(1,N_alpha):
        v = unit_vector(vecs[i])
        sum_avg += angle(u,v)
        sum_ang += direction_angle(u,v)
        u = v
    return (sum_avg/N_alpha, sum_ang)