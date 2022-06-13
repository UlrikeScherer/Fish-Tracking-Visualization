import numpy as np
from src.config import THRESHOLD_AREA_PX, BACK
from methods import distance_to_wall_chunk

def get_error_indices(dataframe):
    """
   @params: dataframe
   returns a boolean pandas array with all indices to filter set to True
    """
    x = dataframe.xpx
    y = dataframe.ypx
    indexNames = ((x == -1) & (y == -1)) | ((x == 0) & (y == 0)) # except the last index for time recording
    return indexNames

def all_error_filters(data, area_tuple):
    """
    Summery of all error filters
    @params: dataframe
    returns a boolean numpy array with all indices to filter out -- set to True!
    """
    return error_default_points(data) | error_points_out_of_area(data, area_tuple)

def error_default_points(data):
    x,y = data[:,0], data[:,1]
    return  ((x == -1) & (y == -1)) | ((x == 0) & (y == 0))

def error_points_out_of_range(data, area_tuple):
    ## error out of range 
    area = area_tuple[1]
    th = THRESHOLD_AREA_PX
    xmin, xmax = min(area[:,0])-th, max(area[:,0])+th
    ymin, ymax = min(area[:,1])-th, max(area[:,1])+th
    error_out_of_range = (data[:,0] < xmin) | (data[:,0] > xmax) | (data[:,1] < ymin) | (data[:,1] > ymax)
    return error_out_of_range

def error_points_out_of_area(data, area_tuple, day=""):
    """ returns a boolean np.array, where true indecates weather the corresponding datapoint is on the wrong side of the tank"""
    key, area = area_tuple
    is_back = BACK in key # key in the shape of <<camera>>_<<position>>
    error_out_of_range = error_points_out_of_range(data, area_tuple)
    ## over the diagonal 
    AB = area[2]-area[1]
    AP = data - area[1]
    if is_back:
        error_filter = (AP[:,1]*AB[0] - AP[:,0]*AB[1] <= 0) # cross product (a1b2−a2b1) 
    else: 
        error_filter = (AP[:,1]*AB[0] - AP[:,0]*AB[1] >= 0) # cross product (a1b2−a2b1) 
    error_filter = error_filter

    err_default = error_default_points(data)
    error_non_default = error_filter & ~ err_default
    error_non_default[error_non_default] = distance_to_wall_chunk(data[error_non_default], area) > THRESHOLD_AREA_PX   # in pixels 
    error_non_default = error_non_default  | (error_out_of_range & ~ err_default)
    if np.any(error_non_default): # ef and not ed 
        print("For id: %s, %s %d dataframes out of %d where on the other side of the tank. They are beeing filtered out." % (key,day, error_non_default.sum(), data.shape[0]))
    return error_non_default             