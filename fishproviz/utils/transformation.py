import numpy as np
from .tank_area_config import get_calibration_functions
CONST_PX2CM = 0.02275
FUNCS_PX2CM = get_calibration_functions()

def normalize_origin_of_compartment(data, area, is_back):
    if is_back:
        origin1 = area[0,0], area[1,1]
        new_area = (area-origin1)
        origin2 = new_area[2,0],new_area[3,1]
        new_area = -new_area+origin2
        data = -data+origin1+origin2
    else:
        origin1 = area[1,0], area[0,1]
        new_area = area - origin1
        data = data - origin1
    return data, new_area

def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def px2cm(a, camera=None, day=None):
    if camera: return a * FUNCS_PX2CM(camera, day)
    return a * CONST_PX2CM


def pixel_to_cm(pixels, camera=None, day=None):
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    R = rotation(np.pi / 4)
    if camera and day:
        t = [FUNCS_PX2CM(camera, day), FUNCS_PX2CM(camera, day)]
    else:
        t = [CONST_PX2CM, CONST_PX2CM]
    T = np.diag(t)
    trans_cm = np.array([19.86765585, -1.16965425])
    return (pixels @ R @ T) - trans_cm
