import numpy as np
from .tank_area_config import get_calibration_functions
from fishproviz.config import DEFAULT_CALIBRATION

FUNCS_PX2CM = None

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


def px2cm(a, fish_key=None):
    global FUNCS_PX2CM
    if not FUNCS_PX2CM:
        FUNCS_PX2CM = get_calibration_functions()
    if fish_key: return a * FUNCS_PX2CM(fish_key)
    return a * DEFAULT_CALIBRATION


def pixel_to_cm(pixels, fish_key=None):
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    global FUNCS_PX2CM
    if not FUNCS_PX2CM:
        FUNCS_PX2CM = get_calibration_functions()
    R = rotation(np.pi / 4)
    if fish_key:
        t = [FUNCS_PX2CM(fish_key), FUNCS_PX2CM(fish_key)]
    else:
        t = [DEFAULT_CALIBRATION, DEFAULT_CALIBRATION]
    T = np.diag(t)
    trans_cm = np.array([19.86765585, -1.16965425])
    return (pixels @ R @ T) - trans_cm
