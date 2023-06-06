import numpy as np
from .tank_area_config import get_area_functions, get_calibration_functions
import fishproviz.config as config

FUNCS_PX2CM = None
AREA_FUNCS = None


def normalize_origin_of_compartment(data, area, is_back):
    if is_back:
        origin1 = area[0, 0], area[1, 1]
        new_area = area - origin1
        origin2 = new_area[2, 0], new_area[3, 1]
        new_area = -new_area + origin2
        data = -data + origin1 + origin2
    else:
        origin1 = area[1, 0], area[0, 1]
        new_area = area - origin1
        data = data - origin1
    return data, new_area


def rotation(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def px2cm(a, fish_key=None):
    global FUNCS_PX2CM
    if not FUNCS_PX2CM:
        FUNCS_PX2CM = get_calibration_functions()
    if fish_key:
        return a * FUNCS_PX2CM(fish_key)
    return a * config.DEFAULT_CALIBRATION


def pixel_to_cm(pixels, fish_key=None):
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    global FUNCS_PX2CM
    global AREA_FUNCS
    if not AREA_FUNCS:
        AREA_FUNCS = get_area_functions()
    if AREA_FUNCS(fish_key) is None:
        origin = np.array([450, 450])  # default origin
    else:
        origin = AREA_FUNCS(fish_key)[1]  # origin of the area is the second point
    pixels = pixels - origin
    if not FUNCS_PX2CM:
        FUNCS_PX2CM = get_calibration_functions()
    R = rotation(np.pi / 4)
    if fish_key:
        t = [FUNCS_PX2CM(fish_key), FUNCS_PX2CM(fish_key)]
    else:
        t = [config.DEFAULT_CALIBRATION, config.DEFAULT_CALIBRATION]
    T = np.diag(t)
    cm_data = pixels @ R @ T
    return cm_data
