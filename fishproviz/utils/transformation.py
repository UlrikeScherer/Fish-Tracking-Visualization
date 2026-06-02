import functools
from typing import Optional

import numpy as np
from .tank_area_config import get_area_functions, get_calibration_functions
import fishproviz.config as config


def normalize_origin_of_compartment(data: np.ndarray, area: np.ndarray, is_back: bool) -> tuple[np.ndarray, np.ndarray]:
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


def rotation(t: float) -> np.ndarray:
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


@functools.lru_cache(maxsize=1)
def _calibration_functions():
    return get_calibration_functions()


@functools.lru_cache(maxsize=1)
def _area_functions():
    return get_area_functions()


def clear_transformation_cache() -> None:
    _calibration_functions.cache_clear()
    _area_functions.cache_clear()


def px2cm(a: float, fish_key: Optional[str] = None) -> float:
    funcs = _calibration_functions()
    if fish_key:
        return a * funcs(fish_key)
    return a * config.DEFAULT_CALIBRATION


def pixel_to_cm(pixels: np.ndarray, fish_key: Optional[str] = None) -> np.ndarray:
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    area_funcs = _area_functions()
    if area_funcs(fish_key) is None:
        origin = np.array([450, 450])  # default origin
    else:
        origin = area_funcs(fish_key)[1]  # origin of the area is the second point
    pixels = pixels - origin
    funcs = _calibration_functions()
    R = rotation(np.pi / 4)
    if fish_key:
        t = [funcs(fish_key), funcs(fish_key)]
    else:
        t = [config.DEFAULT_CALIBRATION, config.DEFAULT_CALIBRATION]
    T = np.diag(t)
    cm_data = pixels @ R @ T
    return cm_data
