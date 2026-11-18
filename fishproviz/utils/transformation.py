import functools
import numpy as np
from .tank_area_config import get_area_functions, get_calibration_functions


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


def px2cm(a: float, fish_key: str) -> float:
    return a * _calibration_functions()(fish_key)


def pixel_to_cm(pixels: np.ndarray, fish_key: str) -> np.ndarray:
    """
    @params: pixels (Nx2)
    returns: cm (Nx2)
    """
    origin = _area_functions()(fish_key)[1]  # origin of the area is the second point
    pixels = pixels - origin
    scale = _calibration_functions()(fish_key)
    return pixels @ rotation(np.pi / 4) @ np.diag([scale, scale])
