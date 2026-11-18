import glob, json, os
from typing import Callable, Optional

import numpy as np
import fishproviz.config as config
from .utile import get_camera_pos_keys


def get_area_functions() -> Callable[[str], np.ndarray]:
    """returns a function to deliver the area, given a fish_key"""
    area_data = {k: np.array(v) for k, v in get_areas().items()}

    def area_of(key: str) -> np.ndarray:
        if key not in area_data:
            raise KeyError(f"No tank area for '{key}'; known keys: {sorted(area_data)}")
        return area_data[key]

    return area_of


def get_calibration_functions() -> Callable[[str], float]:
    """Recomputes the calibration from the area files; calibration.json is only a record."""
    calibration = compute_calibrations()

    def calibration_of(cam: str) -> float:
        camera = cam.split("_")[0]
        if camera not in calibration:
            raise KeyError(f"No calibration for camera '{camera}' (from '{cam}'); known cameras: {sorted(calibration)}")
        return calibration[camera]

    return calibration_of


def read_area_data_from_json():
    with open("{}/area_data.json".format(config.CONFIG_DATA), "r") as infile:
        area_data = json.load(infile)
        for k in area_data.keys():
            area_data[k] = np.array(area_data[k])
        return area_data


def _area_csv_files(path: str) -> list[str]:
    files = glob.glob("%s/*.csv" % (path), recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"No area csv files found in the directory: {path}")
    return files


def _parse_polygon(file: str, line: str) -> np.ndarray:
    try:
        poly = [ll.split(",") for ll in line.split("#")[2].split(";")]
        return np.array(poly).astype(np.float64)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Malformed 'Last' line in area file {file}: expected '...#<points>#<polygon>' with 'x,y' pairs ({e})") from e


def _merge_area_file(area_data: dict[str, np.ndarray], key: str, file: str) -> None:
    """Stores the smallest polygon on the file's 'Last' lines under key; raises if there is none."""
    found = False
    with open(file, "r") as file_read:
        for line in file_read.readlines():
            if "Last" in line:
                found = True
                data_a = _parse_polygon(file, line)
                if key not in area_data or area_data[key].size > data_a.size:
                    area_data[key] = data_a
    if not found:
        raise ValueError(f"No 'Last' line found in area file {file} ({key})")


def get_areas() -> dict[str, np.ndarray]:
    area_data = dict()
    example_dict = {config.FRONT: np.array([]), config.BACK: np.array([])}
    for p, path in zip([config.FRONT, config.BACK], [config.area_front, config.area_back]):
        for f in _area_csv_files(path):
            c = os.path.basename(f)[:8]
            if c.isnumeric():
                key = "%s_%s" % (c, p)
                _merge_area_file(area_data, key, f)
                if area_data[key].shape[0] == 5 and len(example_dict[p]) == 0:
                    example_dict[p] = area_data[key]

    for k, v in area_data.items():
        if v.shape[0] != 5:
            area_data[k] = update_area(example_dict[k.split("_")[1]], v)
            if area_data[k] is None:
                del area_data[k]

    searched = {config.FRONT: config.area_front, config.BACK: config.area_back}
    for key in get_camera_pos_keys():
        if key not in area_data:
            raise ValueError(f"No tank area for camera '{key}' in area folder {searched[key.split('_')[1]]}")

    write_area_data_to_json(area_data)
    return area_data


def update_area(example: np.ndarray, area: np.ndarray) -> Optional[np.ndarray]:
    indices = []
    for i, p in enumerate(example):
        x = (area[:, 0] - p[0]) ** 2
        y = (area[:, 1] - p[1]) ** 2
        idx = np.argmin(x + y)
        if idx in indices:
            return None
        indices.append(idx)
    return area[indices]


def write_area_data_to_json(area_data):
    area_d = dict(zip(area_data.keys(), map(lambda v: v.tolist(), area_data.values())))
    with open("{}/area_data.json".format(config.CONFIG_DATA), "w") as outfile:
        json.dump(area_d, outfile, indent=2)


def get_line_starting_with(file, matchstr="Last"):
    with open(file, "r") as file_read:
        for line in file_read.readlines():
            if matchstr == line[: len(matchstr)]:
                return line


def _calibration_of_file(file: str) -> float:
    line = get_line_starting_with(file)
    if line is None:
        raise ValueError(f"No line starting with 'Last' found in area file {file}")
    try:
        cal = np.array([(list(map(lambda x: int(x), coord.split(","))) if "," in coord else None) for coord in line.split("#")[1].split(";")])
        return config.CALIBRATION_DIST_CM / np.mean(np.sqrt(np.sum((cal[:2] - cal[2:]) ** 2, axis=1)))
    except (IndexError, ValueError, TypeError) as e:
        raise ValueError(f"Malformed 'Last' line in area file {file}: expected '...#<4 reference points as x,y>#...' ({e})") from e


def compute_calibrations() -> dict[str, float]:
    calibration = dict()
    for path in [config.area_front, config.area_back]:
        for file in _area_csv_files(path):
            c = os.path.basename(file)[:8]
            if c.isnumeric():
                if c not in calibration:
                    calibration[c] = _calibration_of_file(file)
    with open(f"{config.CONFIG_DATA}/calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration
