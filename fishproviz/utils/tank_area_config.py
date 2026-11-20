import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import fishproviz.config as config
from .utile import get_camera_pos_keys

_LAST_PREFIX = "Last#"
_POLYGON_VERTICES = 5
_REFERENCE_POINTS = 4


@dataclass(frozen=True, eq=False)
class AreaFile:
    """One BioTracker area file: its camera, 5-point tank polygon and 4 calibration reference points (px)."""

    path: str
    camera: str
    polygon: NDArray[np.float64]
    reference_points: NDArray[np.float64]


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


def _parse_points(path: str, field: str, what: str) -> NDArray[np.float64]:
    try:
        return np.array([[float(v) for v in point.split(",", 1)] for point in field.split(";")], dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Area file {path}: malformed {what} '{field}', expected 'x,y' pairs separated by ';' ({e})") from e


def _segments_intersect(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64], d: NDArray[np.float64]) -> bool:
    """True if the closed segments ab and cd share a point (touching and collinear overlap included)."""

    def orient(p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]) -> float:
        return float(np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])))

    def on_segment(p: NDArray[np.float64], q: NDArray[np.float64], r: NDArray[np.float64]) -> bool:
        return bool(min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4):
        return True
    return any(o == 0 and on_segment(p, q, r) for o, p, q, r in ((o1, a, b, c), (o2, a, b, d), (o3, c, d, a), (o4, c, d, b)))


def _polygon_violation(polygon: NDArray[np.float64]) -> str | None:
    """Returns the first broken polygon rule, or None for a valid simple 5-point polygon."""
    if polygon.shape != (_POLYGON_VERTICES, 2):
        return f"expected {_POLYGON_VERTICES} vertices of shape ({_POLYGON_VERTICES}, 2), got shape {polygon.shape}"
    if not np.all(np.isfinite(polygon)):
        return "all vertices must be finite"
    n = len(polygon)
    for i in range(n):
        if np.array_equal(polygon[i], polygon[(i + 1) % n]):
            return f"consecutive vertices {i} and {(i + 1) % n} are equal ({polygon[i].tolist()})"
    for i in range(n):
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            if _segments_intersect(polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return f"not simple: edges {i}-{(i + 1) % n} and {j}-{(j + 1) % n} intersect"
    return None


def parse_area_file(path: str) -> AreaFile:
    """Parses the single 'Last#' line of a BioTracker area file into a validated AreaFile."""
    camera = os.path.basename(path)[:8]
    with open(path, "r") as f:
        lines = [line.rstrip("\r\n") for line in f if line.startswith(_LAST_PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"Area file {path} (camera {camera}) must contain exactly one line starting with '{_LAST_PREFIX}', found {len(lines)}")
    fields = lines[0].split("#")
    if len(fields) < 3:
        raise ValueError(f"Area file {path} (camera {camera}): expected '{_LAST_PREFIX}<reference points>#<polygon>#...', got {len(fields)} '#' fields")
    reference_points = _parse_points(path, fields[1], "reference points")
    if reference_points.shape != (_REFERENCE_POINTS, 2) or not np.all(np.isfinite(reference_points)):
        raise ValueError(f"Area file {path} (camera {camera}): expected {_REFERENCE_POINTS} finite reference points, got shape {reference_points.shape}")
    if np.any(np.all(reference_points[:2] == reference_points[2:], axis=1)):
        raise ValueError(f"Area file {path} (camera {camera}): a calibration reference pair (points i and i+2) coincides, so the px-to-cm scale is undefined")
    polygon = _parse_points(path, fields[2], "polygon")
    violation = _polygon_violation(polygon)
    if violation is not None:
        raise ValueError(f"Area file {path} (camera {camera}): invalid tank polygon, {violation}")
    return AreaFile(path=path, camera=camera, polygon=polygon, reference_points=reference_points)


def _area_files(folder: str) -> dict[str, AreaFile]:
    """Parses the folder's area files in sorted order, keyed by camera; files without an 8-digit camera prefix are skipped."""
    files = sorted(glob(os.path.join(folder, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"No area csv files found in the directory: {folder}")
    by_camera: dict[str, AreaFile] = {}
    for path in files:
        if not os.path.basename(path)[:8].isnumeric():
            continue
        area = parse_area_file(path)
        if area.camera in by_camera:
            raise ValueError(f"Two area files for camera {area.camera} in {folder}: {by_camera[area.camera].path} and {path}")
        by_camera[area.camera] = area
    return by_camera


def get_areas() -> dict[str, np.ndarray]:
    """Returns the tank polygon per '<camera>_<side>' key and records them in area_data.json."""
    searched = {config.FRONT: config.area_front, config.BACK: config.area_back}
    area_data = {f"{camera}_{side}": area.polygon for side, folder in searched.items() for camera, area in _area_files(folder).items()}
    for key in get_camera_pos_keys():
        if key not in area_data:
            raise ValueError(f"No tank area for camera '{key}' in area folder {searched[key.split('_')[1]]}")
    write_area_data_to_json(area_data)
    return area_data


def write_area_data_to_json(area_data: dict[str, np.ndarray]) -> None:
    """Writes the polygons to CONFIG_DATA/area_data.json as a record."""
    with open(f"{config.CONFIG_DATA}/area_data.json", "w") as outfile:
        json.dump({k: v.tolist() for k, v in area_data.items()}, outfile, indent=2)


def compute_calibrations() -> dict[str, float]:
    """Returns px-to-cm per camera from the reference points of its front file (back if there is none); records calibration.json."""
    calibration: dict[str, float] = {}
    for folder in [config.area_front, config.area_back]:
        for camera, area in _area_files(folder).items():
            if camera not in calibration:
                ref = area.reference_points
                calibration[camera] = config.CALIBRATION_DIST_CM / np.mean(np.sqrt(np.sum((ref[:2] - ref[2:]) ** 2, axis=1)))
    with open(f"{config.CONFIG_DATA}/calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration
