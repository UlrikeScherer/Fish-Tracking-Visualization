from dataclasses import asdict, dataclass, fields
from dotenv import load_dotenv
import os
import fishproviz

module_path = os.path.dirname(fishproviz.__file__)
load_dotenv(os.path.join(module_path, "config.env"), override=True)


def _env(key: str) -> str:
    val = os.environ.get(key)
    if val is None:
        raise KeyError(f"Missing required config variable '{key}'. " f"Copy fishproviz/default_config.env to fishproviz/config.env and fill in the values.")
    return val


# --- Hardcoded constants ---
# TODO: FRONT and BACK are domain identifiers, not user config. Move to a
#       constants module and remove from config (touches 28+ sites across 8 files).
FRONT, BACK = "front", "back"
# TODO: sep is never changed in practice, consider adding to above constants module.
sep = ";"
MAZE_FILE = "maze_data.json"


def validate_config(
    *,
    spike_threshold: float,
    frames_per_second: int,
    batch_size: int,
    min_batch_idx: int,
    max_batch_idx: int,
    aggregate_every_n: int,
    tortuosity_window_cm: float,
    tortuosity_min_l_cm: float,
    entropy_bins: int,
    entropy_margin_px: int,
) -> None:
    """Raises ValueError naming the key and value of the first invalid setting."""
    if not spike_threshold > 0:
        raise ValueError(f"SPIKE_THRESHOLD must be > 0 (cm per frame), got {spike_threshold}")
    if not frames_per_second > 0:
        raise ValueError(f"FRAMES_PER_SECOND must be > 0, got {frames_per_second}")
    if not batch_size > 0:
        raise ValueError(f"BATCH_SIZE must be > 0, got {batch_size}")
    if not 0 <= min_batch_idx <= max_batch_idx:
        raise ValueError(f"Batch indices must satisfy 0 <= MIN_BATCH_IDX <= MAX_BATCH_IDX, got MIN_BATCH_IDX={min_batch_idx}, MAX_BATCH_IDX={max_batch_idx}")
    if not aggregate_every_n >= 1:
        raise ValueError(f"AGGREGATE_EVERY_N must be >= 1, got {aggregate_every_n}")
    if not tortuosity_window_cm > 0:
        raise ValueError(f"TORTUOSITY_WINDOW_CM must be > 0 (cm), got {tortuosity_window_cm}")
    if not tortuosity_min_l_cm > 0:
        raise ValueError(f"TORTUOSITY_MIN_L_CM must be > 0 (cm), got {tortuosity_min_l_cm}")
    if not entropy_bins >= 1:
        raise ValueError(f"ENTROPY_BINS must be >= 1 (cells per side), got {entropy_bins}")
    if not entropy_margin_px >= 0:
        raise ValueError(f"ENTROPY_MARGIN_PX must be >= 0 (px), got {entropy_margin_px}")


# --- Filtering thresholds ---
SPIKE_THRESHOLD = float(_env("SPIKE_THRESHOLD"))  # cm per frame
DIRT_THRESHOLD = int(_env("DIRT_THRESHOLD"))
THRESHOLD_AREA_PX = int(_env("THRESHOLD_AREA_PX"))
AREA_FILTER = int(_env("AREA_FILTER"))  # 1 to filter by area, 0 to not filter
DIRT_FILTER = int(_env("DIRT_FILTER"))  # 1 to filter by dirt, 0 to not filter

# --- Dataset parameters ---
MIN_BATCH_IDX = int(_env("MIN_BATCH_IDX"))
MAX_BATCH_IDX = int(_env("MAX_BATCH_IDX"))
BATCH_SIZE = int(_env("BATCH_SIZE"))
FRAMES_PER_SECOND = int(_env("FRAMES_PER_SECOND"))

# --- Metrics ---
TANGLE_N_SKIP = int(_env("TANGLE_N_SKIP"))
UNAVERAGED = bool(int(_env("UNAVERAGED")))
AGGREGATE_EVERY_N = int(_env("AGGREGATE_EVERY_N"))
REMOVE_0_VECS = bool(int(_env("REMOVE_0_VECS")))
DIST_FROM_WALL_TANGLE_IGNORED = float(_env("DIST_FROM_WALL_TANGLE_IGNORED"))
TANGLE_MIN_STEP_PX = float(_env("TANGLE_MIN_STEP_PX"))
STREAK_STRAIGHT_PX = float(_env("STREAK_STRAIGHT_PX"))
TORTUOSITY_WINDOW_CM = float(_env("TORTUOSITY_WINDOW_CM"))  # path length per window
TORTUOSITY_MIN_L_CM = float(_env("TORTUOSITY_MIN_L_CM"))  # floor on the straight-line distance
ENTROPY_BINS = int(_env("ENTROPY_BINS"))  # cells per side of the entropy grid
ENTROPY_MARGIN_PX = int(_env("ENTROPY_MARGIN_PX"))  # pixels added around the compartment bounding box

validate_config(
    spike_threshold=SPIKE_THRESHOLD,
    frames_per_second=FRAMES_PER_SECOND,
    batch_size=BATCH_SIZE,
    min_batch_idx=MIN_BATCH_IDX,
    max_batch_idx=MAX_BATCH_IDX,
    aggregate_every_n=AGGREGATE_EVERY_N,
    tortuosity_window_cm=TORTUOSITY_WINDOW_CM,
    tortuosity_min_l_cm=TORTUOSITY_MIN_L_CM,
    entropy_bins=ENTROPY_BINS,
    entropy_margin_px=ENTROPY_MARGIN_PX,
)
SQRT_N = (MAX_BATCH_IDX + 2 - MIN_BATCH_IDX) ** 0.5
FIG_WIDTH = 0.95 / (-(-SQRT_N // 1.0))

# --- Paths ---
PATH_RECORDINGS = _env("path_recordings")
OBJECT_ZONE_COORDS_PATH = _env("OBJECT_ZONE_COORDS_PATH")
PROJECT_ID = _env("PROJECT_ID")


@dataclass(frozen=True)
class RootPaths:
    """Every path derived from the data root directory."""

    DIR_CSV_LOCAL: str
    dir_front: str
    dir_back: str
    CONFIG_DATA: str
    VIS_DIR: str
    PLOTS_DIR: str
    RESULTS_PATH: str
    TEX_DIR: str
    DIR_TRACES: str
    err_file: str
    area_front: str
    area_back: str


def build_root_paths(root: str) -> RootPaths:
    """Derives all root-dependent paths from the root and the env names."""
    plots_dir = os.path.join(root, _env("PLOTS_DIR"))
    results_path = os.path.join(root, _env("RESULTS"))
    area_root = os.path.join(root, _env("area_folder"))
    return RootPaths(
        DIR_CSV_LOCAL=str(root),
        dir_front=os.path.join(root, _env("POSITION_STR_FRONT")),
        dir_back=os.path.join(root, _env("POSITION_STR_BACK")),
        CONFIG_DATA=os.path.join(root, _env("CONFIG_DATA")),
        VIS_DIR=os.path.join(root, _env("VIS_DIR")),
        PLOTS_DIR=plots_dir,
        RESULTS_PATH=results_path,
        TEX_DIR=os.path.join(plots_dir, _env("TEX_DIR")),
        DIR_TRACES=os.path.join(results_path, PROJECT_ID, "traces"),
        err_file=os.path.join(results_path, "log_error.csv"),
        area_front=os.path.join(area_root, _env("area_front")),
        area_back=os.path.join(area_root, _env("area_back")),
    )


def _apply_root_paths(root: str) -> None:
    globals().update(asdict(build_root_paths(root)))


def _current_root_paths() -> RootPaths:
    """Reads the live module attributes, which tests and callers may have set directly."""
    return RootPaths(**{f.name: globals()[f.name] for f in fields(RootPaths)})


_apply_root_paths(_env("path_csv_local"))

# --- Program section names ---
# TODO: These are output directory names that users should never change. Move
#       to a constants module alongside FRONT/BACK and drop from default_config.env.
P_TRAJECTORY = _env("P_TRAJECTORY")
P_FEEDING = _env("P_FEEDING")
P_NOVEL_OBJECT = _env("P_NOVEL_OBJECT")
P_SOCIABILITY = _env("P_SOCIABILITY")

# --- Feeding experiment ---
FEEDING_SHAPE = _env("FEEDING_SHAPE")  # "patch", "ellipse"
FEEDING_SHAPE_WIDTH = float(_env("FEEDING_SHAPE_WIDTH"))
FEEDING_SHAPE_HEIGHT = float(_env("FEEDING_SHAPE_HEIGHT"))
MAGNET_LENGTH_CM = float(_env("MAGNET_LENGTH_CM"))
FEEDING_PATCH_COORDS_FILE = _env("FEEDING_PATCH_COORDS_FILE")
FEEDING_PATCH_COORDS_SEP = _env("FEEDING_PATCH_COORDS_SEP")
SERVER_FEEDING_TIMES_FILE = _env("SERVER_FEEDING_TIMES_FILE")
SERVER_FEEDING_TIMES_SEP = _env("SERVER_FEEDING_TIMES_SEP")
TRIAL_TIMES_CSV = _env("TRIAL_TIMES_CSV")

# --- Calibration ---
CALIBRATION_DIST_CM = float(_env("CALIBRATION_DIST_CM"))


def set_config_paths(root: str) -> None:
    """Points every root-derived path at a new data root and drops cached calibration/area functions."""
    _apply_root_paths(root)
    from fishproviz.utils.transformation import clear_transformation_cache

    clear_transformation_cache()


def create_directories():
    """Creates the output directory structure under DIR_CSV_LOCAL."""
    paths = _current_root_paths()
    if not os.path.exists(paths.DIR_CSV_LOCAL):
        raise Exception("path_csv_local does not exist: %s" % paths.DIR_CSV_LOCAL)
    for d in [paths.VIS_DIR, paths.PLOTS_DIR, paths.RESULTS_PATH, paths.DIR_TRACES, paths.TEX_DIR, paths.CONFIG_DATA]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print("Created directory: %s" % d)
    if not os.path.exists(paths.err_file):
        with open(paths.err_file, "w") as f:
            f.write(
                ";".join(
                    [
                        "fish_key",
                        "day",
                        "duration",
                        "xpx",
                        "ypx",
                        "start_idx",
                        "end_idx",
                    ]
                )
                + "\n"
            )
