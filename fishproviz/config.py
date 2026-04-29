from dotenv import load_dotenv
import os
import fishproviz

module_path = os.path.dirname(fishproviz.__file__)
load_dotenv(os.path.join(module_path, "config.env"))


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

# --- Filtering thresholds ---
SPIKE_THRESHOLD = int(_env("SPIKE_THRESHOLD"))
DIRT_THRESHOLD = int(_env("DIRT_THRESHOLD"))
THRESHOLD_AREA_PX = int(_env("THRESHOLD_AREA_PX"))
AREA_FILTER = int(_env("AREA_FILTER"))  # 1 to filter by area, 0 to not filter
DIRT_FILTER = int(_env("DIRT_FILTER"))  # 1 to filter by dirt, 0 to not filter

# --- Dataset parameters ---
MIN_BATCH_IDX = int(_env("MIN_BATCH_IDX"))
MAX_BATCH_IDX = int(_env("MAX_BATCH_IDX"))
HOURS_PER_DAY = float(_env("HOURS_PER_DAY"))
BATCH_SIZE = int(_env("BATCH_SIZE"))
FRAMES_PER_SECOND = int(_env("FRAMES_PER_SECOND"))

# --- Metrics ---
TANGLE_N_SKIP = int(_env("TANGLE_N_SKIP"))
UNAVERAGED = bool(int(_env("UNAVERAGED")))
AGGREGATE_EVERY_N = int(_env("AGGREGATE_EVERY_N"))
REMOVE_0_VECS = bool(int(_env("REMOVE_0_VECS")))

# --- Paths ---
DIR_CSV_LOCAL = _env("path_csv_local")
PATH_RECORDINGS = _env("path_recordings")
OBJECT_ZONE_COORDS_PATH = _env("OBJECT_ZONE_COORDS_PATH")
PROJECT_ID = _env("PROJECT_ID")
dir_front = os.path.join(DIR_CSV_LOCAL, _env("POSITION_STR_FRONT"))
dir_back = os.path.join(DIR_CSV_LOCAL, _env("POSITION_STR_BACK"))

# --- Output directories ---
CONFIG_DATA = os.path.join(DIR_CSV_LOCAL, _env("CONFIG_DATA"))
VIS_DIR = os.path.join(DIR_CSV_LOCAL, _env("VIS_DIR"))
PLOTS_DIR = os.path.join(DIR_CSV_LOCAL, _env("PLOTS_DIR"))
RESULTS_PATH = os.path.join(DIR_CSV_LOCAL, _env("RESULTS"))
TEX_DIR = os.path.join(PLOTS_DIR, _env("TEX_DIR"))
DIR_TRACES = os.path.join(RESULTS_PATH, PROJECT_ID, "traces")
err_file = os.path.join(RESULTS_PATH, "log_error.csv")

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

# --- Calibration / area config ---
area_back = os.path.join(DIR_CSV_LOCAL, _env("area_folder"), _env("area_back"))
area_front = os.path.join(DIR_CSV_LOCAL, _env("area_folder"), _env("area_front"))
CALIBRATION_DIST_CM = float(_env("CALIBRATION_DIST_CM"))
DEFAULT_CALIBRATION = float(_env("DEFAULT_CALIBRATION"))


def set_config_paths(root):
    global DIR_CSV_LOCAL, CONFIG_DATA, VIS_DIR, PLOTS_DIR, RESULTS_PATH, err_file, TEX_DIR
    DIR_CSV_LOCAL = f"{root}"
    CONFIG_DATA = os.path.join(root, _env("CONFIG_DATA"))
    VIS_DIR = os.path.join(root, _env("VIS_DIR"))
    PLOTS_DIR = os.path.join(root, _env("PLOTS_DIR"))
    RESULTS_PATH = os.path.join(root, _env("RESULTS"))
    err_file = os.path.join(RESULTS_PATH, "log_error.csv")
    TEX_DIR = os.path.join(PLOTS_DIR, _env("TEX_DIR"))
    # Clear cached area/calibration functions so they are reloaded from the new paths.
    from fishproviz.utils.transformation import clear_transformation_cache

    clear_transformation_cache()


def create_directories():
    """Creates the output directory structure under DIR_CSV_LOCAL."""
    if not os.path.exists(DIR_CSV_LOCAL):
        raise Exception("path_csv_local does not exist: %s" % DIR_CSV_LOCAL)
    for d in [VIS_DIR, PLOTS_DIR, RESULTS_PATH, DIR_TRACES, TEX_DIR, CONFIG_DATA]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print("Created directory: %s" % d)
    if not os.path.exists(err_file):
        with open(err_file, "w") as f:
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
