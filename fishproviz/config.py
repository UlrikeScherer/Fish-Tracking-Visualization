from envbash import load_envbash
import os
import fishproviz

module_path = os.path.dirname(fishproviz.__file__)
load_envbash(module_path + "/config.env")
# THRESHOLDS for the data set filtered for erroneous frames
SPIKE_THRESHOLD = int(os.environ["SPIKE_THRESHOLD"])
DIRT_THRESHOLD = int(os.environ["DIRT_THRESHOLD"])
THRESHOLD_AREA_PX = int(os.environ["THRESHOLD_AREA_PX"])
# FILTERING
AREA_FILTER = int(os.environ["AREA_FILTER"])  # 1 to filter by area, 0 to not filter
DIRT_FILTER = int(os.environ["DIRT_FILTER"])  # 1 to filter by dirt, 0 to not filter
ROOT = os.environ["rootserver"]
DIR_CSV = os.environ["path_csv"]
DIR_CSV_LOCAL = os.environ["path_csv_local"]
PATH_RECORDINGS = os.environ["path_recordings"]
FRONT, BACK = "front", "back"

projectPath = "/Volumes/Extreme_SSD/content/Fish_moves_final"
CONFIG_DATA = f"{DIR_CSV_LOCAL}/" + os.environ["CONFIG_DATA"]
VIS_DIR = f"{DIR_CSV_LOCAL}/" + os.environ["VIS_DIR"]
PLOTS_DIR = f"{DIR_CSV_LOCAL}/" + os.environ["PLOTS_DIR"]
RESULTS_PATH = f"{DIR_CSV_LOCAL}/" + os.environ["RESULTS"]
P_TRAJECTORY = os.environ["P_TRAJECTORY"]
P_FEEDING = os.environ["P_FEEDING"]
TEX_DIR = f"{PLOTS_DIR}/" + os.environ["TEX_DIR"]
FEEDING_PATCH_COORDS_FILE = os.environ["FEEDING_PATCH_COORDS_FILE"]
FEEDING_PATCH_COORDS_SEP = os.environ["FEEDING_PATCH_COORDS_SEP"]
SERVER_FEEDING_TIMES_FILE = os.environ[
    "SERVER_FEEDING_TIMES_FILE"
]  # "/Volumes/Extreme_SSD/SE_tracks_final/SE_recordings_phasell_maze_trials_times.csv"
SERVER_FEEDING_TIMES_SEP = os.environ["SERVER_FEEDING_TIMES_SEP"]
TRIAL_TIMES_CSV = os.environ["TRIAL_TIMES_CSV"]
START_END_FEEDING_TIMES_FILE = f"{CONFIG_DATA}/recordings_feeding_times.json"  #
MAZE_FILE = f"maze_data.json"

FEEDING_SHAPE = os.environ["FEEDING_SHAPE"]  # "square", "ellipse"
FEEDING_SHAPE_WIDTH = float(os.environ["FEEDING_SHAPE_WIDTH"]) # e.g.: 5 for 5cm width
FEEDING_SHAPE_HEIGHT = float(os.environ["FEEDING_SHAPE_HEIGHT"]) # e.g.: 5 for 5cm height
MAGNET_LENGTH_CM = float(os.environ["MAGNET_LENGTH_CM"])
# TRAJECTORY
dir_front = os.environ["dir_front"]
dir_back = os.environ["dir_back"]
PROJECT_ID = os.environ["PROJECT_ID"]
# print("PROJECTID:",PROJECT_ID)
DIR_TRACES = "%s/%s/%s" % (RESULTS_PATH, PROJECT_ID, "traces")

N_BATCHES = int(os.environ["N_BATCHES"])
MIN_BATCH_IDX = int(os.environ["MIN_BATCH_IDX"])
MAX_BATCH_IDX = int(os.environ["MAX_BATCH_IDX"])
HOURS_PER_DAY = float(os.environ["HOURS_PER_DAY"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
FRAMES_PER_SECOND = int(os.environ["FRAMES_PER_SECOND"])
N_SECONDS_PER_HOUR = 3600
N_SECONDS_OF_DAY = 24 * N_SECONDS_PER_HOUR

# METRICS
float_format = "%.10f"
sep = ";"

# Calibrations
# AREA CONFIG
area_back = os.environ["area_back"]
area_front = os.environ["area_front"]
CALIBRATION_DIST_CM = float(os.environ["CALIBRATION_DIST_CM"])
DEFAULT_CALIBRATION = float(os.environ["DEFAULT_CALIBRATION"])
err_file = f"{RESULTS_PATH}/log_error.csv"


def set_config_paths(root):
    global DIR_CSV_LOCAL, CONFIG_DATA, VIS_DIR, PLOTS_DIR, RESULTS_PATH, err_file, TEX_DIR
    DIR_CSV_LOCAL = f"{root}"
    CONFIG_DATA = f"{root}/" + os.environ["CONFIG_DATA"]
    VIS_DIR = f"{root}/" + os.environ["VIS_DIR"]
    PLOTS_DIR = f"{root}/" + os.environ["PLOTS_DIR"]
    RESULTS_PATH = f"{root}/" + os.environ["RESULTS"]
    err_file = f"{RESULTS_PATH}/log_error.csv"
    TEX_DIR = f"{PLOTS_DIR}/" + os.environ["TEX_DIR"]


def create_directories():
    """
    Creates the directories used in the project
    """
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
