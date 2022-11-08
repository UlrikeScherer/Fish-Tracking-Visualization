import shutil
from envbash import load_envbash
import os, warnings
import pandas as pd

load_envbash("scripts/config.sh")
load_envbash("scripts/env.sh")

# Calculated MEAN and SD for the data set filtered for erroneous frames
MEAN_GLOBAL = 0.22746102241709162
SD_GLOBAL = 1.0044248513034164
# THRESHOLDS for the data set filtered for erroneous frames
SPIKE_THRESHOLD = 8  # In centimeters. to consider a step as a spike (alternative definition MEAN_GLOBAL + 3 * SD_GLOBAL)
DIRT_THRESHOLD = (
    60 * 5
)  # Threshold for dirt detection, indicates the number of consecutive frames that, when equal, are classified as dirt.
THRESHOLD_AREA_PX = 50  # The threshold in pixels for the exclusion of data points that are not within the area of the tank.
ROOT = os.environ["rootserver"]
DIR_CSV = os.environ["path_csv"]  #
DIR_CSV_LOCAL = os.environ["path_csv_local"]  #
PATH_RECORDINGS=os.environ["path_recordings"]
BLOCK = os.environ["BLOCK"]  # block1 or block2
BLOCK1 = "block1"
BLOCK2 = "block2"
FRONT, BACK = "front", "back"

CONFIG_DATA_PATH = f"{DIR_CSV_LOCAL}/" + os.environ["CONFIG_DATA"]
VIS_DIR = f"{DIR_CSV_LOCAL}/" + os.environ["VIS_DIR"]
PLOTS_DIR = f"{DIR_CSV_LOCAL}/" + os.environ["PLOTS_DIR"]
RESULTS_PATH = f"{DIR_CSV_LOCAL}/" + os.environ["RESULTS"]
P_TRAJECTORY = os.environ["P_TRAJECTORY"]
P_FEEDING = os.environ["P_FEEDING"]
DIR_TRACES = "%s/%s/%s" % (RESULTS_PATH, BLOCK, "traces")
TEX_DIR = f"{PLOTS_DIR}/" + os.environ["TEX_DIR"]
SERVER_FEEDING_TIMES_FILE= "/Volumes/Extreme_SSD/SE_tracks_final/SE_recordings_phasell_maze_trials_times.csv" # os.environ["SERVER_FEEDING_TIMES_FILE"]

START_END_FEEDING_TIMES_FILE = f"{CONFIG_DATA_PATH}/{BLOCK}_recordings_feeding_times.json" #
MAZE_FILE = f"{BLOCK}_maze_data.json"

FEEDING_SHAPE = os.environ["FEEDING_SHAPE"] #"square", "ellipse"
# TRAJECTORY
dir_front = os.environ["dir_front"]
dir_back = os.environ["dir_back"]

N_BATCHES = int(os.environ["N_BATCHES"])
MIN_BATCH_IDX = int(os.environ["MIN_BATCH_IDX"])
MAX_BATCH_IDX = int(os.environ["MAX_BATCH_IDX"])
HOURS_PER_DAY = int(os.environ["HOURS_PER_DAY"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
FRAMES_PER_SECOND = int(os.environ["FRAMES_PER_SECOND"])
N_SECONDS_PER_HOUR = 3600
N_SECONDS_OF_DAY = 24 * N_SECONDS_PER_HOUR

# METRICS
float_format = "%.10f"
sep = ";"

# TRACE TABLE
CAM_POS = "CAMERA_POSITION"
DAY = "DAY"
BATCH = "BATCH"
DATAFRAME = "DATEFRAME"

def create_directories():
    """
    Creates the directories used in the project
    """
    for d in [VIS_DIR, PLOTS_DIR, RESULTS_PATH, DIR_TRACES, TEX_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
