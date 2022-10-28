from envbash import load_envbash
import os

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
ROOT_LOCAL = os.environ["root_local"]
DIR_CSV = os.environ["path_csv"]  #
DIR_CSV_LOCAL = os.environ["path_csv_local"]  #
BLOCK = os.environ["BLOCK"]  # block1 or block2
BLOCK1 = "block1"
BLOCK2 = "block2"
FRONT, BACK = "front", "back"
PLOTS_TRAJECTORY = os.environ["PLOTS_TRAJECTORY"]
DATA_DIR = "data"
VIS_DIR = os.environ["VIS_DIR"]

# TRAJECTORY
dir_front = os.environ["dir_front"]
dir_back = os.environ["dir_back"]
STIME = os.environ["STIME"]

# FEEDING
dir_feeding_front = os.environ["dir_feeding_front"]
dir_feeding_back = os.environ["dir_feeding_back"]
FEEDINGTIME = os.environ["FEEDINGTIME"]
START_END_FEEDING_TIMES = f"{DATA_DIR}/DevEx_FE_feeding_times.csv"

N_BATCHES = int(os.environ["N_BATCHES"])
N_BATCHES_FEEDING = int(os.environ["N_BATCHES_FEEDING"])
HOURS_PER_DAY = int(os.environ["HOURS_PER_DAY"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
FRAMES_PER_SECOND = int(os.environ["FRAMES_PER_SECOND"])
N_SECONDS_PER_HOUR = 3600
N_SECONDS_OF_DAY = 24 * N_SECONDS_PER_HOUR

# METRICS
DATA_results = "results"
float_format = "%.10f"
sep = ";"

DIR_TRACES = "%s/%s/%s" % (DATA_results, BLOCK, "traces")
# TRACE TABLE
CAM_POS = "CAMERA_POSITION"
DAY = "DAY"
BATCH = "BATCH"
DATAFRAME = "DATEFRAME"
