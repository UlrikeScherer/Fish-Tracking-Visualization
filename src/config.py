from envbash import load_envbash
import os
load_envbash('scripts/env.sh')

# Calculated MEAN and SD for the data set filtered for erroneous frames
MEAN_GLOBAL = 0.22746102241709162
SD_GLOBAL = 1.0044248513034164
# THRESHOLDS for the data set filtered for erroneous frames
SPIKE_THRESHOLD = 15 # In centimeters. to consider a step as a spike (alternative definition MEAN_GLOBAL + 3 * SD_GLOBAL)
DIRT_THRESHOLD = 60*5 # Threshold for dirt detection, indicates the number of consecutive frames that, when equal, are classified as dirt.
THRESHOLD_AREA_PX = 50 # The threshold in pixels for the exclusion of data points that are not within the area of the tank.
BATCH_SIZE = 10000 #9999
FRAMES_PER_SECOND = 5
ROOT=os.environ["rootserver"]
ROOT_LOCAL=os.environ["root_local"]
DIR_CSV=os.environ["path_csv"] #
DIR_CSV_LOCAL=os.environ["path_csv_local"] #
BLOCK = os.environ["BLOCK"] # block1 or block2
BLOCK1 = "block1"
BLOCK2 = "block2"

# TRAJECTORY
dir_front = os.environ["dir_front"]
dir_back = os.environ["dir_back"]
STIME = os.environ["STIME"]

# FEEDING
dir_feeding_front = os.environ["dir_feeding_front"]
dir_feeding_back = os.environ["dir_feeding_back"]
FEEDINGTIME = os.environ["FEEDINGTIME"]

FRONT, BACK = "front", "back"
ROOT_img = "plots"
DATA_DIR = "data"
VIS_DIR="vis"

N_BATCHES=15
N_BATCHES_FEEDING=8

N_FISHES = 24
N_DAYS = 28
HOURS_PER_DAY = 8
N_SECONDS_PER_HOUR = 3600
N_SECONDS_OF_DAY = 24*N_SECONDS_PER_HOUR

# METRICS 
DATA_results = "results"
float_format='%.10f'
sep=";"

# TRACE TABLE 
CAM_POS = "CAMERA_POSITION"
DAY = "DAY"
BATCH = "BATCH"
DATAFRAME="DATEFRAME"

