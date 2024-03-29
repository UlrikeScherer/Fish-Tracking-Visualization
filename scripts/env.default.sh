#!/bin/bash

N_BATCHES=15 # SPECIFY THE TRUE NUMBER OF BATCHES per day. Which is the highest index plus one
# 15 for 8 hours and 8 for feeding f.e. 
MIN_BATCH_IDX=0 # SPECIFY THE MINIMUM BATCH-INDEX OF THE DAY TO INCLUDE
MAX_BATCH_IDX=$((N_BATCHES - 1)) # SPECIFY THE MAXIMUM BATCH-INDEX OF THE DAY TO INCLUDE
HOURS_PER_DAY=8 # SPECIFY THE NUMBER OF HOURS TRACKED PER DAY
BATCH_SIZE=10000  # Number of data frames per batch
FRAMES_PER_SECOND=5 # Number of frames per second
PROJECT_ID="block1" # The BLOCKID of the experiment, not necessar to specify, just a name for directories

# ---- PATHS -----
rootserver="/Volumes/data/loopbio_data/FE_(fingerprint_experiment)_SepDec2021"  # On MocOS the path to the root of the data
path_recordings="/Volumes/data/loopbio_data/02_SE_(superstition_experiment)_JunNov2022/SE_recordings_noisyBackup_07/SE_recordings_phaseII_maze_trials_original/SE_recodings_phaseII_maze_block1_12Jul27Jul2022"
path_csv="$rootserver/FE_tracks"
# --- The Path to the local data ------------
path_csv_local="/Volumes/Extreme_SSD/SE_tracks_final"
POSITION_STR_FRONT="front" 
POSITION_STR_BACK="back"
dir_front=$path_csv_local/$POSITION_STR_FRONT
dir_back=$path_csv_local/$POSITION_STR_BACK

FEEDING_SHAPE="ellipse" # ellipse, patch or ( rectangle not implimented yet )
SERVER_FEEDING_TIMES_FILE="/Volumes/data/loopbio_data/02_SE_(superstition_experiment)_JunNov2022/SE_recordings_phasell_maze_trials_times.csv"
TRIAL_TIMES_CSV="/Volumes/Extreme_SSD/SE_tracks_final/SE_phase4_trial_times.csv"

# Calibration 
CALIBRATION_DIST_CM=83.0
DEFAULT_CALIBRATION=0.02278

# Area Data Directory 
area_back="$path_csv_local/area_config/areas_back"
area_front="$path_csv_local/area_config/areas_front"

# FILTERING 
AREA_FILTER=0 # 1 to filter by area, 0 to not filter
DIRT_FILTER=0 # 1 to filter by dirt, 0 to not filter
SPIKE_THRESHOLD=8  # In centimeters. to consider a step as a spike (alternative definition MEAN_GLOBAL + 3 * SD_GLOBAL)
DIRT_THRESHOLD=300  # Threshold for dirt detection, indicates the number of consecutive frames that, when equal, are classified as dirt.
THRESHOLD_AREA_PX=50  # The threshold in pixels for the exclusion of data points that are not within the area of the tank.

# shared variables that are used in the scripts
# NO Changes needed
VIS_DIR="visualisations" # path to stroe the visualisations
PLOTS_DIR="$VIS_DIR/plots" # path to the plots
P_FEEDING="feeding"
P_TRAJECTORY="trajectory"
CONFIG_DATA="config_data" # To stroe the config data, feeding times, area coordinates, calibration, etc.
RESULTS="results" # To store the results of the analysis
TEX_DIR="tex" # To store the tex files

export rootserver
export path_csv
export path_csv_local
export path_recordings
export POSITION_STR_BACK
export POSITION_STR_FRONT
export dir_back
export dir_front
