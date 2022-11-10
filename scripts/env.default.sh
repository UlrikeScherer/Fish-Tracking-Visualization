#!/bin/bash
# src/utile.py imports the paths of this file and the bash scripts in tex/ to generate the pdfs do the same

N_BATCHES=15 # SPECIFY THE TRUE NUMBER OF BATCHES per day. Which is the highest index plius one
# 15 for 8 hours and 8 for feeding f.e. 
MIN_BATCH_IDX=0 # SPECIFY THE MINIMUM BATCH-INDEX OF THE DAY TO INCLUDE
MAX_BATCH_IDX=$((N_BATCHES - 1)) # SPECIFY THE MAXIMUM BATCH-INDEX OF THE DAY TO INCLUDE
HOURS_PER_DAY=8 # SPECIFY THE NUMBER OF HOURS TRACKED PER DAY
BATCH_SIZE=10000  # Number of data frames per batch
FRAMES_PER_SECOND=5 # Number of frames per second
# --- BLOCK -------
BLOCK="block1" # SPICIFY THE BLOCK TO RUN THE SCRIPT ON (block1, block2,...)
# -----------------
# rootserver is the path to the data on the server and the directory where the data is stored
rootserver="/Volumes/data/loopbio_data/FE_(fingerprint_experiment)_SepDec2021"  # On MocOS the path to the root of the data
#path_recordings="$rootserver/FE_recordings/FE_recordings_$BLOCK/FE_${BLOCK}_recordings_*" # path to the recordings, will allway be on the server
path_recordings="/Volumes/data/loopbio_data/02_SE_(superstition_experiment)_JunNov2022/SE_recordings_noisyBackup_07/SE_recordings_phaseII_maze_trials_original/SE_recodings_phaseII_maze_block1_12Jul27Jul2022"
path_csv="$rootserver/FE_tracks"
# --- The Path to the local data ------------
path_csv_local="/Volumes/Extreme_SSD/SE_tracks_final"
POSITION_STR_FRONT="SE_phase2_tracks_final_block1_front" 
POSITION_STR_BACK="SE_phase2_tracks_final_block1_back"
dir_front=$path_csv_local/$POSITION_STR_FRONT
dir_back=$path_csv_local/$POSITION_STR_BACK

FEEDING_SHAPE="ellipse" # ellipse or rectangle
SERVER_FEEDING_TIMES_FILE="/Volumes/data/loopbio_data/02_SE_(superstition_experiment)_JunNov2022/SE_recordings_phasell_maze_trials_times.csv"
TRIAL_TIMES_CSV="/Volumes/Extreme_SSD/SE_tracks_final/SE_phase4_trial_times.csv"

export BLOCK
export rootserver
export path_csv
export path_csv_local
export path_recordings
export POSITION_STR_BACK
export POSITION_STR_FRONT
export dir_back
export dir_front
