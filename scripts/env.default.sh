#!/bin/bash
# src/utile.py imports the paths of this file and the bash scripts in tex/ to generate the pdfs do the same
# --- BLOCK -------
BLOCK="block1" # SPICIFY THE BLOCK TO RUN THE SCRIPT ON (block1, block2,...)
# -----------------
RECORDINGTIME="060000" # START TIME FOR THE EXPERIMENT
FEEDINGTIME="140000" # START TIME FOR THE FEEDING SETUP
STIME=$RECORDINGTIME   
rootserver="/Volumes/data/loopbio_data/FE_(fingerprint_experiment)_SepDec2021"  # On MocOS the path to the root of the data
path_recordings="$rootserver/FE_recordings/FE_recordings_$BLOCK/FE_${BLOCK}_recordings_*" # path to the recordings, will allway be on the server

# --- The Path to the local data------------
root_local="/Volumes/Extreme_SSD" # ".."
path="FE_tracks_retracked/FE_tracks_${STIME}_${BLOCK}_retracked"
#path_original="FE_tracks_original/FE_tracks_${STIME}/FE_tracks_${STIME}_$BLOCK"
# --------------------------------------

# TRAJECTORY PATHS
path_csv="$rootserver/$path"
path_csv_local="$root_local/$path"
POSITION_STR_FRONT="FE_${STIME}_tracks_${BLOCK}_front_retracked"
POSITION_STR_BACK="FE_${STIME}_tracks_${BLOCK}_back_retracked"
dir_front=$path_csv_local/$POSITION_STR_FRONT
dir_back=$path_csv_local/$POSITION_STR_BACK

# FEEDING PATHS
path_csv_feeding="$rootserver/FE_tracks_original/FE_tracks_${FEEDINGTIME}/FE_tracks_${FEEDINGTIME}_$BLOCK"
path_csv_feeding_local="$root_local/FE_tracks_original/FE_tracks_${FEEDINGTIME}/FE_tracks_${FEEDINGTIME}_$BLOCK"
POSITION_STR_FRONT_FEEDING="FE_${FEEDINGTIME}_tracks_${BLOCK}_front"
POSITION_STR_BACK_FEEDING="FE_${FEEDINGTIME}_tracks_${BLOCK}_back"

dir_feeding_front="${path_csv_feeding_local}/$POSITION_STR_FRONT_FEEDING"
dir_feeding_back="${path_csv_feeding_local}/$POSITION_STR_BACK_FEEDING"

export BLOCK
export STIME
export FEEDINGTIME
export rootserver
export root_local
export path_csv
export path_csv_local
export path_recordings
export POSITION_STR_BACK
export POSITION_STR_FRONT
export POSITION_STR_FRONT_FEEDING
export POSITION_STR_BACK_FEEDING
export dir_back
export dir_front
export dir_feeding_back
export dir_feeding_front