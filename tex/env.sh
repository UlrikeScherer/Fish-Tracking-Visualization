#!/bin/bash

BLOCK="block1"
STARTTIME="060000_"
rootserver="/Volumes/data/loopbio_data/FE_(fingerprint_experiment)_SepDec2021"
local_root="/Users/lukastaerk/fish"
path_recordings="$rootserver/FE_recordings/FE_recordings_$BLOCK/FE_${BLOCK}_recordings_*" 
path_csv_local="../$BLOCK" #"/Volumes/Extreme SSD/FE_tracks_$BLOCK"
path_csv="$rootserver/FE_tracks/FE_${STARTTIME}tracks_$BLOCK" #"$local_root/$BLOCK" 

export BLOCK
export STARTTIME
export rootserver
export path_csv
export path_csv_local
export path_recordings