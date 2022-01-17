#!/bin/bash

BLOCK="block1"
rootserver="/Volumes/data/loopbio_data/1_FE_(fingerprint_experiment)_SepDec2021"
path_recordings="$rootserver/FE_recordings/FE_recordings_$BLOCK"
path_csv="$rootserver/FE_tracks/FE_tracks_$BLOCK"

export BLOCK, rootserver, path_csv, path_recordings