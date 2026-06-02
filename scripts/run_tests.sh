#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fishproviz
python -m unittest discover tests
