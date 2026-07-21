#!/usr/bin/python3

import os

# File and Path
DATA_RAW_PATH="../../Datasets/IndustrialSurfaceDetects/metal_nut"
DATA_GOOD_PATH=os.path.join( DATA_RAW_PATH, "good" )
DATA_BAD_PATH=os.path.join( DATA_RAW_PATH, "bad" )
DATA_PROCESSED_PATH="data/processed/surface_defects"
DATA_IGNORED_PATH="data/ignored"

# Launch Parameters
IMAGE_LOAD_NAME=None #"test_scratch_006" # None : load all images in DATA_BAD_PATH
GOOD_IMAGE_BASE_NAME="test_good_000"
IMAGE_EXTENSION=".png"
CAN_OVERRIDE=True
SHOW_IMAGE=True