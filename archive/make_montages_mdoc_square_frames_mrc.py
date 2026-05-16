#!/bin/python
# ARCHIVED — original monolithic script, superseded by the package in square_aperture_montage/
# Kept for reference only. Do not use for new work.

import sys
import glob
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.signal as signal
from tqdm import tqdm
import pyserialem as pyserialem
import os
import subprocess
import tifffile
import imagecodecs

skip_crop = True
write_output = True

dose_per_tilt = 3.0
frames_number = 4
square_size = 3840
frames_dir = "frames_mrc/"
split_frames_dir = "frames/split"
frames_cropped_dir = "cropped_frames"
frames_cropped_stk_dir = f'{frames_cropped_dir}/stk'
mdocs_dir = "Mdoc"
average_dir = "frames/average"
average_cropped_dir = "frames/average/cropped"
average_blend_dir = "frames/average/blend"
clipped_dir = "clipped_mics"
tiltstack_dir = "tiltstacks/"
frames_output_dir = "output_frames"

clip_x = 11664
clip_y = 11664

ts_list = [ "VLP3x3_p01_ts_002", "VLP3x3_p01_ts_003", "VLP3x3_p01_ts_004", "VLP3x3_p01_ts_005", "VLP3x3_p01_ts_006", "VLP3x3_p01_ts_007", "VLP3x3_p01_ts_008", "VLP3x3_p01_ts_009", "VLP3x3_p02_ts_002", "VLP3x3_p02_ts_003", "VLP3x3_p02_ts_004", "VLP3x3_p02_ts_005", "VLP3x3_p02_ts_006", "VLP3x3_p02_ts_007", "VLP3x3_p02_ts_008", "VLP3x3_p02_ts_009", "VLP3x3_p03_ts_002", "VLP3x3_p03_ts_003", "VLP3x3_p03_ts_004", "VLP3x3_p03_ts_005", "VLP3x3_p03_ts_006", "VLP3x3_p03_ts_007", "VLP3x3_p03_ts_008", "VLP3x3_p03_ts_009", "VLP3x3_p04_ts_002", "VLP3x3_p04_ts_003", "VLP3x3_p04_ts_004", "VLP3x3_p04_ts_005", "VLP3x3_p04_ts_006", "VLP3x3_p04_ts_007", "VLP3x3_p04_ts_008", "VLP3x3_p04_ts_009", "VLP3x3_p05_ts_002", "VLP3x3_p05_ts_003", "VLP3x3_p05_ts_004", "VLP3x3_p05_ts_005", "VLP3x3_p05_ts_006", "VLP3x3_p05_ts_007", "VLP3x3_p05_ts_008", "VLP3x3_p05_ts_009" ]
