"""Global configuration shared across transmitter and receiver."""

import numpy as np

# Visual grid resolution used on screen and by the receiver lock.
VISUAL_GRID_SIZE = 32

# Default frames-per-second for automatic transmission.
TRANSMISSION_FPS = 2.0

# Default tensor dimensions for tests (match visual grid to fully utilize cells).
TEST_TENSOR_ROWS = VISUAL_GRID_SIZE
TEST_TENSOR_COLS = VISUAL_GRID_SIZE

# 5-color palette for codec transmission (in BGR format for OpenCV)
COLORS_5 = np.array([
    [255, 255, 255],   # 0: White
    [0, 0, 0],         # 1: Black
    [0, 0, 255],       # 2: Red (BGR)
    [255, 0, 0],       # 3: Blue (BGR)
    [0, 255, 0],       # 4: Green
], dtype=np.uint8)

# Color names for debugging
COLOR_NAMES = ["White", "Black", "Red", "Blue", "Green"]

# Codec configuration for 5-color system
CODEC_MIN_VAL = 0
CODEC_MAX_VAL = 4  # 5 colors: [0, 1, 2, 3, 4]
