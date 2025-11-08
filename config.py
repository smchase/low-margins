"""Global configuration shared across transmitter and receiver."""

# Visual grid resolution used on screen and by the receiver lock.
VISUAL_GRID_SIZE = 32

# Default frames-per-second for automatic transmission.
TRANSMISSION_FPS = 2.0

# Default tensor dimensions for tests (match visual grid to fully utilize cells).
TEST_TENSOR_ROWS = VISUAL_GRID_SIZE
TEST_TENSOR_COLS = VISUAL_GRID_SIZE
