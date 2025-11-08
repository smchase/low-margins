"""
Shared configuration for transmitter and receiver
"""

# Display and grid settings
GRID_SIZE = 64  # 64x64 grid
DISPLAY_WIDTH = 1512
DISPLAY_HEIGHT = 982
CELL_WIDTH = DISPLAY_WIDTH // GRID_SIZE  # ~23.6 pixels
CELL_HEIGHT = DISPLAY_HEIGHT // GRID_SIZE  # ~15.3 pixels

# Timing
TARGET_FPS = 20
FRAME_TIME_MS = 1000 // TARGET_FPS  # 50ms per frame

# Network
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 5000
RECEIVER_HOST = "localhost"  # Change to transmitter IP when running on different machine

# Message protocol
FRAME_HEADER = b'\xDE\xAD\xBE\xEF'  # Sync marker
FRAME_SIZE = 2048  # bytes per frame (4,096 cells × 4 bits)

# Color palette - 16 highly distinct colors optimized for camera detection
# Format: (B, G, R) for OpenCV
COLOR_PALETTE = {
    0: (0, 0, 0),           # Black
    1: (0, 0, 255),         # Red
    2: (0, 255, 0),         # Green
    3: (255, 0, 0),         # Blue
    4: (255, 255, 0),       # Cyan
    5: (255, 0, 255),       # Magenta
    6: (0, 255, 255),       # Yellow
    7: (255, 255, 255),     # White
    8: (0, 0, 128),         # Dark Red
    9: (0, 128, 0),         # Dark Green
    10: (128, 0, 0),        # Dark Blue
    11: (0, 165, 255),      # Orange
    12: (128, 0, 128),      # Purple
    13: (192, 192, 255),    # Pink
    14: (0, 255, 127),      # Spring Green
    15: (128, 128, 0),      # Teal
}

# Reverse lookup for color classification
COLOR_NAMES = {
    0: "BLK", 1: "RED", 2: "GRN", 3: "BLU", 4: "CYN", 5: "MAG",
    6: "YEL", 7: "WHT", 8: "DRD", 9: "DGN", 10: "DBL", 11: "ORN",
    12: "PUR", 13: "PNK", 14: "SGR", 15: "TEA"
}

print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print(f"Cell size: {CELL_WIDTH}x{CELL_HEIGHT} pixels")
print(f"Frame rate: {TARGET_FPS} FPS ({FRAME_TIME_MS}ms per frame)")
print(f"Data per frame: {FRAME_SIZE} bytes")
