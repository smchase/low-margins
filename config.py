"""
Shared configuration for transmitter and receiver
"""

# Display and grid settings
GRID_SIZE = 16  # 16x16 grid
DISPLAY_WIDTH = 1512
DISPLAY_HEIGHT = 982
CELL_WIDTH = DISPLAY_WIDTH // GRID_SIZE  # 94.5 pixels
CELL_HEIGHT = DISPLAY_HEIGHT // GRID_SIZE  # 61.4 pixels
GRID_BORDER_THICKNESS = 32  # Thickness of the bright green border in pixels
GRID_BORDER_COLOR = (0, 255, 128)  # Dedicated neon border color (BGR)

# Reference markers reserved for orientation (row, col, color index)
REFERENCE_MARKERS = [
    {"name": "TL", "position": (1, 1), "color": 1},  # Top-left red
    {"name": "TR", "position": (1, GRID_SIZE - 2), "color": 2},  # Top-right green
    {"name": "BL", "position": (GRID_SIZE - 2, 1), "color": 3},  # Bottom-left blue
    {"name": "BR", "position": (GRID_SIZE - 2, GRID_SIZE - 2), "color": 6},  # Bottom-right yellow
]
REFERENCE_CELL_COUNT = len(REFERENCE_MARKERS)

# Timing
TARGET_FPS = 20
FRAME_TIME_MS = 1000 // TARGET_FPS  # 50ms per frame

# Network
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 5000
RECEIVER_HOST = "localhost"  # Change to transmitter IP when running on different machine

# Message protocol
FRAME_HEADER = b'\xDE\xAD\xBE\xEF'  # Sync marker
ACTIVE_CELL_COUNT = GRID_SIZE * GRID_SIZE - REFERENCE_CELL_COUNT
FRAME_SIZE = ACTIVE_CELL_COUNT // 2  # bytes per frame (usable payload)

# Color palette - 16 colors for 4-bit encoding (nibbles 0-15)
# Format: (B, G, R) for OpenCV
# Note: Color 0 is white (default/empty cells) so it blends with white background
COLOR_PALETTE = {
    0: (255, 255, 255),     # 0: White
    1: (0, 0, 255),         # 1: Red
    2: (0, 255, 0),         # 2: Green
    3: (255, 0, 0),         # 3: Blue
    4: (255, 255, 0),       # 4: Cyan
    5: (255, 0, 255),       # 5: Magenta
    6: (0, 255, 255),       # 6: Yellow
    7: (0, 0, 0),           # 7: Black
    8: (128, 128, 255),     # 8: Light Red
    9: (128, 255, 128),     # 9: Light Green
    10: (255, 128, 128),    # 10: Light Blue
    11: (255, 255, 128),    # 11: Light Cyan
    12: (255, 128, 255),    # 12: Light Magenta
    13: (128, 255, 255),    # 13: Light Yellow
    14: (192, 192, 192),    # 14: Gray
    15: (128, 128, 128),    # 15: Dark Gray
}

# Reverse lookup for color classification
COLOR_NAMES = {
    0: "WHT", 1: "RED", 2: "GRN", 3: "BLU",
    4: "CYN", 5: "MAG", 6: "YEL", 7: "BLK",
    8: "LRED", 9: "LGRN", 10: "LBLU", 11: "LCYN",
    12: "LMAG", 13: "LYEL", 14: "GRY", 15: "DGRY"
}

print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print(f"Cell size: {CELL_WIDTH}x{CELL_HEIGHT} pixels")
print(f"Reference markers: {REFERENCE_CELL_COUNT}")
print(f"Frame rate: {TARGET_FPS} FPS ({FRAME_TIME_MS}ms per frame)")
print(f"Usable data per frame: {FRAME_SIZE} bytes")
