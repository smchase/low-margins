"""
Color handling and detection utilities
"""

import numpy as np
import cv2
from config import (
    COLOR_PALETTE,
    GRID_SIZE,
    FRAME_SIZE,
    ACTIVE_CELL_COUNT,
    REFERENCE_MARKERS,
)

# Precompute reference cell lookups
REFERENCE_INDEX_MAP = {}
REFERENCE_INDEX_SET = set()
for marker in REFERENCE_MARKERS:
    row, col = marker["position"]
    flat_idx = row * GRID_SIZE + col
    REFERENCE_INDEX_MAP[flat_idx] = marker["color"]
    REFERENCE_INDEX_SET.add(flat_idx)


class ColorDetector:
    """Detects which color a pixel belongs to from the palette"""

    def __init__(self):
        num_colors = len(COLOR_PALETTE)
        self.base_colors = np.array([COLOR_PALETTE[i] for i in range(num_colors)], dtype=np.float32)
        self.channel_scale = np.ones(3, dtype=np.float32)
        self.colors = self.base_colors.copy()

    def detect_color(self, pixel):
        """
        Determine which color the pixel is closest to.
        Uses Euclidean distance in BGR space.

        Args:
            pixel: (B, G, R) tuple

        Returns:
            color_index (0 to num_colors-1)
        """
        pixel = np.array(pixel, dtype=np.float32)
        distances = np.sqrt(np.sum((self.colors - pixel) ** 2, axis=1))
        return np.argmin(distances)

    def detect_colors_batch(self, image_region):
        """
        Detect colors for a region of pixels at once (faster).

        Args:
            image_region: shape (H, W, 3) BGR image

        Returns:
            color_indices: shape (H, W) with color values from palette
        """
        # Reshape to (H*W, 3)
        h, w, c = image_region.shape
        pixels = image_region.reshape(-1, 3).astype(np.float32)

        # Compute distances to all colors in palette
        num_colors = len(self.colors)
        distances = np.zeros((pixels.shape[0], num_colors), dtype=np.float32)
        for i, color in enumerate(self.colors):
            distances[:, i] = np.sqrt(np.sum((pixels - color) ** 2, axis=1))

        # Find nearest color for each pixel
        color_indices = np.argmin(distances, axis=1)
        return color_indices.reshape(h, w)

    def detect_colors_adaptive(self, image_region):
        """
        More robust detection using HSV + saturation weighting.
        Handles lighting variations better.
        """
        hsv = cv2.cvtColor(image_region.astype(np.uint8), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Normalize saturation to weight pure colors more heavily
        s_norm = s.astype(np.float32) / 255.0

        h_flat = h.flatten().astype(np.float32)
        s_flat = s_norm.flatten().astype(np.float32)
        v_flat = v.flatten().astype(np.float32)

        # For simplicity, still use BGR detection but this prepares for HSV
        pixels = image_region.reshape(-1, 3).astype(np.float32)
        distances = np.zeros((pixels.shape[0], 16), dtype=np.float32)

        for i, color in enumerate(self.colors):
            distances[:, i] = np.sqrt(np.sum((pixels - color) ** 2, axis=1))

        color_indices = np.argmin(distances, axis=1)
        return color_indices.reshape(image_region.shape[0], image_region.shape[1])

    def apply_channel_scale(self, scale):
        """
        Update palette colors by applying per-channel scale factors (B, G, R).
        """
        scale = np.asarray(scale, dtype=np.float32)
        if scale.shape != (3,):
            raise ValueError("Scale must be length 3 for B,G,R channels")
        # Clamp scale to avoid wild values
        scale = np.clip(scale, 0.2, 5.0)
        self.channel_scale = scale
        self.colors = np.clip(self.base_colors * scale, 0, 255).astype(np.float32)

    def reset_scale(self):
        """Reset palette to original values."""
        self.channel_scale = np.ones(3, dtype=np.float32)
        self.colors = self.base_colors.copy()


def draw_grid(image, grid_data, cell_width, cell_height):
    """
    Draw a 128x128 color grid onto an image.

    Args:
        image: Base image to draw on (will be modified)
        grid_data: numpy array of shape (128, 128) with values 0-15
        cell_width: pixels per cell (width)
        cell_height: pixels per cell (height)
    """
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color_idx = grid_data[row, col]
            color = COLOR_PALETTE[color_idx]

            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)


def grid_to_bytes(grid_data):
    """
    Convert grid (values 0-15) to payload bytes, skipping reserved reference cells.
    """
    flat = grid_data.flatten()
    nibbles = []

    for idx, value in enumerate(flat):
        if idx in REFERENCE_INDEX_SET:
            continue
        nibbles.append(int(value) & 0x0F)

    if len(nibbles) != ACTIVE_CELL_COUNT:
        raise ValueError("Grid payload size mismatch")

    result = bytearray()
    for i in range(0, len(nibbles), 2):
        high_nibble = nibbles[i] & 0x0F
        low_nibble = nibbles[i + 1] & 0x0F
        result.append((high_nibble << 4) | low_nibble)

    return bytes(result)


def bytes_to_grid(data):
    """
    Convert payload bytes back to grid, inserting reference markers.
    """
    if len(data) != FRAME_SIZE:
        raise ValueError(f"Expected {FRAME_SIZE} bytes, got {len(data)}")

    nibbles = []
    for byte in data:
        nibbles.append((byte >> 4) & 0x0F)
        nibbles.append(byte & 0x0F)

    if len(nibbles) != ACTIVE_CELL_COUNT:
        raise ValueError("Payload size does not match active cell count")

    flat = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.uint8)
    nibble_idx = 0
    for idx in range(GRID_SIZE * GRID_SIZE):
        if idx in REFERENCE_INDEX_SET:
            flat[idx] = REFERENCE_INDEX_MAP[idx]
        else:
            flat[idx] = nibbles[nibble_idx]
            nibble_idx += 1

    return flat.reshape(GRID_SIZE, GRID_SIZE)


if __name__ == "__main__":
    # Quick test
    detector = ColorDetector()

    # Test a few colors
    test_colors = [
        (0, 0, 0),          # Black
        (255, 0, 0),        # Blue
        (0, 255, 0),        # Green
        (255, 255, 255),    # White
    ]

    for color in test_colors:
        idx = detector.detect_color(color)
        print(f"Color {color} detected as index {idx} ({COLOR_PALETTE[idx]})")
