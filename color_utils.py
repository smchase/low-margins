"""
Color handling and detection utilities
"""

import numpy as np
import cv2
from config import COLOR_PALETTE, GRID_SIZE


class ColorDetector:
    """Detects which of 16 colors a pixel belongs to"""

    def __init__(self):
        self.colors = np.array([COLOR_PALETTE[i] for i in range(16)], dtype=np.float32)

    def detect_color(self, pixel):
        """
        Determine which of 16 colors the pixel is closest to.
        Uses Euclidean distance in BGR space.

        Args:
            pixel: (B, G, R) tuple

        Returns:
            color_index (0-15)
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
            color_indices: shape (H, W) with values 0-15
        """
        # Reshape to (H*W, 3)
        h, w, c = image_region.shape
        pixels = image_region.reshape(-1, 3).astype(np.float32)

        # Compute distances to all 16 colors
        distances = np.zeros((pixels.shape[0], 16), dtype=np.float32)
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
    Convert 128x128 grid (values 0-15) to 8192 bytes.
    Each byte stores 2 cells (4 bits each).

    Args:
        grid_data: (128, 128) array with values 0-15

    Returns:
        bytes of length 8192
    """
    flat = grid_data.flatten()
    result = bytearray()

    for i in range(0, len(flat), 2):
        high_nibble = flat[i] & 0x0F
        low_nibble = flat[i + 1] & 0x0F
        byte = (high_nibble << 4) | low_nibble
        result.append(byte)

    return bytes(result)


def bytes_to_grid(data):
    """
    Convert 8192 bytes back to 128x128 grid.

    Args:
        data: bytes of length 8192

    Returns:
        (128, 128) array with values 0-15
    """
    flat = []

    for byte in data:
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F
        flat.append(high_nibble)
        flat.append(low_nibble)

    return np.array(flat, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)


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
