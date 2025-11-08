"""
Composite Camera - Real camera with TX grid overlaid on top.

Captures from real camera and renders the transmitter's grid
on top of the camera feed (smaller, centered).

This allows testing without needing two devices:
- One camera sees both the real world AND the TX grid
- RX can calibrate and decode from the composite image
"""

import cv2
import numpy as np
from visual_data_link import Transmitter, FIXED_MESSAGE


class CompositeCamera:
    """
    Real camera with TX grid rendered on top (centered, smaller).

    Implements cv2.VideoCapture interface.
    """

    def __init__(self, grid_size=16, cell_size=40):
        """
        Initialize composite camera.

        Args:
            grid_size: Grid size for TX (default: 16)
            cell_size: Cell size in pixels (default: 40 - smaller for overlay)
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

        self.grid_size = grid_size
        self.cell_size = cell_size

        # Create transmitter for grid rendering
        self.tx = Transmitter(grid_size=grid_size, cell_size=cell_size)
        self.tx.set_message(FIXED_MESSAGE)

        # Start in calibration mode (flashing)
        self.tx.calibration_mode = True
        self.calibration_frames = 0

        # Try to set camera resolution, but detect actual resolution on first read
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Actual width and height will be detected on first read
        self.width = None
        self.height = None
        self.first_read = True

        print(f"[CompositeCamera] Initialized with {grid_size}x{grid_size} grid, {cell_size}px cells")
        print(f"[CompositeCamera] Starting in calibration mode (flashing)...")

    def read(self):
        """
        Read frame from real camera and composite TX grid on top.

        Returns:
            (success, composite_frame) where composite_frame has TX grid overlaid
        """
        ret, frame = self.camera.read()
        if not ret:
            return False, None

        # Detect actual camera resolution on first read
        if self.first_read:
            self.height, self.width = frame.shape[:2]
            self.first_read = False
            print(f"[CompositeCamera] Detected camera resolution: {self.width}x{self.height}")

        # Render TX grid
        grid_img = self.tx.render()  # Grayscale image
        grid_bgr = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)

        # Get grid dimensions
        grid_h, grid_w = grid_bgr.shape[:2]

        # If grid is too large for frame, resize it
        if grid_h > self.height or grid_w > self.width:
            scale = min(self.height / grid_h, self.width / grid_w) * 0.9  # 90% of available space
            new_h, new_w = int(grid_h * scale), int(grid_w * scale)
            grid_bgr = cv2.resize(grid_bgr, (new_w, new_h))
            grid_h, grid_w = new_h, new_w
            print(f"[CompositeCamera] Grid too large, scaled to {grid_w}x{grid_h}")

        # Calculate position to center grid on frame
        x = (self.width - grid_w) // 2
        y = (self.height - grid_h) // 2

        # Ensure within bounds (should be safe now, but keep as safety check)
        x = max(0, min(x, self.width - grid_w))
        y = max(0, min(y, self.height - grid_h))

        # Create composite image
        composite = frame.copy()

        # Place grid directly on top (fully opaque, no blending)
        composite[y:y+grid_h, x:x+grid_w] = grid_bgr

        # Auto-switch from calibration to normal mode
        self.calibration_frames += 1
        if self.calibration_frames == 90 and self.tx.calibration_mode:  # ~3 seconds at 30 FPS
            self.tx.calibration_mode = False
            print("\n[CompositeCamera] Switching to normal mode (transmitting message)...")

        return True, composite

    def set(self, prop_id, value):
        """Set camera property (pass through to real camera)."""
        self.camera.set(prop_id, value)

    def isOpened(self):
        """Check if camera is open."""
        return self.camera.isOpened()

    def release(self):
        """Release camera."""
        self.camera.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.release()
