"""
Codec Composite Camera - Real camera with codec-encoded tensor grid overlay.

Transmits an FP16 tensor using 8-color encoding:
- Codec encodes tensor into 4 grids of values [0-15]
- Each grid defaults to VISUAL_GRID_SIZE×VISUAL_GRID_SIZE (configurable via config.py)
- Each value [0-15] is transmitted as 2 color frames (8 colors = 3 bits/cell)
- Total: 4 grids × 2 frames = 8 frames
"""

import cv2
import numpy as np
from codec import codec
from config import VISUAL_GRID_SIZE, TRANSMISSION_FPS

# 8 distinct colors (corners of RGB cube) - in BGR format for OpenCV
COLORS_8 = np.array([
    [0, 0, 0],         # 0: Black
    [0, 0, 255],       # 1: Red (BGR)
    [0, 255, 0],       # 2: Green
    [0, 255, 255],     # 3: Yellow (BGR)
    [255, 0, 0],       # 4: Blue (BGR)
    [255, 0, 255],     # 5: Magenta (BGR)
    [255, 255, 0],     # 6: Cyan (BGR)
    [255, 255, 255],   # 7: White
], dtype=np.uint8)


class CodecCompositeCamera:
    """
    Real camera with codec-encoded tensor grid overlaid.

    Uses 8-color encoding to transmit values [0-15]:
    - Grid size: configurable (defaults to VISUAL_GRID_SIZE×VISUAL_GRID_SIZE cells)
    - 8 colors per cell (3 bits per cell)
    - 2 color frames per grid: lower 3 bits (0-7), upper bit + padding (0-1)
    - Total: 4 grids × 2 frames = 8 frames
    """

    def __init__(self, tensor, cell_size=30, grid_size=VISUAL_GRID_SIZE, fps=TRANSMISSION_FPS):
        """
        Initialize codec composite camera.

        Args:
            tensor: FP16 tensor to transmit
            cell_size: Pixels per cell (default: 30)
            grid_size: Visual grid resolution (default: VISUAL_GRID_SIZE)
            fps: Frames per second for auto-transmission (default: TRANSMISSION_FPS)
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

        self.base_cell_size = cell_size
        self.cell_size = cell_size
        self.tensor = tensor
        self.fps = fps

        # Get tensor dimensions
        rows, cols = tensor.shape

        # Encode tensor with codec
        self.codec = codec(rows, cols, min_val=0, max_val=15)
        self.encoded_grids = self.codec.encode(tensor)
        print(f"[CodecCompositeCamera] Encoded {rows}x{cols} tensor into {self.encoded_grids.shape[0]} grids")

        # Visual grid size (fixed for camera overlay)
        self.grid_size = grid_size

        # Calculate spatial slices needed to fit codec grid into visual grid
        self.rows_per_slice = grid_size
        self.cols_per_slice = grid_size
        self.num_row_slices = int(np.ceil(rows / self.rows_per_slice))
        self.num_col_slices = int(np.ceil(cols / self.cols_per_slice))
        self.num_slices = self.num_row_slices * self.num_col_slices

        total_frames = self.encoded_grids.shape[0] * self.num_slices * 2
        print(f"[CodecCompositeCamera] Visual grid: {self.grid_size}x{self.grid_size} cells")
        print(f"[CodecCompositeCamera] Codec grid {rows}x{cols} → "
              f"{self.num_row_slices} row slices × {self.num_col_slices} col slices "
              f"({self.num_slices} total slices)")
        print(f"[CodecCompositeCamera] Using 8-color encoding: 2 color frames per slice")
        print(f"[CodecCompositeCamera] Total: {self.encoded_grids.shape[0]} grids × {self.num_slices} slices × 2 frames = {total_frames} frames")
        print(f"[CodecCompositeCamera] Auto-transmission at {fps} FPS (~{total_frames / max(fps, 1e-6):.1f} seconds)")

        # Transmission state
        self.current_grid_idx = 0
        self.current_row_slice = 0  # Which row slice (0 to num_row_slices-1)
        self.current_col_slice = 0  # Which column slice (0 to num_col_slices-1)
        self.current_color_frame = 0  # 0-1 for the 2 color frames
        self.calibration_mode = True
        self.flash_counter = 0
        self.transmitting = False
        self.done = False  # True when all frames sent

        # Auto-advance timing
        import time
        self.last_frame_time = time.time()
        self.frame_interval = 1.0 / fps

        # Camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.width = None
        self.height = None
        self.first_read = True

        print(f"[CodecCompositeCamera] Ready. Press 'p' to start transmission.")

    def start_transmission(self):
        """Start auto-transmitting frames at configured FPS"""
        import time
        self.transmitting = True
        self.current_grid_idx = 0
        self.current_row_slice = 0
        self.current_col_slice = 0
        self.current_color_frame = 0
        self.done = False
        self.last_frame_time = time.time()
        num_grids = self.encoded_grids.shape[0]
        total_frames = num_grids * self.num_slices * 2
        print(f"\n[CodecCompositeCamera] Starting auto-transmission at {self.fps} FPS ({total_frames} frames)")
        print(f"[CodecCompositeCamera] Grid 1/{num_grids}, Slice R1/{self.num_row_slices}, C1/{self.num_col_slices}, Frame 0/2")

    def render_grid(self):
        """Render current grid as RGB image (showing current color frame)"""
        total_size = self.grid_size * self.cell_size

        # Create RGB image
        grid = np.zeros((total_size, total_size, 3), dtype=np.uint8)

        if not self.transmitting:
            # Calibration mode - flash pattern (grayscale)
            self.flash_counter += 1
            flash_state = (self.flash_counter // 10) % 3

            if flash_state == 0:
                grid[:] = [255, 255, 255]  # White
            elif flash_state == 1:
                grid[:] = [0, 0, 0]  # Black
            else:
                # Checkerboard
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        color = [255, 255, 255] if (i + j) % 2 else [0, 0, 0]
                        y1 = i * self.cell_size
                        y2 = (i + 1) * self.cell_size
                        x1 = j * self.cell_size
                        x2 = (j + 1) * self.cell_size
                        grid[y1:y2, x1:x2] = color
        elif self.done:
            # Done frame - solid green (indicates transmission complete)
            grid[:] = [0, 255, 0]  # Green
        else:
            # Normal mode - show current color frame of current slice of current grid
            full_grid = self.encoded_grids[self.current_grid_idx]  # e.g., 748×64 array of values [0-15]

            # Extract current slice
            start_row = self.current_row_slice * self.rows_per_slice
            end_row = min(start_row + self.rows_per_slice, full_grid.shape[0])
            start_col = self.current_col_slice * self.cols_per_slice
            end_col = min(start_col + self.cols_per_slice, full_grid.shape[1])

            slice_values = np.zeros((self.rows_per_slice, self.cols_per_slice), dtype=full_grid.dtype)
            window = full_grid[start_row:end_row, start_col:end_col]
            slice_values[:window.shape[0], :window.shape[1]] = window

            if self.current_color_frame == 0:
                # Frame 0: lower 3 bits (values 0-7) → 8 colors
                color_indices = slice_values & 0x07  # Extract lower 3 bits
            else:
                # Frame 1: upper bit (value 0-1) → use first 2 colors (black/red)
                color_indices = (slice_values >> 3) & 0x01  # Extract bit 3

            # Render using color palette
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    color_idx = color_indices[i, j]
                    color = COLORS_8[color_idx]
                    y1 = i * self.cell_size
                    y2 = (i + 1) * self.cell_size
                    x1 = j * self.cell_size
                    x2 = (j + 1) * self.cell_size
                    grid[y1:y2, x1:x2] = color

        # Add white border
        border = max(20, total_size // 20)
        final_size = total_size + 2 * border
        image = np.ones((final_size, final_size, 3), dtype=np.uint8) * 255
        image[border:-border, border:-border] = grid

        # Add black frame
        frame_width = 2
        image[border:border+frame_width, border:-border] = [0, 0, 0]
        image[-border-frame_width:-border, border:-border] = [0, 0, 0]
        image[border:-border, border:border+frame_width] = [0, 0, 0]
        image[border:-border, -border-frame_width:-border] = [0, 0, 0]

        return image

    def read(self):
        """Read frame from camera and composite grid on top"""
        ret, frame = self.camera.read()
        if not ret:
            return False, None

        # Detect actual camera resolution on first read
        if self.first_read:
            self.height, self.width = frame.shape[:2]
            self.first_read = False
            print(f"[CodecCompositeCamera] Detected camera resolution: {self.width}x{self.height}")
            self._update_cell_size()
        else:
            self.height, self.width = frame.shape[:2]
            self._update_cell_size()

        # Auto-advance frames if transmitting
        if self.transmitting and not self.done:
            import time
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                self.last_frame_time = current_time
                self._advance_frame()

        # Always render grid (so it's visible for calibration)
        # Render current grid (already in BGR format)
        grid_bgr = self.render_grid()

        # Get grid dimensions
        grid_h, grid_w = grid_bgr.shape[:2]

        # Scale grid to fill available frame (while preserving borders)
        scale = min(self.height / grid_h, self.width / grid_w)
        if scale != 1.0:
            if scale > 1.0:
                scale *= 0.98  # leave a small padding from edges
            new_h, new_w = int(grid_h * scale), int(grid_w * scale)
            new_h = max(1, new_h)
            new_w = max(1, new_w)
            if new_h != grid_h or new_w != grid_w:
                grid_bgr = cv2.resize(grid_bgr, (new_w, new_h))
                grid_h, grid_w = new_h, new_w

        # Calculate centered position
        x = (self.width - grid_w) // 2
        y = (self.height - grid_h) // 2
        x = max(0, min(x, self.width - grid_w))
        y = max(0, min(y, self.height - grid_h))

        # Composite image
        composite = frame.copy()
        composite[y:y+grid_h, x:x+grid_w] = grid_bgr

        return True, composite

    def _advance_frame(self):
        """Auto-advance to next color frame"""
        self.current_color_frame += 1
        num_grids = self.encoded_grids.shape[0]

        if self.current_color_frame >= 2:
            # Move to next slice
            self.current_color_frame = 0
            self.current_col_slice += 1

            if self.current_col_slice >= self.num_col_slices:
                self.current_col_slice = 0
                self.current_row_slice += 1

                if self.current_row_slice >= self.num_row_slices:
                    # Move to next grid
                    self.current_row_slice = 0
                    self.current_grid_idx += 1

                    if self.current_grid_idx >= num_grids:
                        # All frames sent!
                        self.done = True
                        print(f"\n[CodecCompositeCamera] ✓ All frames transmitted! Showing DONE frame (solid green)")
                        return
                    else:
                        print(f"[CodecCompositeCamera] Grid {self.current_grid_idx+1}/{num_grids}, "
                              f"Slice R1/{self.num_row_slices}, C1/{self.num_col_slices}, Frame 0/2")
                else:
                    print(f"[CodecCompositeCamera] Grid {self.current_grid_idx+1}/{num_grids}, "
                          f"Slice R{self.current_row_slice+1}/{self.num_row_slices}, C1/{self.num_col_slices}, Frame 0/2")
            else:
                print(f"[CodecCompositeCamera] Grid {self.current_grid_idx+1}/{num_grids}, "
                      f"Slice R{self.current_row_slice+1}/{self.num_row_slices}, C{self.current_col_slice+1}/{self.num_col_slices}, Frame 0/2")
        else:
            print(f"[CodecCompositeCamera] Grid {self.current_grid_idx+1}/{num_grids}, "
                  f"Slice R{self.current_row_slice+1}/{self.num_row_slices}, C{self.current_col_slice+1}/{self.num_col_slices}, "
                  f"Frame {self.current_color_frame}/2")

    def get_current_frame_id(self):
        """Get unique ID for current frame (for change detection)"""
        if self.done:
            return -1  # Done frame
        if not self.transmitting:
            return -2  # Calibration
        # Unique ID: grid * (slices * 2) + slice * 2 + color_frame
        slice_index = (self.current_row_slice * self.num_col_slices + self.current_col_slice)
        return (self.current_grid_idx * self.num_slices * 2 +
                slice_index * 2 +
                self.current_color_frame)

    def _update_cell_size(self):
        """Auto-adjust pixels per grid cell so overlay fills most of the frame."""
        if self.width is None or self.height is None:
            return

        target_span = int(min(self.width, self.height) * 0.98)
        if target_span <= 0:
            return

        # Estimate best cell size so final grid (including border) nearly fills target_span
        estimated_cell = max(1, target_span // max(1, self.grid_size))

        def final_size(cell):
            total = self.grid_size * cell
            border = max(20, total // 20)
            return total + 2 * border

        cell = estimated_cell
        while cell > 1 and final_size(cell) > target_span:
            cell -= 1

        if cell != self.cell_size:
            self.cell_size = cell
            print(f"[CodecCompositeCamera] Auto cell size set to {self.cell_size}px (target span {target_span}px)")

    def set(self, prop_id, value):
        """Set camera property"""
        self.camera.set(prop_id, value)

    def isOpened(self):
        """Check if camera is open"""
        return self.camera.isOpened()

    def release(self):
        """Release camera"""
        self.camera.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
