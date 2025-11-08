#!/usr/bin/env python3
"""
Visual Data Link - Simple version
Fixed message transmission between two MacBooks facing each other
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

from codec import codec
from codec_composite_camera import COLORS_8
from config import (
    VISUAL_GRID_SIZE,
    TRANSMISSION_FPS,
    TEST_TENSOR_ROWS,
    TEST_TENSOR_COLS,
)

class Receiver:
    """Receives and decodes grid patterns"""

    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.calibrated = False
        self.locked_corners = None
        self.warp_matrix = None
        self.warp_size = 600
        self.debug_warped = None  # Store warped image for debugging
        self.debug_gray = None  # Store grayscale for debugging
        self.border_ratio = 0.0  # No border - calibration locks to outer edge
        self.flip_horizontal = False  # Toggle for horizontal flip

    def calibrate(self, frame):
        """Detect and lock onto the grid"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find quadrilaterals
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    candidates.append({'contour': approx, 'area': area})

        if not candidates:
            return False

        # Use largest
        candidates.sort(key=lambda x: x['area'], reverse=True)
        best = candidates[0]

        # Store corners
        pts = best['contour'].reshape(4, 2).astype(np.float32)
        self.locked_corners = self._order_points(pts)

        # Compute warp matrix
        dst = np.array([
            [0, 0],
            [self.warp_size - 1, 0],
            [self.warp_size - 1, self.warp_size - 1],
            [0, self.warp_size - 1]
        ], dtype=np.float32)
        self.warp_matrix = cv2.getPerspectiveTransform(self.locked_corners, dst)

        self.calibrated = True
        print(f"\n✓ CALIBRATED!")
        return True

    def read(self, frame):
        """Read grid from locked position"""
        if not self.calibrated:
            return None

        # Apply perspective transform
        warped = cv2.warpPerspective(frame, self.warp_matrix, (self.warp_size, self.warp_size))

        # Convert to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Store grayscale for debugging
        self.debug_gray = gray.copy()

        # Use simple binary threshold (not adaptive) to preserve solid regions
        # Threshold at 127: values > 127 become white (255), values < 127 become black (0)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Create debug visualization showing the binary image
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Read grid (accounting for border)
        grid_start = int(self.warp_size * self.border_ratio)
        grid_end = int(self.warp_size * (1 - self.border_ratio))
        grid_size_px = grid_end - grid_start

        # Draw the grid region boundary
        cv2.rectangle(debug_img, (grid_start, grid_start), (grid_end, grid_end), (255, 0, 0), 2)

        pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        cell_size = grid_size_px / self.grid_size

        # Sample stats for debugging
        white_count = 0
        black_count = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y = int(grid_start + i * cell_size + cell_size / 2)
                x = int(grid_start + j * cell_size + cell_size / 2)

                sample_size = max(3, int(cell_size * 0.4))
                y1 = max(0, y - sample_size)
                y2 = min(self.warp_size, y + sample_size)
                x1 = max(0, x - sample_size)
                x2 = min(self.warp_size, x + sample_size)

                region = binary[y1:y2, x1:x2]
                white_ratio = np.sum(region == 255) / region.size
                pattern[i, j] = 1 if white_ratio > 0.5 else 0

                if pattern[i, j] == 1:
                    white_count += 1
                else:
                    black_count += 1

                # Draw sample region
                color = (0, 255, 0) if pattern[i, j] == 1 else (255, 0, 0)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 1)

        # Store debug image
        self.debug_warped = debug_img

        # Debug print
        total_cells = self.grid_size * self.grid_size
        print(f"\rRead: {white_count} white, {black_count} black (total {total_cells}) | Border: {self.border_ratio:.3f} ({int(self.warp_size * self.border_ratio)}px)", end='', flush=True)

        return pattern

    def decode(self, pattern):
        """Decode pattern to message"""
        if pattern is None:
            return None

        # Flip horizontally if enabled (for facing cameras)
        if self.flip_horizontal:
            pattern = np.fliplr(pattern)

        # Convert to bytes
        total_bits = self.grid_size * self.grid_size
        num_bytes = (total_bits + 7) // 8
        data = bytearray(num_bytes)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                byte_index = bit_index // 8
                bit_offset = bit_index % 8
                if pattern[i, j]:
                    data[byte_index] |= (1 << bit_offset)

        message_bytes = bytes(data)

        # Decode as UTF-8
        try:
            message = message_bytes.decode('utf-8', errors='strict').rstrip('\x00')
            return message, message_bytes
        except UnicodeDecodeError:
            return None, message_bytes

    def read_color_pattern(self, frame, grid_size, border_ratio=None):
        """Sample palette indices from the locked region."""
        if not self.calibrated or self.locked_corners is None:
            return None
        if border_ratio is None:
            border_ratio = self.border_ratio
        return read_color_pattern(frame, self.locked_corners, grid_size, border_ratio)

    def _order_points(self, pts):
        """Order points: TL, TR, BR, BL"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


def read_color_pattern(frame, locked_corners, grid_size, border_ratio=0.05, brightness_adjust=0):
    """Warp locked region and quantize each cell to the nearest palette color."""
    if locked_corners is None:
        return None

    src_pts = locked_corners.astype(np.float32)
    size = 512
    dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(frame, M, (size, size))
    
    # Apply brightness adjustment if needed
    if brightness_adjust != 0:
        warped = cv2.convertScaleAbs(warped, alpha=1.0, beta=brightness_adjust)

    cell_size = size / grid_size
    border = int(cell_size * border_ratio)
    color_indices = np.zeros((grid_size, grid_size), dtype=np.uint8)
    samples = []

    for i in range(grid_size):
        row_samples = []
        for j in range(grid_size):
            y1 = int(i * cell_size + border)
            y2 = int((i + 1) * cell_size - border)
            x1 = int(j * cell_size + border)
            x2 = int((j + 1) * cell_size - border)

            if y2 <= y1 or x2 <= x1:
                row_samples.append(np.array([0, 0, 0], dtype=np.float32))
                continue

            cell_region = warped[y1:y2, x1:x2]
            mean_color = cell_region.mean(axis=(0, 1))
            row_samples.append(mean_color)
        samples.append(row_samples)

    samples = np.array(samples, dtype=np.float32)
    palette = COLORS_8.astype(np.float32)
    diff = samples[:, :, None, :] - palette[None, None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    color_indices = np.argmin(dist, axis=2).astype(np.uint8)

    return color_indices


def is_calibration_pattern(color_indices):
    """Heuristic: calibration flashes only use black/white cells."""
    if color_indices is None:
        return True
    uniques = np.unique(color_indices)
    if uniques.size == 0:
        return True
    return np.all(np.isin(uniques, (0, 7)))


START_SIGNAL_COLOR_INDEX = 2  # Green in COLORS_8 palette


def is_start_signal(color_indices, min_ratio=0.95):
    if color_indices is None:
        return False
    total = color_indices.size
    if total == 0:
        return False
    count = np.count_nonzero(color_indices == START_SIGNAL_COLOR_INDEX)
    return count >= total * min_ratio


def count_changed_cells(a, b):
    if a is None or b is None:
        return 0
    return int(np.count_nonzero(a != b))


def classify_frame_phase(pattern):
    """Return 'lower' for multi-color frames, 'upper' for binary frames."""
    if pattern is None:
        return None

    total = pattern.size
    if total == 0:
        return None

    max_val = pattern.max()
    unique_vals = np.unique(pattern)
    
    # Upper phase should only have values 0 or 1
    # But due to camera noise, we might see some other values
    # Check if it's mostly 0s and 1s
    if max_val <= 1 and len(unique_vals) <= 2:
        return "upper"
    
    # Also check if vast majority are 0 or 1 (allowing for some noise)
    binary_count = np.sum((pattern == 0) | (pattern == 1))
    if binary_count >= pattern.size * 0.9:  # 90% or more are 0 or 1
        return "upper"
        
    return "lower"


def compute_slice_layout(rows, cols, grid_size):
    rows_per_slice = grid_size
    cols_per_slice = grid_size
    num_row_slices = int(math.ceil(rows / rows_per_slice))
    num_col_slices = int(math.ceil(cols / cols_per_slice))
    return rows_per_slice, cols_per_slice, num_row_slices, num_col_slices


class CodecTransmitterDisplay:
    """Full-screen transmitter that cycles codec grids using 8-color encoding."""

    def __init__(self, tensor, grid_size, fps, start_signal_seconds=1.0):
        self.tensor = tensor
        self.grid_size = grid_size
        self.fps = max(fps, 1e-6)
        self.rows, self.cols = tensor.shape
        self.codec = codec(self.rows, self.cols, min_val=0, max_val=15)
        self.encoded_grids = self.codec.encode(tensor)

        layout = compute_slice_layout(self.rows, self.cols, self.grid_size)
        (self.rows_per_slice, self.cols_per_slice,
         self.num_row_slices, self.num_col_slices) = layout
        self.total_frames = (self.encoded_grids.shape[0] *
                             self.num_row_slices *
                             self.num_col_slices * 2)

        self.cell_size = 24
        self.border = 40
        self.frame_interval = 1.0 / self.fps
        self.start_signal_seconds = max(0.0, start_signal_seconds)
        if self.start_signal_seconds <= 0:
            self.start_signal_frames = 0
        else:
            self.start_signal_frames = max(1, int(round(self.start_signal_seconds * self.fps)))
        self.remaining_start_frames = 0

        self.reset_state()

    def reset_state(self):
        import time as _time

        self.transmitting = False
        self.done = False
        self.current_grid_idx = 0
        self.current_row_slice = 0
        self.current_col_slice = 0
        self.current_color_frame = 0
        self.flash_counter = 0
        self.last_frame_time = _time.time()
        self.remaining_start_frames = 0

    def start_transmission(self):
        self.reset_state()
        self.transmitting = True
        self.done = False
        self.remaining_start_frames = self.start_signal_frames
        print(f"\n[TX] Starting auto-transmission at {self.fps:.2f} FPS "
              f"({self.total_frames} frames)")

    def update(self):
        if not self.transmitting or self.done:
            return
        import time as _time

        now = _time.time()
        if self.remaining_start_frames > 0:
            if now - self.last_frame_time >= self.frame_interval:
                self.last_frame_time = now
                self.remaining_start_frames -= 1
                if self.remaining_start_frames == 0:
                    print("[TX] Start signal complete, streaming data...")
            return

        if now - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = now
            self._advance()

    def _advance(self):
        self.current_color_frame += 1
        if self.current_color_frame >= 2:
            self.current_color_frame = 0
            self.current_col_slice += 1

            if self.current_col_slice >= self.num_col_slices:
                self.current_col_slice = 0
                self.current_row_slice += 1

                if self.current_row_slice >= self.num_row_slices:
                    self.current_row_slice = 0
                    self.current_grid_idx += 1

                    if self.current_grid_idx >= self.encoded_grids.shape[0]:
                        self.done = True
                        print("\n[TX] ✓ All frames transmitted. Showing DONE frame.")
                        return

    def _current_slice_values(self):
        full_grid = self.encoded_grids[self.current_grid_idx]
        start_row = self.current_row_slice * self.rows_per_slice
        end_row = min(start_row + self.rows_per_slice, full_grid.shape[0])
        start_col = self.current_col_slice * self.cols_per_slice
        end_col = min(start_col + self.cols_per_slice, full_grid.shape[1])

        slice_values = np.zeros((self.rows_per_slice, self.cols_per_slice), dtype=full_grid.dtype)
        window = full_grid[start_row:end_row, start_col:end_col]
        slice_values[:window.shape[0], :window.shape[1]] = window
        return slice_values

    def render(self):
        total_size = self.grid_size * self.cell_size
        grid = np.zeros((total_size, total_size, 3), dtype=np.uint8)

        if not self.transmitting:
            self.flash_counter += 1
            flash_state = (self.flash_counter // 10) % 3
            if flash_state == 0:
                grid[:] = [255, 255, 255]
            elif flash_state == 1:
                grid[:] = [0, 0, 0]
            else:
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        color = [255, 255, 255] if (i + j) % 2 else [0, 0, 0]
                        y1 = i * self.cell_size
                        y2 = (i + 1) * self.cell_size
                        x1 = j * self.cell_size
                        x2 = (j + 1) * self.cell_size
                        grid[y1:y2, x1:x2] = color
        elif self.remaining_start_frames > 0:
            grid[:] = COLORS_8[START_SIGNAL_COLOR_INDEX]
        elif self.done:
            grid[:] = [0, 255, 0]
        else:
            slice_values = self._current_slice_values()
            if self.current_color_frame == 0:
                color_indices = slice_values & 0x07  # Lower 3 bits: values 0-7
            else:
                color_indices = (slice_values >> 3) & 0x01  # Bit 3: values 0-1
                
            # Debug: print unique color indices every few frames (only if verbose)
            if hasattr(self, 'debug_verbose') and self.debug_verbose:
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 0
                    
                if self._debug_counter % 10 == 0:
                    unique_indices = np.unique(color_indices)
                    print(f"\n[TX DEBUG] Frame type: {'lower' if self.current_color_frame == 0 else 'upper'}, "
                          f"unique color indices: {unique_indices}")

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    color_idx = color_indices[i, j]
                    color = COLORS_8[color_idx]
                    y1 = i * self.cell_size
                    y2 = (i + 1) * self.cell_size
                    x1 = j * self.cell_size
                    x2 = (j + 1) * self.cell_size
                    grid[y1:y2, x1:x2] = color

        border = max(20, total_size // 20)
        final_size = total_size + 2 * border
        image = np.ones((final_size, final_size, 3), dtype=np.uint8) * 255
        image[border:-border, border:-border] = grid

        frame_width = 3
        image[border:border+frame_width, border:-border] = [0, 0, 0]
        image[-border-frame_width:-border, border:-border] = [0, 0, 0]
        image[border:-border, border:border+frame_width] = [0, 0, 0]
        image[border:-border, -border-frame_width:-border] = [0, 0, 0]

        phase_str = "lower(0-7)" if self.current_color_frame == 0 else "upper(0-1)"
        status = "CALIBRATION" if not self.transmitting else (
            "START SIGNAL" if self.remaining_start_frames > 0 else
            "DONE" if self.done else
            f"G{self.current_grid_idx+1}/{self.encoded_grids.shape[0]} "
            f"R{self.current_row_slice+1}/{self.num_row_slices} "
            f"C{self.current_col_slice+1}/{self.num_col_slices} "
            f"Phase: {phase_str}"
        )
        cv2.putText(image, f"TX MODE - {status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "Controls: P=start  R=reset  Q=quit",
                    (20, final_size - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return image

    def expected_frames(self):
        return self.total_frames


def load_tensor(args):
    if args.tensor:
        tensor = np.load(args.tensor)
        if tensor.shape != (args.rows, args.cols):
            raise ValueError(f"Tensor shape {tensor.shape} does not match --rows/--cols")
        if tensor.dtype != np.float16:
            tensor = tensor.astype(np.float16)
        return tensor

    rng = np.random.default_rng(args.seed)
    tensor = rng.standard_normal((args.rows, args.cols)).astype(np.float16)
    return tensor


def decode_captured_frames(color_frames, codec_obj, rows, cols, grid_size, force_alternating=False):
    rows_per_slice, cols_per_slice, num_row_slices, num_col_slices = compute_slice_layout(rows, cols, grid_size)
    expected = codec_obj.grids_needed() * num_row_slices * num_col_slices * 2
    if len(color_frames) < expected:
        raise ValueError(f"Need {expected} frames, captured {len(color_frames)}")

    # Classify frames by phase
    frame_phases = []
    for frame in color_frames:
        phase = classify_frame_phase(frame)
        frame_phases.append(phase)
    
    print(f"[RX] Frame phases: {frame_phases}")
    
    # Try to decode assuming frames are in order
    frame_iter = iter(color_frames)
    frame_idx = 0
    grids = []
    try:
        for grid_idx in range(codec_obj.grids_needed()):
            row_bands = []
            for row_idx in range(num_row_slices):
                col_blocks = []
                for col_idx in range(num_col_slices):
                    # Get next two frames - try to get one of each phase
                    frame1 = next(frame_iter)
                    phase1 = classify_frame_phase(frame1)
                    frame2 = next(frame_iter)
                    phase2 = classify_frame_phase(frame2)
                    
                    if force_alternating:
                        # Force alternating pattern: even frames are lower, odd are upper
                        lower, upper = frame1, frame2
                        print(f"[RX] Grid {grid_idx+1}: Forcing alternating pattern (frame1=lower, frame2=upper)")
                    else:
                        # For robustness, determine phase by looking at value distribution
                        # Lower phase should have more diverse values (0-7)
                        # Upper phase should be mostly 0s and 1s
                        
                        unique1 = np.unique(frame1)
                        unique2 = np.unique(frame2)
                        
                        # Count how many cells are 0 or 1 in each frame
                        binary_ratio1 = np.sum((frame1 == 0) | (frame1 == 1)) / frame1.size
                        binary_ratio2 = np.sum((frame2 == 0) | (frame2 == 1)) / frame2.size
                        
                        # The frame with higher binary ratio is likely the upper phase
                        if binary_ratio2 > binary_ratio1:
                            lower, upper = frame1, frame2
                            print(f"[RX] Grid {grid_idx+1}: frame1 is lower (binary ratio {binary_ratio1:.2f}), "
                                  f"frame2 is upper (binary ratio {binary_ratio2:.2f})")
                        else:
                            lower, upper = frame2, frame1
                            print(f"[RX] Grid {grid_idx+1}: frame2 is lower (binary ratio {binary_ratio2:.2f}), "
                                  f"frame1 is upper (binary ratio {binary_ratio1:.2f})")
                    
                    lower_bits = (lower & 0x07).astype(np.uint8)
                    upper_bit = (upper & 0x01).astype(np.uint8)
                    slice_values = lower_bits | (upper_bit << 3)
                    col_blocks.append(slice_values)
                    
                    # Debug: show sample values and unique values in each frame
                    if grid_idx == 0 and row_idx == 0 and col_idx == 0:  # First slice of first grid
                        lower_unique = np.unique(lower)
                        upper_unique = np.unique(upper)
                        print(f"[RX] Sample decode: lower unique values: {lower_unique}, upper unique values: {upper_unique}")
                        print(f"[RX] Sample decode: lower[0,0]={lower[0,0]} -> {lower_bits[0,0]}, "
                              f"upper[0,0]={upper[0,0]} -> {upper_bit[0,0]}, "
                              f"combined={slice_values[0,0]}")
                row_bands.append(np.hstack(col_blocks))
            full_grid = np.vstack(row_bands)
            grids.append(full_grid[:rows, :cols])
    except StopIteration as exc:
        raise ValueError("Incomplete frame sequence for decoding") from exc

    stacked_grids = np.stack(grids, axis=0)
    
    # Debug: show the range of values in the decoded grids
    print(f"[RX] Decoded grids shape: {stacked_grids.shape}")
    for i, grid in enumerate(stacked_grids):
        unique_vals = np.unique(grid)
        print(f"[RX] Grid {i+1} unique values: {unique_vals}, min={grid.min()}, max={grid.max()}")
    
    # Ensure values are within valid range [0, 15]
    if stacked_grids.max() > 15:
        print(f"[RX] WARNING: Grid values exceed 15 (max={stacked_grids.max()}), clamping to valid range")
        stacked_grids = np.clip(stacked_grids, 0, 15)
    
    return stacked_grids


def run_tx(args):
    tensor = load_tensor(args)
    tx = CodecTransmitterDisplay(tensor, args.grid_size, args.fps, args.start_signal_seconds)

    cv2.namedWindow("Codec TX", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Codec TX", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n" + "=" * 70)
    print("TX MODE - CODEC VISUAL LINK")
    print("=" * 70)
    print(f"Tensor: {tensor.shape}, dtype={tensor.dtype}")
    print(f"Visual grid: {args.grid_size}x{args.grid_size}, FPS: {args.fps}")
    print(f"Total frames to send: {tx.expected_frames()}")
    if tx.start_signal_frames > 0:
        print(f"Start signal: solid green for {tx.start_signal_frames} frame(s) "
              f"({tx.start_signal_seconds:.2f}s) before data begins")
    print("Controls: 'p' to start, 'r' to reset, 'q' to quit")
    print("Use this window full-screen facing the receiver camera.\n")

    while True:
        tx.update()
        frame = tx.render()
        cv2.imshow("Codec TX", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            tx.start_transmission()
        elif key == ord('r'):
            tx.reset_state()

    cv2.destroyWindow("Codec TX")


def run_rx(args):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Set buffer size to 1 to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    rx = Receiver(grid_size=args.grid_size)
    c = codec(args.rows, args.cols, min_val=0, max_val=15)
    layout = compute_slice_layout(args.rows, args.cols, args.grid_size)
    total_frames_needed = c.grids_needed() * layout[2] * layout[3] * 2

    captured_frames = []
    capture_active = False
    last_pattern = None
    pending_pattern = None
    pending_phase = None
    pending_consistency = 0
    phase_seen_time = None
    capture_threshold = max(0.0, min(1.0, args.capture_threshold))
    capture_tolerance = max(0.0, min(1.0, args.capture_tolerance))
    capture_stability = max(1, args.capture_stability)
    capture_timeout = max(0.0, args.capture_timeout)
    expected_phase = "lower"
    last_capture_time = 0  # Track when we last captured a frame
    min_capture_interval = args.capture_interval  # Minimum seconds between captures

    print("\n" + "=" * 70)
    print("RX MODE - CODEC VISUAL LINK")
    print("=" * 70)
    print(f"Expecting {total_frames_needed} frames "
          f"({c.grids_needed()} grids × {layout[2]} row slices × {layout[3]} col slices × 2 color frames)")
    print("Controls:")
    print("  c   - Calibrate (lock onto TX grid)")
    print("  p   - Start capture (after calibration)")
    print("  r   - Reset captured frames")
    print("  f   - Toggle horizontal flip")
    print("  +/- - Adjust border offset")
    print("  q   - Quit")
    print("=" * 70)
    print(f"Capture threshold: {capture_threshold*100:.1f}% change, "
          f"tolerance {capture_tolerance*100:.1f}%, stability {capture_stability} frame(s)")
    print(f"Capture interval: {min_capture_interval:.1f}s minimum between frames")

    # Add frame counter for debugging
    frame_count = 0
    last_fps_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    while True:
        # Flush camera buffer - grab multiple frames and use only the last one
        # This helps ensure we get the most recent frame
        for _ in range(2):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break
            
        frame_count += 1
        fps_frame_count += 1
        
        # Calculate FPS every second
        now = time.time()
        if now - last_fps_time >= 1.0:
            current_fps = fps_frame_count / (now - last_fps_time)
            fps_frame_count = 0
            last_fps_time = now

        display = frame.copy()
        status_text = "NOT CALIBRATED - Press 'c'"
        status_color = (0, 0, 255)

        if rx.calibrated and rx.locked_corners is not None:
            corners = rx.locked_corners.astype(np.int32)
            cv2.polylines(display, [corners], True, (0, 255, 0), 3)
            status_text = "LOCKED"
            status_color = (0, 255, 0)

            if capture_active:
                pattern = rx.read_color_pattern(frame, args.grid_size)
                if pattern is None:
                    continue
                if is_calibration_pattern(pattern) or is_start_signal(pattern):
                    continue

                phase = classify_frame_phase(pattern)
                if phase is None:
                    continue

                total_cells = pattern.size
                min_changed = max(1, int(total_cells * capture_threshold))
                tolerance_cells = max(1, int(total_cells * capture_tolerance))
                diff_last = total_cells if last_pattern is None else count_changed_cells(pattern, last_pattern)
                
                # For debugging, let's capture all frames with any significant change
                if args.debug_capture_all:
                    min_changed = max(1, int(total_cells * 0.01))  # Only 1% change needed
                elif phase == "upper":
                    min_changed = max(1, int(total_cells * 0.05))  # Only 5% change needed for upper frames
                
                if diff_last < min_changed:
                    continue

                now = time.time()

                # We detected significant change, so this is likely a new frame
                # Only print debug for phase changes or unusual patterns when requested
                if args.debug_capture_all:
                    unique_vals = np.unique(pattern)
                    if phase == "upper" or len(unique_vals) > 4 or frame_count % 30 == 0:
                        print(f"\n[RX DEBUG] Phase: {phase}, unique values: {unique_vals}, "
                              f"max={pattern.max()}, diff={diff_last}/{total_cells}")

                if phase_seen_time is None:
                    phase_seen_time = now

                if pending_pattern is None or pending_phase != phase:
                    pending_pattern = pattern.copy()
                    pending_phase = phase
                    pending_consistency = 1
                    continue

                diff_pending = count_changed_cells(pattern, pending_pattern)
                if diff_pending <= tolerance_cells:
                    pending_consistency += 1
                    pending_pattern = pattern.copy()
                else:
                    pending_pattern = pattern.copy()
                    pending_consistency = 1
                    continue

                stable_enough = pending_consistency >= capture_stability
                timeout_elapsed = (
                    capture_timeout > 0 and
                    phase_seen_time is not None and
                    now - phase_seen_time >= capture_timeout
                )

                if stable_enough or timeout_elapsed:
                    # Check if enough time has passed since last capture
                    time_since_last = now - last_capture_time if last_capture_time > 0 else min_capture_interval
                    if time_since_last < min_capture_interval:
                        # Too soon, keep accumulating stability
                        continue
                        
                    last_pattern = pending_pattern.copy()
                    captured_frames.append(last_pattern.copy())
                    pending_pattern = None
                    pending_phase = None
                    pending_consistency = 0
                    phase_seen_time = None
                    last_capture_time = now  # Update last capture time
                    total_captured = len(captured_frames)
                    print(f"\n[RX] Captured frame {total_captured}/{total_frames_needed} "
                          f"(phase {phase}, diff vs last {diff_last}, interval {time_since_last:.2f}s)")
                    
                    # Don't enforce strict phase alternation - just track what we captured

                    if total_captured >= total_frames_needed:
                        print("\n[RX] Frames collected - decoding...")
                        try:
                            grids = decode_captured_frames(captured_frames, c, args.rows, args.cols, args.grid_size, 
                                                          force_alternating=args.force_alternating)
                            decoded_tensor = c.decode(grids)
                            
                            # Check for NaN values
                            nan_count = np.isnan(decoded_tensor).sum()
                            if nan_count > 0:
                                print(f"[RX] ⚠️  Warning: Decoded tensor contains {nan_count} NaN values!")
                                # Find which grids might be problematic
                                for i, grid in enumerate(grids):
                                    unique_vals = np.unique(grid)
                                    expected = set(range(16))
                                    missing = expected - set(unique_vals)
                                    if missing:
                                        print(f"[RX] Grid {i+1} is missing values: {sorted(missing)}")
                                        if 0 in missing:
                                            print("[RX] Tip: Missing value 0 (black) - try increasing border with '+' key")
                                        if 7 in missing:
                                            print("[RX] Tip: Missing value 7 (white) - might be clipping, try decreasing border with '-' key")
                        else:
                            print("[RX] ✓ Decode complete!")
                            
                            print(f"[RX] Decoded tensor: {decoded_tensor.shape}, dtype={decoded_tensor.dtype}")
                            print(f"[RX] Sample row: {decoded_tensor[0, :5]}")
                            
                            # Show statistics
                            valid_values = decoded_tensor[~np.isnan(decoded_tensor)]
                            if len(valid_values) > 0:
                                print(f"[RX] Valid values: min={valid_values.min():.3f}, "
                                      f"max={valid_values.max():.3f}, mean={valid_values.mean():.3f}")
                            
                            if args.save_decoded:
                                np.save(args.save_decoded, decoded_tensor)
                                print(f"[RX] Saved tensor to {args.save_decoded}")
                        except Exception as err:
                            print(f"[RX] ✗ Decode failed: {err}")
                        capture_active = False
                        phase_seen_time = None
                        expected_phase = "lower"
            else:
                capture_active = False
                last_pattern = None
                pending_pattern = None
                pending_phase = None
                pending_consistency = 0
                phase_seen_time = None
                expected_phase = "lower"

        if capture_active:
            info = f"CAPTURING {len(captured_frames)}/{total_frames_needed}"
            cv2.putText(display, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Show FPS and frame count
        fps_text = f"FPS: {current_fps:.1f} | Frame: {frame_count}"
        cv2.putText(display, fps_text, (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Codec RX", display)

        if rx.debug_gray is not None:
            cv2.imshow('Debug: Grayscale', rx.debug_gray)
        if rx.debug_warped is not None:
            cv2.imshow('Debug: Binary + Grid', rx.debug_warped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if rx.calibrated:
                rx.calibrated = False
                rx.locked_corners = None
                capture_active = False
                captured_frames.clear()
                last_pattern = None
                pending_pattern = None
                pending_consistency = 0
                pending_phase = None
                phase_seen_time = None
                expected_phase = "lower"
                last_capture_time = 0  # Reset capture timing
                print("\n[RX] Calibration reset")
            else:
                if rx.calibrate(frame):
                    print("[RX] ✓ Calibration locked")
                else:
                    print("[RX] ✗ Calibration failed")
        elif key == ord('p'):
            if not rx.calibrated:
                print("\n[RX] ✗ Calibrate before starting capture")
            else:
                capture_active = True
                captured_frames.clear()
                last_pattern = None
                pending_pattern = None
                pending_consistency = 0
                pending_phase = None
                phase_seen_time = None
                expected_phase = "lower"
                last_capture_time = 0  # Reset capture timing
                print("\n[RX] Capture started - waiting for frames...")
        elif key == ord('r'):
            captured_frames.clear()
            last_pattern = None
            pending_pattern = None
            pending_consistency = 0
            pending_phase = None
            phase_seen_time = None
            expected_phase = "lower"
            last_capture_time = 0  # Reset capture timing
            capture_active = False
            print("\n[RX] Captured frames cleared")
        elif key == ord('f'):
            rx.flip_horizontal = not rx.flip_horizontal
            print(f"\n[RX] Flip horizontal: {'ON' if rx.flip_horizontal else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            rx.border_ratio += 0.01
            print(f"\n[RX] Border: {rx.border_ratio:.3f}")
        elif key == ord('-') or key == ord('_'):
            rx.border_ratio = max(0.0, rx.border_ratio - 0.01)
            print(f"\n[RX] Border: {rx.border_ratio:.3f}")

    cap.release()
    cv2.destroyAllWindows()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Codec Visual Data Link")
    parser.add_argument('--mode', required=True, choices=['tx', 'rx'], help='Run as transmitter or receiver')
    parser.add_argument('--grid-size', type=int, default=VISUAL_GRID_SIZE, help='Visual grid resolution')
    parser.add_argument('--fps', type=float, default=TRANSMISSION_FPS, help='TX frames per second')
    parser.add_argument('--start-signal-seconds', type=float, default=1.0,
                        help='Duration to show solid green start signal before frames')
    parser.add_argument('--rows', type=int, default=TEST_TENSOR_ROWS, help='Tensor rows')
    parser.add_argument('--cols', type=int, default=TEST_TENSOR_COLS, help='Tensor cols')
    parser.add_argument('--tensor', type=str, help='Path to .npy tensor for TX mode')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random tensor (TX)')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index for RX mode')
    parser.add_argument('--save-decoded', type=str, help='Path to save decoded tensor (RX mode)')
    parser.add_argument('--capture-threshold', type=float, default=0.05,
                        help='Fraction of cells that must change before counting a new frame (RX)')
    parser.add_argument('--capture-tolerance', type=float, default=0.02,
                        help='Fraction of cells allowed to differ between stability samples (RX)')
    parser.add_argument('--capture-stability', type=int, default=2,
                        help='Number of consecutive stable samples before recording a frame (RX)')
    parser.add_argument('--capture-timeout', type=float, default=0.4,
                        help='Seconds to wait before accepting latest sample even if stability not reached (RX)')
    parser.add_argument('--debug-capture-all', action='store_true',
                        help='Capture all frames for debugging (ignores change threshold)')
    parser.add_argument('--force-alternating', action='store_true',
                        help='Force alternating lower/upper phase decoding (RX)')
    parser.add_argument('--capture-interval', type=float, default=0.3,
                        help='Minimum seconds between frame captures (RX)')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        print("rows/cols must be positive")
        sys.exit(1)

    if args.mode == 'tx':
        run_tx(args)
    else:
        run_rx(args)


if __name__ == "__main__":
    main()
