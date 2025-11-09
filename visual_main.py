#!/usr/bin/env python3
"""
Visual Data Link - 5-Color System
Fixed message transmission between two MacBooks facing each other

Uses 5 distinct colors (White, Black, Red, Blue, Green) for codec transmission.
- Each color represents a value 0-4
- FP16 tensors encoded into 7 grids (vs 4 grids with 8-color system)
- Simpler, more reliable color detection
- No phase splitting - direct color-to-value mapping
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np
import torch

from codec import codec
from config import (
    VISUAL_GRID_SIZE,
    TRANSMISSION_FPS,
    TEST_TENSOR_ROWS,
    TEST_TENSOR_COLS,
    COLORS_5,
    COLOR_NAMES,
    CODEC_MIN_VAL,
    CODEC_MAX_VAL,
)

class Receiver:
    """Receives and decodes grid patterns"""

    def __init__(self, grid_size=16, brightness_adjust=0):
        self.grid_size = grid_size
        self.calibrated = False
        self.locked_corners = None
        self.warp_matrix = None
        self.warp_size = 600
        self.debug_warped = None  # Store warped image for debugging
        self.debug_gray = None  # Store grayscale for debugging
        self.debug_color_samples = None  # Store color sampling overlay
        self.debug_sample_grid = None  # Store sample grid visualization
        self.border_ratio = 0.0  # No border - calibration locks to outer edge
        self.flip_horizontal = False  # Toggle for horizontal flip
        self.show_sample_regions = False
        self.sample_polys = None
        self.brightness_adjust = brightness_adjust  # Brightness adjustment for color sampling
        self.debug_color_sampling = False  # Print color sampling debug info

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

    def read_color_pattern(self, frame, grid_size, border_ratio=None, expected_indices=None):
        """Sample palette indices from the locked region."""
        if not self.calibrated or self.locked_corners is None:
            return None
        if border_ratio is None:
            border_ratio = self.border_ratio

        # Determine if we need samples for debugging
        return_samples = self.debug_color_sampling

        pattern_result = read_color_pattern(
            frame,
            self.locked_corners,
            grid_size,
            border_ratio,
            brightness_adjust=self.brightness_adjust,
            return_debug=self.show_sample_regions,
            return_samples=return_samples,
        )

        if self.show_sample_regions and return_samples:
            pattern, debug_img, sample_polys, sample_grid, samples = pattern_result
            self.debug_color_samples = debug_img
            self.sample_polys = sample_polys
            self.debug_sample_grid = sample_grid
            if self.debug_color_sampling:
                print_color_sampling_debug(samples, pattern, expected_indices)
        elif self.show_sample_regions:
            pattern, debug_img, sample_polys, sample_grid = pattern_result
            self.debug_color_samples = debug_img
            self.sample_polys = sample_polys
            self.debug_sample_grid = sample_grid
        elif return_samples:
            pattern, samples = pattern_result
            if self.debug_color_sampling:
                print_color_sampling_debug(samples, pattern, expected_indices)
        else:
            pattern = pattern_result
            self.debug_color_samples = None
            self.sample_polys = None
            self.debug_sample_grid = None

        # Flip horizontally if enabled (for facing cameras)
        if pattern is not None and self.flip_horizontal:
            pattern = np.fliplr(pattern)
        return pattern

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


def read_color_pattern(frame, locked_corners, grid_size, border_ratio=0.05,
                       brightness_adjust=0, return_debug=False, return_samples=False):
    """Warp locked region and quantize each cell to the nearest palette color.

    When return_debug is True, also returns:
      - the warped image with sample rectangles drawn
      - the polygons (in the original camera space) for overlaying
      - a simplified sample-grid visualization
    When return_samples is True, also returns the raw RGB samples (for debugging)
    """
    if locked_corners is None:
        return None

    src_pts = locked_corners.astype(np.float32)
    size = 512
    dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(frame, M, (size, size))
    debug_img = None
    sample_polys = None
    sample_grid = None
    if return_debug:
        debug_img = warped.copy()
        sample_polys = []
        sample_grid = np.zeros_like(warped)
        sample_grid[:] = 32
        M_inv = np.linalg.inv(M).astype(np.float32)

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
            if return_debug:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.rectangle(sample_grid, (x1, y1), (x2, y2), (0, 255, 0), 1)
                rect = np.array([[[x1, y1],
                                   [x2, y1],
                                   [x2, y2],
                                   [x1, y2]]], dtype=np.float32)
                orig_rect = cv2.perspectiveTransform(rect, M_inv)
                sample_polys.append(orig_rect.reshape(-1, 2))
        samples.append(row_samples)

    samples = np.array(samples, dtype=np.float32)
    palette = COLORS_5.astype(np.float32)
    diff = samples[:, :, None, :] - palette[None, None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    color_indices = np.argmin(dist, axis=2).astype(np.uint8)

    if return_debug:
        sample_polys = [poly.astype(np.int32) for poly in sample_polys]
        if return_samples:
            return color_indices, debug_img, sample_polys, sample_grid, samples
        return color_indices, debug_img, sample_polys, sample_grid
    if return_samples:
        return color_indices, samples
    return color_indices


def print_color_sampling_debug(samples, color_indices, expected_indices=None):
    """Print debug info about color sampling quality."""
    print("\n[COLOR DEBUG] Sampled RGB vs Palette Colors:")
    print("Palette reference (5-color system):")
    for idx, (color, name) in enumerate(zip(COLORS_5, COLOR_NAMES)):
        print(f"  {idx}: {name:6s} RGB{tuple(color)} (B={color[0]}, G={color[1]}, R={color[2]})")

    print("\nFirst few sampled cells:")
    flat_samples = samples.reshape(-1, 3)
    flat_indices = color_indices.flatten()
    if expected_indices is not None:
        flat_expected = expected_indices.flatten()

    for i in range(min(16, len(flat_samples))):
        sample_rgb = flat_samples[i]
        matched_idx = flat_indices[i]
        palette_rgb = COLORS_5[matched_idx]
        color_name = COLOR_NAMES[matched_idx]
        distance = np.sqrt(np.sum((sample_rgb - palette_rgb) ** 2))

        status = "✓"
        if expected_indices is not None and matched_idx != flat_expected[i]:
            status = f"✗ (expected {flat_expected[i]})"

        print(f"  Cell {i}: Sampled RGB({sample_rgb[0]:.1f},{sample_rgb[1]:.1f},{sample_rgb[2]:.1f}) "
              f"→ matched palette {matched_idx}:{color_name} RGB{tuple(palette_rgb)} "
              f"(dist={distance:.1f}) {status}")


def is_calibration_pattern(color_indices):
    """Heuristic: calibration patterns alternate between all-black, all-white, or checkerboard."""
    if color_indices is None:
        return True
    uniques = np.unique(color_indices)
    if uniques.size == 0:
        return True
    
    # In 5-color system: 0=White, 1=Black
    # Only consider it calibration if it's strictly black/white AND has a specific pattern
    if set(uniques) == {0, 1}:  # Only white and black
        # Check if it's a checkerboard pattern
        rows, cols = color_indices.shape
        is_checkerboard = True
        for i in range(rows):
            for j in range(cols):
                expected = 0 if (i + j) % 2 == 0 else 1
                if color_indices[i, j] != expected and color_indices[i, j] != (1 - expected):
                    is_checkerboard = False
                    break
            if not is_checkerboard:
                break
        return is_checkerboard
    
    # All one color patterns during calibration are only black or white
    if len(uniques) == 1 and uniques[0] in [0, 1]:
        # But during data transmission, we might have all-black or all-white frames that are valid data
        # So we can't just reject all single-color frames
        return False  # Consider single-color frames as data, not calibration
    
    return False


START_SIGNAL_COLOR_INDEX = 4  # Green in COLORS_5 palette


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
    """With 5-color system, no phases - all frames are direct color mappings."""
    # This function is kept for compatibility but is no longer needed
    # All frames in 5-color system are equivalent (no lower/upper distinction)
    return "data"


def compute_slice_layout(rows, cols, grid_size):
    rows_per_slice = grid_size
    cols_per_slice = grid_size
    num_row_slices = int(math.ceil(rows / rows_per_slice))
    num_col_slices = int(math.ceil(cols / cols_per_slice))
    return rows_per_slice, cols_per_slice, num_row_slices, num_col_slices


class CodecTransmitterDisplay:
    """Full-screen transmitter that cycles codec grids using 5-color encoding."""

    def __init__(self, tensor, grid_size, fps, start_signal_seconds=1.0, stream_count=1):
        self.tensor = tensor
        self.grid_size = grid_size
        self.fps = max(fps, 1e-6)
        self.rows, self.cols = tensor.shape
        self.codec = codec(self.rows, self.cols, min_val=CODEC_MIN_VAL, max_val=CODEC_MAX_VAL)
        self.encoded_grids = self.codec.encode(tensor)
        self.stream_count = max(1, stream_count)  # Number of times to transmit
        self.current_stream_iteration = 0

        layout = compute_slice_layout(self.rows, self.cols, self.grid_size)
        (self.rows_per_slice, self.cols_per_slice,
         self.num_row_slices, self.num_col_slices) = layout
        # 5-color system: no phases, just direct grid display
        self.frames_per_iteration = (self.encoded_grids.shape[0] *
                                      self.num_row_slices *
                                      self.num_col_slices)
        self.total_frames = self.frames_per_iteration * self.stream_count

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
        self.current_stream_iteration = 0
        self.flash_counter = 0
        self.last_frame_time = _time.time()
        self.remaining_start_frames = 0
        self.transmission_start_time = None

    def start_transmission(self):
        import time as _time
        self.reset_state()
        self.transmitting = True
        self.done = False
        self.remaining_start_frames = self.start_signal_frames
        self.transmission_start_time = _time.time()
        
        tensor_bytes = self.rows * self.cols * 2  # FP16 = 2 bytes per value
        total_bytes = tensor_bytes * self.stream_count
        
        print(f"\n[TX] Starting auto-transmission at {self.fps:.2f} FPS")
        if self.stream_count > 1:
            print(f"[TX] Stream mode: {self.stream_count} tensors × {tensor_bytes} bytes = {total_bytes} bytes ({total_bytes/1024:.2f} KB)")
            print(f"[TX] Each tensor: {self.rows}×{self.cols} = {self.rows*self.cols} FP16 values")
            print(f"[TX] Frames per tensor: {self.frames_per_iteration}, Total frames: {self.total_frames}")
        else:
            print(f"[TX] Single tensor: {tensor_bytes} bytes ({tensor_bytes/1024:.2f} KB)")
            print(f"[TX] Total frames: {self.total_frames}")
        print(f"[TX] First data frame will be Grid 1 (5-color system: colors 0-4)")

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
                    print(f"[TX] Starting with Grid 1")
            return

        if now - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = now
            self._advance()

    def _advance(self):
        # 5-color system: no phases, just advance through slices and grids
        self.current_col_slice += 1

        if self.current_col_slice >= self.num_col_slices:
            self.current_col_slice = 0
            self.current_row_slice += 1

            if self.current_row_slice >= self.num_row_slices:
                self.current_row_slice = 0
                self.current_grid_idx += 1

                if self.current_grid_idx >= self.encoded_grids.shape[0]:
                    # Finished one iteration
                    self.current_stream_iteration += 1
                    
                    if self.current_stream_iteration >= self.stream_count:
                        # All iterations complete
                        self.done = True
                        import time as _time
                        elapsed = _time.time() - self.transmission_start_time
                        tensor_bytes = self.rows * self.cols * 2
                        total_bytes = tensor_bytes * self.stream_count
                        throughput_kbps = (total_bytes / 1024) / elapsed if elapsed > 0 else 0
                        
                        print(f"\n[TX] ✓ All transmissions complete!")
                        if self.stream_count > 1:
                            print(f"[TX] Transmitted: {self.stream_count} tensors × {tensor_bytes} bytes = {total_bytes} bytes ({total_bytes/1024:.2f} KB)")
                            print(f"[TX] Time: {elapsed:.2f} seconds")
                            print(f"[TX] Throughput: {throughput_kbps:.2f} KB/s ({throughput_kbps*8:.2f} Kbps)")
                            print(f"[TX] Avg per tensor: {elapsed/self.stream_count:.2f} seconds")
                        return
                    else:
                        # Start next iteration
                        self.current_grid_idx = 0
                        import time as _time
                        elapsed = _time.time() - self.transmission_start_time
                        print(f"\n[TX STREAM] Iteration {self.current_stream_iteration}/{self.stream_count} complete!")
                        print(f"[TX STREAM] Elapsed so far: {elapsed:.2f}s, starting next tensor...")

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
            grid[:] = COLORS_5[START_SIGNAL_COLOR_INDEX]
        elif self.done:
            grid[:] = [0, 255, 0]
        else:
            # 5-color system: directly map grid values (0-4) to colors
            slice_values = self._current_slice_values()
            color_indices = slice_values  # Direct mapping, no bit extraction
                
            # Debug: print unique color indices every few frames (only if verbose)
            if hasattr(self, 'debug_verbose') and self.debug_verbose:
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 0
                    
                if self._debug_counter % 10 == 0:
                    unique_indices = np.unique(color_indices)
                    print(f"\n[TX DEBUG] Unique color indices (0-4): {unique_indices}")

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    color_idx = color_indices[i, j]
                    color = COLORS_5[color_idx]
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

        return image

    def expected_frames(self):
        return self.total_frames


def create_test_pattern_tensor(rows, cols):
    """Create a test pattern tensor with known values for verification."""
    tensor = np.zeros((rows, cols), dtype=np.float16)
    
    # Pattern 1: Diagonal gradient
    for i in range(rows):
        for j in range(cols):
            # Create a gradient that changes along diagonals
            value = ((i + j) % 256) / 128.0 - 1.0  # Range -1 to 1
            tensor[i, j] = value
    
    # Pattern 2: Add some specific known values at corners and center
    if rows > 0 and cols > 0:
        tensor[0, 0] = 0.5      # Top-left
        tensor[0, -1] = -0.5    # Top-right
        tensor[-1, 0] = 0.25    # Bottom-left
        tensor[-1, -1] = -0.25  # Bottom-right
        
        # Center values
        center_r, center_c = rows // 2, cols // 2
        tensor[center_r, center_c] = 1.0
        if center_r > 0 and center_c > 0:
            tensor[center_r-1, center_c-1] = -1.0
            
    return tensor.astype(np.float16)


def create_color_test_pattern_tensor(rows, cols):
    """Create a simple test pattern that will successfully roundtrip through codec.

    Instead of trying to force specific grid values, we create a simple FP16 pattern
    and let the codec handle it naturally. The key is that it should roundtrip perfectly.
    """
    # Create a simple gradient pattern that will roundtrip through the codec
    # Use values that are well-represented in float16
    tensor = np.zeros((rows, cols), dtype=np.float16)
    for i in range(rows):
        for j in range(cols):
            # Create a repeating pattern with well-represented float16 values
            # These will exercise all colors during transmission
            idx = (i * cols + j) % 16
            if idx < 4:
                tensor[i, j] = 0.25 * idx  # 0.0, 0.25, 0.5, 0.75
            elif idx < 8:
                tensor[i, j] = -0.25 * (idx - 4)  # 0.0, -0.25, -0.5, -0.75
            elif idx < 12:
                tensor[i, j] = 0.125 * (idx - 8)  # 0.0, 0.125, 0.25, 0.375
            else:
                tensor[i, j] = 1.0 + 0.25 * (idx - 12)  # 1.0, 1.25, 1.5, 1.75

    return tensor.astype(np.float16)


def verify_color_test_pattern(decoded_tensor, rows, cols):
    """Verify that the decoded tensor matches the color test pattern via bit-exact comparison."""
    expected = create_color_test_pattern_tensor(rows, cols)

    print("\n[RX VERIFY] Verifying color test pattern (bit-exact comparison)...")
    print(f"[RX VERIFY] Expected shape: {expected.shape}, Decoded shape: {decoded_tensor.shape}")
    print(f"[RX VERIFY] Expected range: [{expected.min():.3f}, {expected.max():.3f}]")
    print(f"[RX VERIFY] Decoded range:  [{decoded_tensor.min():.3f}, {decoded_tensor.max():.3f}]")
    print(f"[RX VERIFY] Expected first row: {expected[0, :min(10, cols)]}")
    print(f"[RX VERIFY] Decoded first row:  {decoded_tensor[0, :min(10, cols)]}")
    print(f"[RX VERIFY] Expected last row:  {expected[-1, :min(10, cols)]}")
    print(f"[RX VERIFY] Decoded last row:   {decoded_tensor[-1, :min(10, cols)]}")
    
    # Check if it's just horizontally flipped
    decoded_flipped_h = np.fliplr(decoded_tensor)
    matches_flipped_h = np.sum(expected.view(np.uint16) == decoded_flipped_h.view(np.uint16))
    print(f"[RX VERIFY] Matches if horizontally flipped: {matches_flipped_h}/{expected.size} ({matches_flipped_h/expected.size*100:.1f}%)")
    
    # Check if it's vertically flipped
    decoded_flipped_v = np.flipud(decoded_tensor)
    matches_flipped_v = np.sum(expected.view(np.uint16) == decoded_flipped_v.view(np.uint16))
    print(f"[RX VERIFY] Matches if vertically flipped: {matches_flipped_v}/{expected.size} ({matches_flipped_v/expected.size*100:.1f}%)")
    
    # Check if it's rotated 180
    decoded_rot180 = np.rot90(decoded_tensor, 2)
    matches_rot180 = np.sum(expected.view(np.uint16) == decoded_rot180.view(np.uint16))
    print(f"[RX VERIFY] Matches if rotated 180°: {matches_rot180}/{expected.size} ({matches_rot180/expected.size*100:.1f}%)")

    # Do bit-exact comparison by viewing as uint16
    expected_u16 = expected.view(np.uint16)
    decoded_u16 = decoded_tensor.view(np.uint16)
    
    matches = np.sum(expected_u16 == decoded_u16)
    total = expected_u16.size
    accuracy = matches / total * 100 if total > 0 else 0
    
    print(f"[RX] Bit-exact matches: {matches}/{total} cells ({accuracy:.1f}%)")

    if accuracy == 100.0:
        print("[RX] ✅ Color test pattern verified correctly! Perfect roundtrip.")
        return True
    elif accuracy > 99.0:
        print("[RX] ✅ Color test pattern nearly perfect! (acceptable variance)")
        # Show a few mismatches
        mismatches = np.where(expected_u16 != decoded_u16)
        if len(mismatches[0]) > 0:
            print(f"[RX] Sample mismatches (showing up to 5):")
            for idx in range(min(5, len(mismatches[0]))):
                i, j = mismatches[0][idx], mismatches[1][idx]
                exp_val = expected[i, j]
                dec_val = decoded_tensor[i, j]
                exp_bits = expected_u16[i, j]
                dec_bits = decoded_u16[i, j]
                print(f"  [{i},{j}]: decoded={dec_val:.6f} (0x{dec_bits:04x}), "
                      f"expected={exp_val:.6f} (0x{exp_bits:04x})")
        return True
    else:
        print("[RX] ⚠️  Color test pattern did not match - possible transmission errors")
        # Show some mismatches
        mismatches = np.where(expected_u16 != decoded_u16)
        if len(mismatches[0]) > 0:
            print(f"[RX] Sample mismatches (showing up to 10):")
            for idx in range(min(10, len(mismatches[0]))):
                i, j = mismatches[0][idx], mismatches[1][idx]
                exp_val = expected[i, j]
                dec_val = decoded_tensor[i, j]
                print(f"  [{i},{j}]: decoded={dec_val:.3f}, expected={exp_val:.3f}")
        return False


def load_tensor(args):
    if args.tensor:
        tensor = np.load(args.tensor)
        if tensor.shape != (args.rows, args.cols):
            raise ValueError(f"Tensor shape {tensor.shape} does not match --rows/--cols")
        if tensor.dtype != np.float16:
            tensor = tensor.astype(np.float16)
        return tensor

    if args.color_test_pattern:
        # Create a simple color test pattern
        tensor = create_color_test_pattern_tensor(args.rows, args.cols)
        print(f"[TX] Using color test pattern for roundtrip verification")
        print(f"[TX] Colors used: {', '.join(COLOR_NAMES)} (values 0-{CODEC_MAX_VAL})")
        print(f"[TX] Tensor shape: {tensor.shape}")
        print(f"[TX] Visual grid size: {args.grid_size}×{args.grid_size}")
        if args.rows != args.grid_size or args.cols != args.grid_size:
            print(f"[TX] ⚠️  WARNING: Tensor size ({args.rows}×{args.cols}) != grid size ({args.grid_size}×{args.grid_size})")
            print(f"[TX] ⚠️  Tensor will only fill {args.rows/args.grid_size*100:.0f}% of display width, {args.cols/args.grid_size*100:.0f}% of height")
            print(f"[TX] ⚠️  For full-screen display, use --grid-size {args.rows} (or change --rows/--cols to {args.grid_size})")
        print(f"[TX] First row: {tensor[0, :min(10, args.cols)]}")
        print(f"[TX] Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"[TX] Pattern will exercise all 5 colors during transmission")
        
        # Test the codec roundtrip locally to verify it works
        from codec import codec
        test_codec = codec(args.rows, args.cols, CODEC_MIN_VAL, CODEC_MAX_VAL)
        test_grids = test_codec.encode(tensor)
        
        print(f"[TX ENCODE] Encoded into {test_grids.shape[0]} grids")
        for i in range(min(3, test_grids.shape[0])):
            print(f"  Grid {i+1}: unique values = {np.unique(test_grids[i])}, first row = {test_grids[i][0,:5]}")
        
        test_decoded = test_codec.decode(test_grids)
        matches = np.sum(tensor.view(np.uint16) == test_decoded.view(np.uint16))
        print(f"[TX] Local codec test: {matches}/{tensor.size} cells roundtrip correctly ({matches/tensor.size*100:.1f}%)")
        
        if matches == tensor.size:
            print(f"[TX] ✅ Perfect local roundtrip!")
        else:
            print(f"[TX] ⚠️  {tensor.size - matches} cells differ after local roundtrip")
            print(f"[TX] Original first row: {tensor[0,:10]}")
            print(f"[TX] Decoded first row:  {test_decoded[0,:10]}")
        
        return tensor

    if args.test_pattern:
        # Create a known test pattern
        tensor = create_test_pattern_tensor(args.rows, args.cols)
        print(f"[TX] Using test pattern tensor")
        print(f"[TX] First row: {tensor[0, :10]}")
        print(f"[TX] Diagonal: {[tensor[i, i] for i in range(min(10, args.rows, args.cols))]}")
        return tensor

    # Use PyTorch for random tensor generation (more realistic for ML use cases)
    torch.manual_seed(args.seed)
    tensor_torch = torch.randn(args.rows, args.cols, dtype=torch.float32)
    tensor = tensor_torch.numpy().astype(np.float16)
    
    print(f"[TX] Generated random tensor with PyTorch (seed={args.seed})")
    print(f"[TX] Stats: min={tensor.min():.3f}, max={tensor.max():.3f}, mean={tensor.mean():.3f}, std={tensor.std():.3f}")
    print(f"[TX] First row (5 vals): {tensor[0, :5]}")
    print(f"[TX] Hint: Use --verify-random --seed {args.seed} on RX to verify transmission")
    return tensor


def decode_captured_frames(color_frames, codec_obj, rows, cols, grid_size, force_alternating=False, flip_horizontal=False, verbose=True):
    """
    Decode captured color frames into grids for 5-color system.
    No phases - each frame is a direct grid slice with values 0-4.
    
    Args:
        flip_horizontal: If True, flip each frame horizontally before assembly
        verbose: If False, reduce logging output (useful for stream mode)
    """
    # Validate codec_obj is actually a codec object
    if not hasattr(codec_obj, 'grids_needed'):
        raise TypeError(f"codec_obj must be a codec object, got {type(codec_obj)}: {codec_obj}")
    
    rows_per_slice, cols_per_slice, num_row_slices, num_col_slices = compute_slice_layout(rows, cols, grid_size)
    # 5-color system: no phases, so just 1 frame per slice
    expected = codec_obj.grids_needed() * num_row_slices * num_col_slices
    if len(color_frames) < expected:
        raise ValueError(f"Need {expected} frames, captured {len(color_frames)}")

    if verbose:
        print(f"[RX] Decoding {len(color_frames)} frames into {codec_obj.grids_needed()} grids (5-color system)")
    
    # Decode frames in order - each frame is one grid slice
    frame_iter = iter(color_frames)
    grids = []
    frame_counter = 0
    try:
        for grid_idx in range(codec_obj.grids_needed()):
            row_bands = []
            for row_idx in range(num_row_slices):
                col_blocks = []
                for col_idx in range(num_col_slices):
                    # Get next frame - direct mapping, no phases
                    frame = next(frame_iter)
                    frame_counter += 1
                    
                    # Unflip the frame if it was captured with horizontal flip enabled
                    # The Receiver already flipped it during capture, so we need to flip it back
                    # before assembling grids for the codec
                    if flip_horizontal:
                        frame = np.fliplr(frame)
                    
                    slice_values = frame.astype(np.uint8)
                    col_blocks.append(slice_values)
                    
                    # Debug: show unique values and corner samples (only if verbose)
                    if verbose:
                        unique_vals = np.unique(frame)
                        print(f"[RX DECODE] Frame {frame_counter}: Grid {grid_idx+1}, slice ({row_idx},{col_idx})")
                        print(f"  Unique values ({len(unique_vals)}): {unique_vals}")
                        print(f"  Corners: TL={frame[0,0]}, TR={frame[0,-1]}, BL={frame[-1,0]}, BR={frame[-1,-1]}")
                        print(f"  First row (5 vals): {frame[0,:5]}")
                        
                row_bands.append(np.hstack(col_blocks))
            full_grid = np.vstack(row_bands)
            final_grid = full_grid[:rows, :cols]
            
            # Debug: show assembled grid info (only if verbose)
            if verbose:
                print(f"[RX DECODE] Grid {grid_idx+1} assembled: shape={final_grid.shape}")
                print(f"  Final grid (after crop): shape={final_grid.shape}")
                print(f"  First row (5 vals): {final_grid[0,:5]}")
                print(f"  Unique values in final: {np.unique(final_grid)}")
            
            grids.append(final_grid)
    except StopIteration as exc:
        raise ValueError("Incomplete frame sequence for decoding") from exc

    stacked_grids = np.stack(grids, axis=0)
    
    # Debug: show the range of values in the decoded grids (only if verbose)
    if verbose:
        print(f"\n[RX DECODE] Final stacked grids shape: {stacked_grids.shape}")
        for i, grid in enumerate(stacked_grids):
            unique_vals = np.unique(grid)
            print(f"[RX DECODE] Grid {i+1} final: unique={unique_vals}, min={grid.min()}, max={grid.max()}")
            print(f"  First row (5 vals): {grid[0,:5]}")
    
    # Ensure values are within valid range [0, 4]
    if stacked_grids.max() > CODEC_MAX_VAL:
        print(f"[RX] WARNING: Grid values exceed {CODEC_MAX_VAL} (max={stacked_grids.max()}), clamping to valid range")
        stacked_grids = np.clip(stacked_grids, CODEC_MIN_VAL, CODEC_MAX_VAL)
    
    return stacked_grids


def generate_expected_color_frames(tensor, codec_obj, grid_size, flip_horizontal=False):
    """Generate the exact color indices TX should produce for each frame (5-color system)."""
    encoded = codec_obj.encode(tensor)
    rows_per_slice, cols_per_slice, num_row_slices, num_col_slices = compute_slice_layout(
        codec_obj.rows, codec_obj.cols, grid_size)
    frames = []
    for grid_idx in range(encoded.shape[0]):
        full_grid = encoded[grid_idx]
        for row_idx in range(num_row_slices):
            start_row = row_idx * rows_per_slice
            end_row = min(start_row + rows_per_slice, full_grid.shape[0])
            for col_idx in range(num_col_slices):
                start_col = col_idx * cols_per_slice
                end_col = min(start_col + cols_per_slice, full_grid.shape[1])
                slice_values = np.zeros((rows_per_slice, cols_per_slice), dtype=full_grid.dtype)
                window = full_grid[start_row:end_row, start_col:end_col]
                slice_values[:window.shape[0], :window.shape[1]] = window
                # 5-color system: direct mapping, no bit extraction
                if flip_horizontal:
                    slice_values = np.fliplr(slice_values)
                frames.append(slice_values.astype(np.uint8))
    return frames


def compare_captured_to_expected(captured_frames, expected_frames):
    total_expected = len(expected_frames)
    if len(captured_frames) != total_expected:
        print(f"[RX DEBUG] Expected {total_expected} frames but captured {len(captured_frames)}.")
    for idx, (captured, expected) in enumerate(zip(captured_frames, expected_frames), start=1):
        if captured.shape != expected.shape:
            print(f"[RX DEBUG] Frame {idx}: shape mismatch captured {captured.shape} vs expected {expected.shape}")
            continue
        matches = np.count_nonzero(captured == expected)
        total = captured.size
        mismatch = total - matches
        match_pct = matches / max(1, total) * 100.0
        if mismatch == 0:
            print(f"[RX DEBUG] Frame {idx}: ✅ match ({match_pct:.2f}% cells equal)")
        else:
            diff = captured.astype(np.int16) - expected.astype(np.int16)
            unique_diff = np.unique(diff)
            print(f"[RX DEBUG] Frame {idx}: ⚠️ {mismatch}/{total} cells differ ({match_pct:.2f}% match). "
                  f"Differences: {unique_diff[:5]}")


def run_tx(args):
    tensor = load_tensor(args)
    stream_count = args.stream_count if hasattr(args, 'stream_count') and args.stream_count else 1
    tx = CodecTransmitterDisplay(tensor, args.grid_size, args.fps, args.start_signal_seconds, stream_count=stream_count)

    cv2.namedWindow("Codec TX", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Codec TX", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n" + "=" * 70)
    print("TX MODE - CODEC VISUAL LINK (5-Color System)")
    print("=" * 70)
    print(f"Colors: {', '.join(COLOR_NAMES)} (values 0-{CODEC_MAX_VAL})")
    print(f"Tensor: {tensor.shape}, dtype={tensor.dtype}")
    print(f"Visual grid: {args.grid_size}x{args.grid_size}, FPS: {args.fps}")
    print(f"Codec: {tx.codec.grids_needed()} grids needed for FP16 encoding")
    print(f"Total frames to send: {tx.expected_frames()}")
    if tx.start_signal_frames > 0:
        print(f"Start signal: solid green for {tx.start_signal_frames} frame(s) "
              f"({tx.start_signal_seconds:.2f}s) before data begins")
    print("Controls: 'p' to start, 'r' to reset, 'q' to quit")
    print("Use this window full-screen facing the receiver camera.\n")

    while True:
        frame = tx.render()  # Render current state
        cv2.imshow("Codec TX", frame)
        tx.update()  # Then advance to next frame

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

    rx = Receiver(grid_size=args.grid_size, brightness_adjust=args.brightness_adjust)
    if args.flip_horizontal:
        rx.flip_horizontal = True
        print("[RX] Horizontal flip enabled (for facing cameras)")
    if args.brightness_adjust != 0:
        print(f"[RX] Brightness adjust: {args.brightness_adjust}")
    if args.debug_color_sampling:
        rx.debug_color_sampling = True
        print("[RX] Color sampling debug enabled")
    codec_obj = codec(args.rows, args.cols, min_val=CODEC_MIN_VAL, max_val=CODEC_MAX_VAL)
    if not hasattr(codec_obj, 'grids_needed'):
        raise TypeError(f"Failed to create codec object: got {type(codec_obj)}")
    layout = compute_slice_layout(args.rows, args.cols, args.grid_size)
    # 5-color system: no phases, so just 1 frame per slice
    frames_per_tensor = codec_obj.grids_needed() * layout[2] * layout[3]
    stream_count = args.stream_count if hasattr(args, 'stream_count') and args.stream_count else 1
    total_frames_needed = frames_per_tensor * stream_count

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
    last_capture_time = 0  # Track when we last captured a frame
    min_capture_interval = args.capture_interval  # Minimum seconds between captures
    waiting_for_start_signal = False
    waiting_for_first_data_frame = False
    start_signal_detected = False
    start_signal_wait_start = 0.0
    start_signal_wait_logged = False
    expected_frames = None
    
    # Stream mode tracking
    tensors_received = 0
    stream_start_time = None
    frames_per_tensor = frames_per_tensor
    stream_mode = stream_count > 1
    if args.debug_frame_compare or args.verify_random:
        reference_tensor = None
        if args.color_test_pattern:
            reference_tensor = create_color_test_pattern_tensor(args.rows, args.cols)
            print(f"[RX] Using color test pattern as reference for frame comparison")
            print(f"[RX] Reference first row: {reference_tensor[0, :min(10, args.cols)]}")
            print(f"[RX] Reference value range: [{reference_tensor.min():.3f}, {reference_tensor.max():.3f}]")
        elif args.verify_random:
            # Generate the same random tensor as TX using PyTorch with the seed
            torch.manual_seed(args.seed)
            tensor_torch = torch.randn(args.rows, args.cols, dtype=torch.float32)
            reference_tensor = tensor_torch.numpy().astype(np.float16)
            
            print(f"[RX] Generated reference random tensor with PyTorch (seed={args.seed})")
            print(f"[RX] Reference shape: {reference_tensor.shape}")
            print(f"[RX] Reference stats: min={reference_tensor.min():.3f}, max={reference_tensor.max():.3f}, mean={reference_tensor.mean():.3f}")
            print(f"[RX] First row (5 vals): {reference_tensor[0, :5]}")
        elif args.verify_test_pattern:
            reference_tensor = create_test_pattern_tensor(args.rows, args.cols)
        elif args.tensor:
            try:
                reference_tensor = np.load(args.tensor)
                if reference_tensor.shape != (args.rows, args.cols):
                    print(f"[RX] ⚠️  Reference tensor shape {reference_tensor.shape} does not match "
                          f"--rows/--cols ({args.rows}, {args.cols}); ignoring.")
                    reference_tensor = None
                elif reference_tensor.dtype != np.float16:
                    reference_tensor = reference_tensor.astype(np.float16)
            except Exception as ex:
                print(f"[RX] ⚠️  Could not load reference tensor from {args.tensor}: {ex}")
        if reference_tensor is None:
            print("[RX] ⚠️  Frame comparison requested but no reference tensor available.")
        else:
            expected_frames = generate_expected_color_frames(
                reference_tensor, codec_obj, args.grid_size, flip_horizontal=args.flip_horizontal)

    print("\n" + "=" * 70)
    print("RX MODE - CODEC VISUAL LINK (5-Color System)")
    print("=" * 70)
    print(f"Colors: {', '.join(COLOR_NAMES)} (values 0-{CODEC_MAX_VAL})")
    print(f"Tensor size: {args.rows}×{args.cols}, Visual grid: {args.grid_size}×{args.grid_size}")
    if args.rows != args.grid_size or args.cols != args.grid_size:
        print(f"⚠️  WARNING: Tensor size != grid size. TX must use same --grid-size!")
        print(f"   Suggestion: Add --grid-size {args.rows} to BOTH tx and rx commands")
    
    if stream_count > 1:
        print(f"Stream mode: expecting {stream_count} tensors ({total_frames_needed} total frames)")
        print(f"  Frames per tensor: {frames_per_tensor} ({codec_obj.grids_needed()} grids × {layout[2]} row × {layout[3]} col)")
    else:
        print(f"Expecting {total_frames_needed} frames "
              f"({codec_obj.grids_needed()} grids × {layout[2]} row slices × {layout[3]} col slices)")
    print("Controls:")
    print("  c   - Calibrate (lock onto TX grid)")
    print("  p   - Start capture (after calibration)")
    print("  r   - Reset captured frames")
    print("  f   - Toggle horizontal flip (IMPORTANT: enable if cameras face each other!)")
    print("  d   - Toggle sample region overlay (debug visualization)")
    print("  s   - Toggle color sampling debug (shows RGB values vs palette)")
    print("  +/- - Adjust border offset")
    print("  b/B - Decrease/Increase brightness (-100 to +100)")
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

            # Update sample regions for visualization (even when not capturing)
            if rx.show_sample_regions:
                _ = rx.read_color_pattern(frame, args.grid_size)

            # Draw sample region overlays on camera feed
            if rx.show_sample_regions and rx.sample_polys:
                for poly in rx.sample_polys:
                    pts = poly.reshape((-1, 1, 2))
                    cv2.polylines(display, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)

            if capture_active:
                pattern = rx.read_color_pattern(frame, args.grid_size)
                if pattern is None:
                    continue
                # Wait for TX start signal (unless skipped)
                if waiting_for_start_signal:
                    if is_start_signal(pattern):
                        start_signal_detected = True
                        waiting_for_start_signal = False
                        waiting_for_first_data_frame = True
                        start_signal_wait_logged = False
                        print("\n[RX] Start signal detected! Waiting for first data frame...")
                    else:
                        if start_signal_wait_start == 0.0:
                            start_signal_wait_start = time.time()
                        elif (not start_signal_wait_logged and
                              time.time() - start_signal_wait_start > 2.0):
                            print("\n[RX] Waiting for start signal... ensure TX has started and showing solid green.")
                            start_signal_wait_logged = True
                        continue
                if waiting_for_first_data_frame:
                    if is_start_signal(pattern):
                        continue
                    waiting_for_first_data_frame = False
                    print("\n[RX] Start signal complete - capturing data frames.")
                if len(captured_frames) == 0:
                    if is_calibration_pattern(pattern):
                        continue
                    if not args.skip_start_signal and not start_signal_detected:
                        # Still haven't seen the start signal; keep waiting
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
                
                if diff_last < min_changed:
                    continue

                now = time.time()

                # We detected significant change, so this is likely a new frame
                # Only print debug for unusual patterns when requested
                if args.debug_capture_all:
                    unique_vals = np.unique(pattern)
                    if len(unique_vals) > 3 or frame_count % 30 == 0:
                        print(f"\n[RX DEBUG] Frame type: {phase}, unique values: {unique_vals}, "
                              f"max={pattern.max()}, diff={diff_last}/{total_cells}")

                if phase_seen_time is None:
                    phase_seen_time = now

                if pending_pattern is None or pending_phase != phase:
                    pending_pattern = pattern.copy()
                    pending_phase = phase
                    pending_consistency = 1
                    phase_seen_time = now  # Reset timer whenever phase changes
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

                    # Check if we have enough frames for one tensor
                    frames_for_current_tensor = len(captured_frames) - (tensors_received * frames_per_tensor)
                    
                    if frames_for_current_tensor >= frames_per_tensor:
                        if stream_start_time is None:
                            stream_start_time = time.time()
                        
                        # Extract frames for this tensor only
                        start_idx = tensors_received * frames_per_tensor
                        end_idx = start_idx + frames_per_tensor
                        current_tensor_frames = captured_frames[start_idx:end_idx]
                        
                        print(f"\n[RX] Decoding tensor {tensors_received+1}/{stream_count} (frames {start_idx+1}-{end_idx})...")
                        try:
                            # Debug: verify codec object
                            if not hasattr(codec_obj, 'grids_needed'):
                                raise TypeError(f"Codec object is not valid: {type(codec_obj)}")
                            # Use verbose logging only for first tensor in stream mode
                            verbose_decode = not stream_mode or tensors_received == 0
                            grids = decode_captured_frames(current_tensor_frames, codec_obj, args.rows, args.cols, args.grid_size, 
                                                          force_alternating=args.force_alternating,
                                                          flip_horizontal=rx.flip_horizontal,
                                                          verbose=verbose_decode)
                            if expected_frames is not None and not stream_mode:
                                # Only compare frames in non-stream mode to avoid confusion
                                print("\n[RX] Comparing captured frames to expected values...")
                                compare_captured_to_expected(current_tensor_frames, expected_frames)
                            decoded_tensor = codec_obj.decode(grids)
                            
                            # Check for NaN values
                            nan_count = np.isnan(decoded_tensor).sum()
                            if nan_count > 0:
                                print(f"[RX] ⚠️  Warning: Decoded tensor contains {nan_count} NaN values!")
                                # Find which grids might be problematic
                                for i, grid in enumerate(grids):
                                    unique_vals = np.unique(grid)
                                    expected = set(range(CODEC_MAX_VAL + 1))  # 0-4 for 5-color system
                                    missing = expected - set(unique_vals)
                                    if missing:
                                        print(f"[RX] Grid {i+1} is missing values: {sorted(missing)}")
                                        if 0 in missing:
                                            print(f"[RX] Tip: Missing value 0 ({COLOR_NAMES[0]}) - try increasing border with '+' key")
                                        if CODEC_MAX_VAL in missing:
                                            print(f"[RX] Tip: Missing value {CODEC_MAX_VAL} ({COLOR_NAMES[CODEC_MAX_VAL]}) - might be clipping, try decreasing border with '-' key")
                            else:
                                print("[RX] ✓ Decode complete!")
                            
                            print(f"[RX] Decoded tensor: {decoded_tensor.shape}, dtype={decoded_tensor.dtype}")
                            print(f"[RX] Sample row: {decoded_tensor[0, :5]}")
                            
                            # Show statistics
                            valid_values = decoded_tensor[~np.isnan(decoded_tensor)]
                            if len(valid_values) > 0:
                                print(f"[RX] Valid values: min={valid_values.min():.3f}, "
                                      f"max={valid_values.max():.3f}, mean={valid_values.mean():.3f}")
                            
                            # If test pattern mode, verify known values
                            if args.color_test_pattern:
                                verify_color_test_pattern(decoded_tensor, args.rows, args.cols)
                            elif args.verify_random:
                                # Verify against the deterministic random tensor (same for all iterations in stream)
                                verbose_verify = not stream_mode or tensors_received == 0
                                
                                if verbose_verify:
                                    print("\n[RX VERIFY] Verifying against deterministic random tensor...")
                                
                                # Generate same tensor as TX using PyTorch
                                torch.manual_seed(args.seed)
                                tensor_torch = torch.randn(args.rows, args.cols, dtype=torch.float32)
                                expected = tensor_torch.numpy().astype(np.float16)
                                
                                # Bit-exact comparison
                                expected_u16 = expected.view(np.uint16)
                                decoded_u16 = decoded_tensor.view(np.uint16)
                                matches = np.sum(expected_u16 == decoded_u16)
                                total = expected.size
                                accuracy = matches / total * 100 if total > 0 else 0
                                
                                if verbose_verify:
                                    print(f"[RX VERIFY] Expected (seed={args.seed}): shape={expected.shape}")
                                    print(f"[RX VERIFY] Expected stats: min={expected.min():.3f}, max={expected.max():.3f}, mean={expected.mean():.3f}")
                                    print(f"[RX VERIFY] Decoded stats:  min={decoded_tensor.min():.3f}, max={decoded_tensor.max():.3f}, mean={decoded_tensor.mean():.3f}")
                                    print(f"[RX VERIFY] Expected first row (5 vals): {expected[0, :5]}")
                                    print(f"[RX VERIFY] Decoded first row (5 vals):  {decoded_tensor[0, :5]}")
                                print(f"[RX VERIFY] Tensor {tensors_received+1}: Bit-exact matches: {matches}/{total} ({accuracy:.2f}%)")
                                
                                if accuracy == 100.0:
                                    print(f"[RX VERIFY] ✅ Tensor {tensors_received+1}: Perfect match!")
                                elif accuracy > 99.0:
                                    print(f"[RX VERIFY] ✅ Tensor {tensors_received+1}: Nearly perfect ({accuracy:.2f}%)")
                                    if verbose_verify:
                                        # Show a few mismatches
                                        mismatches = np.where(expected_u16 != decoded_u16)
                                        if len(mismatches[0]) > 0:
                                            print(f"[RX VERIFY] Sample mismatches (showing up to 5):")
                                            for idx in range(min(5, len(mismatches[0]))):
                                                i, j = mismatches[0][idx], mismatches[1][idx]
                                                exp_val = expected[i, j]
                                                dec_val = decoded_tensor[i, j]
                                                print(f"  [{i},{j}]: decoded={dec_val:.6f}, expected={exp_val:.6f}")
                                else:
                                    print(f"[RX VERIFY] ⚠️  Tensor {tensors_received+1}: Only {accuracy:.2f}% match - transmission errors!")
                                    if verbose_verify:
                                        # Show some mismatches
                                        mismatches = np.where(expected_u16 != decoded_u16)
                                        if len(mismatches[0]) > 0:
                                            print(f"[RX VERIFY] Sample mismatches (showing up to 10):")
                                            for idx in range(min(10, len(mismatches[0]))):
                                                i, j = mismatches[0][idx], mismatches[1][idx]
                                                exp_val = expected[i, j]
                                                dec_val = decoded_tensor[i, j]
                                                print(f"  [{i},{j}]: decoded={dec_val:.3f}, expected={exp_val:.3f}")
                            elif args.verify_test_pattern:
                                print("\n[RX] Verifying test pattern values...")
                                expected = create_test_pattern_tensor(args.rows, args.cols)
                                
                                # Check corner values
                                checks = [
                                    ((0, 0), "Top-left"),
                                    ((0, -1), "Top-right"), 
                                    ((-1, 0), "Bottom-left"),
                                    ((-1, -1), "Bottom-right"),
                                    ((args.rows//2, args.cols//2), "Center")
                                ]
                                
                                all_correct = True
                                for (r, c), name in checks:
                                    expected_val = expected[r, c]
                                    decoded_val = decoded_tensor[r, c]
                                    if np.isnan(decoded_val):
                                        print(f"[RX] ❌ {name}: Expected {expected_val:.3f}, got NaN")
                                        all_correct = False
                                    elif abs(decoded_val - expected_val) < 0.01:  # Allow small error
                                        print(f"[RX] ✓ {name}: {decoded_val:.3f} (expected {expected_val:.3f})")
                                    else:
                                        print(f"[RX] ❌ {name}: {decoded_val:.3f} (expected {expected_val:.3f})")
                                        all_correct = False
                                
                                if all_correct:
                                    print("[RX] ✅ All test values verified correctly!")
                                else:
                                    print("[RX] ⚠️  Some test values didn't match")
                            
                            if args.save_decoded:
                                if stream_mode:
                                    # Save with iteration number
                                    base_name = args.save_decoded.replace('.npy', '')
                                    save_path = f"{base_name}_tensor{tensors_received}.npy"
                                    np.save(save_path, decoded_tensor)
                                    print(f"[RX] Saved tensor {tensors_received} to {save_path}")
                                else:
                                    np.save(args.save_decoded, decoded_tensor)
                                    print(f"[RX] Saved tensor to {args.save_decoded}")
                            
                            # Stream mode: count received tensors
                            tensors_received += 1
                            
                            if stream_mode:
                                elapsed = time.time() - stream_start_time
                                tensor_bytes = args.rows * args.cols * 2
                                total_bytes = tensor_bytes * tensors_received
                                throughput_kbps = (total_bytes / 1024) / elapsed if elapsed > 0 else 0
                                
                                print(f"\n[RX STREAM] ═══ Tensor {tensors_received}/{stream_count} received ═══")
                                print(f"[RX STREAM] Total data: {total_bytes} bytes ({total_bytes/1024:.2f} KB)")
                                print(f"[RX STREAM] Elapsed: {elapsed:.2f} seconds")
                                print(f"[RX STREAM] Throughput: {throughput_kbps:.2f} KB/s ({throughput_kbps*8:.2f} Kbps)")
                                print(f"[RX STREAM] Avg per tensor: {elapsed/tensors_received:.2f} seconds")
                                
                                if tensors_received >= stream_count:
                                    print(f"\n[RX STREAM] ✅ All {stream_count} tensors received successfully!")
                                    print(f"[RX STREAM] Final throughput: {throughput_kbps:.2f} KB/s ({throughput_kbps*8:.2f} Kbps)")
                                    capture_active = False
                                else:
                                    # Continue capturing next tensor (keep frames in buffer)
                                    print(f"[RX STREAM] Waiting for tensor {tensors_received+1}/{stream_count}...")
                                    last_pattern = None
                                    pending_pattern = None
                                    pending_phase = None
                                    pending_consistency = 0
                                    phase_seen_time = None
                                    continue
                        except Exception as err:
                            print(f"[RX] ✗ Decode failed: {err}")
                        
                        if not stream_mode or tensors_received >= stream_count:
                            capture_active = False
                            phase_seen_time = None
                            waiting_for_start_signal = False
                            waiting_for_first_data_frame = False
                            start_signal_detected = False
                            start_signal_wait_start = 0.0
                            start_signal_wait_logged = False
            else:
                capture_active = False
                last_pattern = None
                pending_pattern = None
                pending_phase = None
                pending_consistency = 0
                phase_seen_time = None
                waiting_for_start_signal = False
                waiting_for_first_data_frame = False
                start_signal_detected = False
                start_signal_wait_start = 0.0
                start_signal_wait_logged = False

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
        if rx.show_sample_regions and rx.debug_color_samples is not None:
            cv2.imshow('Debug: Color Samples', rx.debug_color_samples)
        if rx.show_sample_regions and rx.debug_sample_grid is not None:
            cv2.imshow('Debug: Sample Grid', rx.debug_sample_grid)

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
                last_capture_time = 0  # Reset capture timing
                waiting_for_start_signal = False
                waiting_for_first_data_frame = False
                start_signal_detected = False
                start_signal_wait_start = 0.0
                start_signal_wait_logged = False
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
                last_capture_time = 0  # Reset capture timing
                waiting_for_start_signal = not args.skip_start_signal
                waiting_for_first_data_frame = False
                start_signal_detected = args.skip_start_signal
                start_signal_wait_start = time.time() if waiting_for_start_signal else 0.0
                start_signal_wait_logged = False
                if waiting_for_start_signal:
                    print("\n[RX] Capture started - waiting for TX start signal (solid green)...")
                else:
                    print("\n[RX] Capture started - waiting for frames...")
        elif key == ord('r'):
            captured_frames.clear()
            last_pattern = None
            pending_pattern = None
            pending_consistency = 0
            pending_phase = None
            phase_seen_time = None
            last_capture_time = 0  # Reset capture timing
            capture_active = False
            waiting_for_start_signal = False
            waiting_for_first_data_frame = False
            start_signal_detected = False
            start_signal_wait_start = 0.0
            start_signal_wait_logged = False
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
        elif key == ord('d'):
            rx.show_sample_regions = not rx.show_sample_regions
            state = "ON" if rx.show_sample_regions else "OFF"
            print(f"\n[RX] Sample region overlay: {state}")
            if not rx.show_sample_regions:
                rx.debug_color_samples = None
                rx.sample_polys = None
                rx.debug_sample_grid = None
                try:
                    cv2.destroyWindow('Debug: Color Samples')
                except cv2.error:
                    pass
                try:
                    cv2.destroyWindow('Debug: Sample Grid')
                except cv2.error:
                    pass
        elif key == ord('b'):
            rx.brightness_adjust = max(-100, rx.brightness_adjust - 5)
            print(f"\n[RX] Brightness: {rx.brightness_adjust}")
        elif key == ord('B'):
            rx.brightness_adjust = min(100, rx.brightness_adjust + 5)
            print(f"\n[RX] Brightness: {rx.brightness_adjust}")
        elif key == ord('s'):
            rx.debug_color_sampling = not rx.debug_color_sampling
            state = "ON" if rx.debug_color_sampling else "OFF"
            print(f"\n[RX] Color sampling debug: {state}")

    cap.release()
    cv2.destroyAllWindows()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Codec Visual Data Link")
    parser.add_argument('--mode', required=True, choices=['tx', 'rx'], help='Run as transmitter or receiver')
    parser.add_argument('--grid-size', type=int, default=VISUAL_GRID_SIZE, 
                        help='Visual grid resolution (should match --rows/--cols for full display)')
    parser.add_argument('--fps', type=float, default=TRANSMISSION_FPS, help='TX frames per second')
    parser.add_argument('--start-signal-seconds', type=float, default=1.0,
                        help='Duration to show solid green start signal before frames')
    parser.add_argument('--stream-count', type=int, default=1,
                        help='Number of tensors to transmit in stream mode (calculates throughput)')
    parser.add_argument('--rows', type=int, default=TEST_TENSOR_ROWS, help='Tensor rows')
    parser.add_argument('--cols', type=int, default=TEST_TENSOR_COLS, help='Tensor cols')
    parser.add_argument('--tensor', type=str, help='Path to .npy tensor for TX mode')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random tensor (TX)')
    parser.add_argument('--test-pattern', action='store_true', help='Use test pattern tensor instead of random (TX)')
    parser.add_argument('--color-test-pattern', action='store_true', help='Use simple color test pattern for TX/RX (transmission and verification)')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index for RX mode')
    parser.add_argument('--brightness-adjust', type=int, default=0,
                        help='Brightness adjustment for color sampling, -100 to +100 (RX mode)')
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
    parser.add_argument('--verify-test-pattern', action='store_true',
                        help='Verify decoded tensor against test pattern (RX, deprecated - use --test-pattern instead)')
    parser.add_argument('--verify-random', action='store_true',
                        help='Verify decoded tensor against deterministic random (use same --seed as TX)')
    parser.add_argument('--flip-horizontal', action='store_true',
                        help='Enable horizontal flip for facing cameras (RX)')
    parser.add_argument('--skip-start-signal', action='store_true',
                        help='Start capturing immediately without waiting for TX start signal (RX)')
    parser.add_argument('--debug-frame-compare', action='store_true',
                        help='Compare each captured frame against expected values (requires known tensor)')
    parser.add_argument('--debug-color-sampling', action='store_true',
                        help='Print detailed color sampling debug info for each captured frame (RX)')
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
