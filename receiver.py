"""
Receiver: Captures camera feed, detects grid, decodes message
"""

import cv2
import numpy as np
import threading
import time
from color_utils import ColorDetector, draw_grid
from decoder import MessageDecoder
from config import (
    GRID_SIZE, DISPLAY_WIDTH, DISPLAY_HEIGHT, CELL_WIDTH, CELL_HEIGHT,
    TARGET_FPS, FRAME_TIME_MS, GRID_BORDER_THICKNESS, REFERENCE_MARKERS,
    COLOR_PALETTE, GRID_BORDER_COLOR
)


class GridDetector:
    """Detects and extracts the color grid from camera frame"""

    def __init__(self, cell_width, cell_height, border_thickness=12):
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.detector = ColorDetector()
        self.border_thickness = border_thickness
        self.reference_markers = REFERENCE_MARKERS
        self.aligned_side = GRID_SIZE * 20  # resolution for warped grid sampling
        # Green border color in BGR: (0, 255, 0)
        self.green_border_bgr = np.array(GRID_BORDER_COLOR, dtype=np.uint8)
        self.green_border_hsv = self._bgr_to_hsv(self.green_border_bgr)
        self.green_hue_tolerance = 12  # degrees in OpenCV hue units (0-180)
        self.green_sat_min = 80
        self.green_val_min = 80
        self.border_color_ema_alpha = 0.15

    def _bgr_to_hsv(self, bgr_color):
        color = np.array([[bgr_color]], dtype=np.uint8)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        return hsv[0, 0]

    def _order_points(self, pts):
        pts = np.array(pts, dtype=np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def detect_green_border(self, frame):
        """
        Detect if green border is visible in the frame.
        Returns (grid_found, x1, y1, x2, y2, mask, edges, match_ratio, angle, rot_info)
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hue_channel = hsv[:, :, 0]
        sat_channel = hsv[:, :, 1]
        val_channel = hsv[:, :, 2]

        center_hue = int(self.green_border_hsv[0])
        tol = self.green_hue_tolerance
        lower_h = (center_hue - tol) % 180
        upper_h = (center_hue + tol) % 180

        if lower_h <= upper_h:
            hue_mask = cv2.inRange(hue_channel, lower_h, upper_h)
        else:
            mask1 = cv2.inRange(hue_channel, lower_h, 179)
            mask2 = cv2.inRange(hue_channel, 0, upper_h)
            hue_mask = cv2.bitwise_or(mask1, mask2)

        sat_mask = cv2.inRange(sat_channel, self.green_sat_min, 255)
        val_mask = cv2.inRange(val_channel, self.green_val_min, 255)
        green_mask = cv2.bitwise_and(hue_mask, cv2.bitwise_and(sat_mask, val_mask))

        # Highlight edges of the border for more stable geometry detection
        ring_mask = green_mask.copy()
        if self.border_thickness > 2:
            inner_kernel_size = max(3, self.border_thickness // 2 | 1)
            inner_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (inner_kernel_size, inner_kernel_size))
            inner = cv2.erode(green_mask, inner_kernel, iterations=1)
            ring_mask = cv2.subtract(green_mask, inner)

        mask_pixels = cv2.countNonZero(ring_mask)
        min_green_pixels = max(2000, int(0.0015 * h * w))
        match_ratio = mask_pixels / max(min_green_pixels, 1)

        edges = cv2.Canny(ring_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_rect = None
        best_rot_rect = None
        max_area = 0
        min_contour_area = 2000
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            if area > max_area:
                max_area = area
                best_rot_rect = cv2.minAreaRect(contour)

        # Make the mask sturdier to handle thin borders/noisy detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        green_mask = cv2.dilate(green_mask, kernel, iterations=1)

        mask_for_geometry = ring_mask
        border_angle = 0.0

        rot_info = None

        if best_rot_rect is not None:
            (cx, cy), (width_rect, height_rect), angle = best_rot_rect
            border_angle = angle
            box = cv2.boxPoints(best_rot_rect)
            ordered = self._order_points(box)
            x_coords = ordered[:, 0]
            y_coords = ordered[:, 1]
            x_min = max(0, int(np.floor(x_coords.min())))
            x_max = min(w - 1, int(np.ceil(x_coords.max())))
            y_min = max(0, int(np.floor(y_coords.min())))
            y_max = min(h - 1, int(np.ceil(y_coords.max())))
            w_box = x_max - x_min + 1
            h_box = y_max - y_min + 1
            aligned_side = self.aligned_side
            dest = np.array([
                [0, 0],
                [aligned_side - 1, 0],
                [aligned_side - 1, aligned_side - 1],
                [0, aligned_side - 1]
            ], dtype=np.float32)
            warp_matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), dest)
            inverse_matrix = cv2.getPerspectiveTransform(dest, ordered.astype(np.float32))
            rot_info = {
                "matrix": warp_matrix,
                "inverse": inverse_matrix,
                "box": ordered,
                "size": (max(width_rect, 1.0), max(height_rect, 1.0)),
                "dest_size": (aligned_side, aligned_side)
            }
        else:
            if mask_pixels < min_green_pixels:
                return False, None, None, None, None, ring_mask, edges, match_ratio, border_angle, None

            y_coords, x_coords = np.where(mask_for_geometry > 0)

            if len(x_coords) == 0 or len(y_coords) == 0:
                return False, None, None, None, None, ring_mask, edges, match_ratio, border_angle, None

            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            w_box = x_max - x_min + 1
            h_box = y_max - y_min + 1

        # Verify it's roughly square (grid should be square-ish)
        # Note: Due to cell dimensions (94x61), grid naturally has ~1.54 aspect ratio
        if rot_info is not None:
            width_rect, height_rect = rot_info["size"]
            aspect = max(width_rect, height_rect) / max(1.0, min(width_rect, height_rect))
        else:
            aspect = max(w_box, h_box) / min(w_box, h_box) if min(w_box, h_box) > 0 else 2
        if aspect > 3.2:
            return False, None, None, None, None, ring_mask, edges, match_ratio, border_angle, rot_info

        # Verify the detected region is reasonably sized (not tiny fragments)
        min_border_size = 150  # At least 150 pixels wide/tall
        if w_box < min_border_size or h_box < min_border_size:
            return False, None, None, None, None, ring_mask, edges, match_ratio, border_angle, rot_info

        # Add margin to skip over the border and reach the grid interior
        margin = max(6, self.border_thickness // 2 + 2)
        x1 = max(0, x_min + margin)
        y1 = max(0, y_min + margin)
        x2 = min(w, x_max - margin)
        y2 = min(h, y_max - margin)

        if x2 <= x1 or y2 <= y1:
            return False, None, None, None, None, ring_mask, edges, match_ratio, border_angle, rot_info

        return True, x1, y1, x2, y2, ring_mask, edges, match_ratio, border_angle, rot_info

    def extract_grid(self, frame):
        """
        Extract the 128x128 grid from the camera frame.

        This is a simple approach: assumes the grid fills the frame.
        For robustness, you could add corner detection markers.

        Args:
            frame: OpenCV image (BGR)

        Returns:
            grid: numpy array (128, 128) with values 0-15
        """
        h, w = frame.shape[:2]

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        # Sample color from center of each cell
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Calculate cell bounds in frame space
                x1 = int((col / GRID_SIZE) * w)
                x2 = int(((col + 1) / GRID_SIZE) * w)
                y1 = int((row / GRID_SIZE) * h)
                y2 = int(((row + 1) / GRID_SIZE) * h)

                # Ensure bounds
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                if x2 - x1 > 0 and y2 - y1 > 0:
                    # Extract cell region
                    cell_region = frame[y1:y2, x1:x2]

                    # Detect dominant color
                    colors = self.detector.detect_colors_adaptive(cell_region)
                    color_idx = np.bincount(colors.flatten()).argmax()
                    grid[row, col] = color_idx

        return grid

    def extract_grid_fast(self, frame, x1=None, y1=None, x2=None, y2=None):
        """
        Faster grid extraction using direct color detection on sampled region.

        Args:
            frame: Input frame
            x1, y1, x2, y2: Grid bounds. If None, uses full frame.
        """
        h, w = frame.shape[:2]

        # Use full frame if bounds not provided
        if x1 is None:
            x1, y1, x2, y2 = 0, 0, w, h

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        grid_w = x2 - x1
        grid_h = y2 - y1

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = int(x1 + (col + 0.5) * grid_w / GRID_SIZE)
                y = int(y1 + (row + 0.5) * grid_h / GRID_SIZE)

                x = np.clip(x, 0, w - 1)
                y = np.clip(y, 0, h - 1)

                pixel = frame[y, x]
                grid[row, col] = self.detector.detect_color(pixel)

        return grid

    def extract_grid_aligned(self, aligned_frame):
        """
        Extract grid assuming aligned_frame is canonical square of size aligned_side.
        """
        size = aligned_frame.shape[0]
        step = size / GRID_SIZE
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = int((col + 0.5) * step)
                y = int((row + 0.5) * step)
                x = np.clip(x, 0, size - 1)
                y = np.clip(y, 0, size - 1)
                pixel = aligned_frame[y, x]
                grid[row, col] = self.detector.detect_color(pixel)
        return grid

    def validate_reference_markers(self, grid, min_matches=None):
        """Ensure reserved reference cells match expected colors."""
        total = len(self.reference_markers)
        if total == 0:
            return True, total, []

        required = total if min_matches is None else min_matches
        matches = 0
        mismatches = []
        for marker in self.reference_markers:
            row, col = marker["position"]
            expected = marker["color"]
            if grid[row, col] == expected:
                matches += 1
            else:
                mismatches.append(marker["name"])

        required = required if required is not None else total
        return matches >= required, matches, mismatches

    def update_green_reference(self, frame, mask):
        """Adaptively update the green border color using detected mask."""
        if mask is None:
            return
        green_pixels = frame[mask > 0]
        if green_pixels.size == 0:
            return
        median_color = np.median(green_pixels, axis=0)
        median_color = np.clip(median_color, 0, 255).astype(np.uint8)
        blended = (
            (1.0 - self.border_color_ema_alpha) * self.green_border_bgr.astype(np.float32)
            + self.border_color_ema_alpha * median_color.astype(np.float32)
        )
        self.green_border_bgr = blended.astype(np.uint8)
        self.green_border_hsv = self._bgr_to_hsv(self.green_border_bgr)

    def sample_reference_colors(self, frame, bounds):
        """Sample actual BGR values for each reference marker."""
        if bounds is None:
            return {}
        x1, y1, x2, y2 = bounds
        grid_w = x2 - x1
        grid_h = y2 - y1
        samples = {}
        for marker in self.reference_markers:
            row, col = marker["position"]
            px = int(x1 + (col + 0.5) * grid_w / GRID_SIZE)
            py = int(y1 + (row + 0.5) * grid_h / GRID_SIZE)
            patch = frame[max(py - 4, 0):min(py + 5, frame.shape[0]),
                          max(px - 4, 0):min(px + 5, frame.shape[1])]
            if patch.size == 0:
                continue
            median_bgr = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            median_hsv = np.median(patch_hsv.reshape(-1, 3), axis=0).astype(np.uint8)
            samples[marker["name"]] = {
                "bgr": median_bgr,
                "hsv": median_hsv,
                "expected_bgr": np.array(COLOR_PALETTE[marker["color"]], dtype=np.float32)
            }
        return samples

    def calibrate_colors(self, frame, bounds):
        """Calibrate color detector based on sampled reference markers."""
        samples = self.sample_reference_colors(frame, bounds)
        if not samples:
            return False, "Calibration failed: no reference samples"

        measured = np.array([info["bgr"] for info in samples.values()], dtype=np.float32)
        expected = np.array([info["expected_bgr"] for info in samples.values()], dtype=np.float32)

        measured_mean = np.clip(measured.mean(axis=0), 1.0, None)
        expected_mean = expected.mean(axis=0)
        scale = expected_mean / measured_mean

        self.detector.apply_channel_scale(scale)

        # Update green border reference using sampled green marker if available
        for marker in self.reference_markers:
            if marker["color"] == 2 and marker["name"] in samples:
                self.green_border_bgr = samples[marker["name"]]["bgr"].astype(np.uint8)
                self.green_border_hsv = samples[marker["name"]]["hsv"].astype(np.uint8)
                break

        return True, f"Calibrated using {len(samples)} refs (scale BGR={scale[0]:.2f},{scale[1]:.2f},{scale[2]:.2f})"


class ReceiverApp:
    def __init__(self, camera_id=0, simple_mode=True):
        self.camera_id = camera_id
        self.running = True
        self.simple_mode = simple_mode  # Only decode current frame, don't accumulate
        self.decoder = MessageDecoder()
        self.grid_detector = GridDetector(CELL_WIDTH, CELL_HEIGHT, GRID_BORDER_THICKNESS)
        self.last_frame = None
        self.last_grid = None
        self.last_frame_decoded_text = ""  # For simple mode
        self.last_grid_bounds = None  # For visualization: (x1, y1, x2, y2)
        self.grid_detected = False  # Detection status
        self.last_green_mask = None  # Latest green mask for debugging
        self.last_edge_mask = None  # Canny edges of border
        self.last_mask_ratio = 0.0
        self.last_border_angle = 0.0
        self.last_rot_box = None
        self.last_inverse_warp = None
        self.last_aligned_size = None
        self.reference_matches = 0
        self.reference_mismatches = []
        self.reference_total = len(REFERENCE_MARKERS)
        self.calibration_message = "Press 'k' to calibrate colors"
        self.calibration_timestamp = 0
        self.lock = threading.Lock()
        self.fps_counter = 0
        self.fps_time = time.time()

    def camera_thread(self):
        """Thread that captures camera frames and decodes grid"""
        print(f"Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            self.running = False
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        frame_time = FRAME_TIME_MS / 1000.0

        print("Camera thread started")

        while self.running:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                self.running = False
                break

            # Store raw frame for display
            with self.lock:
                self.last_frame = frame

            # Detect green border to locate grid
            grid_found, x1, y1, x2, y2, green_mask, edge_mask, match_ratio, angle, rot_info = self.grid_detector.detect_green_border(frame)

            # Store detection status and bounds for visualization
            with self.lock:
                self.grid_detected = grid_found
                self.last_green_mask = green_mask
                self.last_edge_mask = edge_mask
                self.last_mask_ratio = match_ratio
                self.last_border_angle = angle
                if grid_found:
                    self.last_grid_bounds = (x1, y1, x2, y2)
                else:
                    self.last_grid_bounds = None
                if rot_info:
                    self.last_rot_box = rot_info.get("box")
                    self.last_inverse_warp = rot_info.get("inverse")
                    dest_size = rot_info.get("dest_size")
                    self.last_aligned_size = dest_size[0] if dest_size else None
                else:
                    self.last_rot_box = None
                    self.last_inverse_warp = None
                    self.last_aligned_size = None

            if grid_found:
                # Grid detected - extract and decode
                if rot_info and rot_info.get("matrix") is not None:
                    aligned_w, aligned_h = rot_info["dest_size"]
                    aligned_frame = cv2.warpPerspective(frame, rot_info["matrix"], (aligned_w, aligned_h))
                    grid = self.grid_detector.extract_grid_aligned(aligned_frame)
                else:
                    grid = self.grid_detector.extract_grid_fast(frame, x1, y1, x2, y2)

                min_matches = max(1, len(REFERENCE_MARKERS) - 1)
                markers_ok, matches, mismatches = self.grid_detector.validate_reference_markers(
                    grid, min_matches=min_matches
                )

                with self.lock:
                    self.reference_matches = matches
                    self.reference_mismatches = mismatches

                if not markers_ok:
                    # Reference markers not aligned; treat as detection failure
                    with self.lock:
                        self.grid_detected = False
                        self.last_grid = None
                        self.last_grid_bounds = None
                        self.last_frame_decoded_text = ""
                    continue

                # Update green reference color using latest mask
                self.grid_detector.update_green_reference(frame, green_mask)

                # Decode
                if self.simple_mode:
                    # Decode current frame independently (no accumulation)
                    from color_utils import grid_to_bytes
                    import struct
                    frame_bytes = grid_to_bytes(grid)
                    frame_counter = struct.unpack('>I', frame_bytes[0:4])[0]
                    message_id = struct.unpack('>H', frame_bytes[4:6])[0]
                    message_length = struct.unpack('>H', frame_bytes[6:8])[0]
                    payload = frame_bytes[8:]
                    try:
                        decoded_text = payload[:message_length].decode('utf-8', errors='ignore')
                    except:
                        decoded_text = ""
                    with self.lock:
                        self.last_frame_decoded_text = decoded_text
                else:
                    decoded_text = self.decoder.process_frame(grid)

                # Store grid for display thread
                with self.lock:
                    self.last_grid = grid
            else:
                # No grid detected - clear the decoded text
                with self.lock:
                    self.last_frame_decoded_text = ""
                    self.last_grid = None
                    self.reference_matches = 0
                    self.reference_mismatches = []

            self.fps_counter += 1

            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        cap.release()

    def display_loop(self):
        """Display split-screen: camera feed + decoded message (runs on main thread)."""
        print("Display loop started")

        window_name = "NO-MARGIN-VIS Receiver"
        display_available = False

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, DISPLAY_WIDTH * 2, DISPLAY_HEIGHT)
            display_available = True
            print("✓ Display window created successfully")
        except Exception as e:
            print(f"⚠ Display not available: {type(e).__name__}: {e}")
            print("  Unable to continue without a desktop window.")
            self.running = False
            return

        while self.running:
            display_start = time.time()

            with self.lock:
                frame = self.last_frame
                grid = self.last_grid
                grid_detected = self.grid_detected
                grid_bounds = self.last_grid_bounds
                green_mask = self.last_green_mask
                edge_mask = self.last_edge_mask
                mask_ratio = self.last_mask_ratio
                border_angle = self.last_border_angle
                rot_box = self.last_rot_box
                inverse_warp = self.last_inverse_warp
                aligned_size = self.last_aligned_size
                frame_text = self.last_frame_decoded_text
                reference_matches = self.reference_matches
                reference_total = self.reference_total
                reference_mismatches = list(self.reference_mismatches)
                calibration_message = self.calibration_message

            if frame is None:
                time.sleep(0.01)
                continue

            # Left side: camera feed
            h, w = frame.shape[:2]
            target_h = DISPLAY_HEIGHT
            target_w = max(1, int(target_h * w / h))
            if target_w > DISPLAY_WIDTH:
                target_w = DISPLAY_WIDTH
                target_h = max(1, int(target_w * h / w))

            camera_resized = cv2.resize(frame, (target_w, target_h))

            mask_resized = None
            edges_resized = None
            if green_mask is not None:
                mask_resized = cv2.resize(green_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            if edge_mask is not None:
                edges_resized = cv2.resize(edge_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            overlay_active = False

            if grid_detected:
                if mask_resized is not None:
                    mask_colored = np.zeros_like(camera_resized)
                    mask_colored[:, :, 1] = mask_resized
                    camera_resized = cv2.addWeighted(camera_resized, 1.0, mask_colored, 0.35, 0)
            else:
                overlay = np.zeros_like(camera_resized)
                if mask_resized is not None:
                    overlay[:, :, 1] = mask_resized
                if edges_resized is not None:
                    overlay[:, :, 2] = edges_resized
                overlay_active = bool(np.count_nonzero(overlay))
                if overlay_active:
                    camera_resized = cv2.addWeighted(camera_resized, 0.5, overlay, 0.5, 0)

            pad_left = (DISPLAY_WIDTH - target_w) // 2
            pad_right = DISPLAY_WIDTH - target_w - pad_left
            pad_top = (DISPLAY_HEIGHT - target_h) // 2
            pad_bottom = DISPLAY_HEIGHT - target_h - pad_top
            camera_display = cv2.copyMakeBorder(
                camera_resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            if grid_detected and grid_bounds is not None:
                x1, y1, x2, y2 = grid_bounds
                scale_x = target_w / w
                scale_y = target_h / h
                sx1 = int(x1 * scale_x) + pad_left
                sy1 = int(y1 * scale_y) + pad_top
                sx2 = int(x2 * scale_x) + pad_left
                sy2 = int(y2 * scale_y) + pad_top
                cv2.rectangle(camera_display, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
                cv2.putText(camera_display, "GRID DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(camera_display, f"Angle: {border_angle:.1f} deg", (10, 65),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2)
                if rot_box is not None:
                    poly = []
                    for px, py in rot_box:
                        sx = int(px * scale_x) + pad_left
                        sy = int(py * scale_y) + pad_top
                        poly.append([sx, sy])
                    cv2.polylines(camera_display, [np.array(poly, dtype=np.int32)], True, (0, 0, 255), 2)
                # Draw expected reference marker locations relative to detected border
                if inverse_warp is not None and aligned_size:
                    coords = []
                    for marker in REFERENCE_MARKERS:
                        row, col = marker["position"]
                        u = (col + 0.5) / GRID_SIZE * (aligned_size - 1)
                        v = (row + 0.5) / GRID_SIZE * (aligned_size - 1)
                        coords.append([u, v])
                    pts = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
                    mapped = cv2.perspectiveTransform(pts, inverse_warp).reshape(-1, 2)
                    for (mx, my) in mapped:
                        sx = int(mx * scale_x) + pad_left
                        sy = int(my * scale_y) + pad_top
                        cv2.circle(camera_display, (sx, sy), 10, (0, 0, 255), 2)
                elif sx2 > sx1 and sy2 > sy1:
                    ref_width = sx2 - sx1
                    ref_height = sy2 - sy1
                    for marker in REFERENCE_MARKERS:
                        row, col = marker["position"]
                        rel_x = sx1 + int((col + 0.5) / GRID_SIZE * ref_width)
                        rel_y = sy1 + int((row + 0.5) / GRID_SIZE * ref_height)
                        cv2.circle(camera_display, (rel_x, rel_y), 10, (0, 0, 255), 2)
            else:
                h_cam, w_cam = camera_display.shape[:2]
                cv2.rectangle(camera_display, (10, 10), (w_cam - 10, h_cam - 10), (0, 0, 255), 2)
                cv2.putText(camera_display, "NO GRID DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if overlay_active:
                    cv2.putText(camera_display, "HSV+CANNY VIEW", (10, 60),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

            cv2.putText(camera_display, f"Mask ratio: {mask_ratio:.2f}", (10, 95),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0), 2)

            # Right side: reconstructed grid
            grid_display = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            grid_to_draw = grid if grid is not None else np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            draw_grid(grid_display, grid_to_draw, CELL_WIDTH, CELL_HEIGHT)
            for marker in REFERENCE_MARKERS:
                row, col = marker["position"]
                marker_color = COLOR_PALETTE[marker["color"]]
                center = (
                    col * CELL_WIDTH + CELL_WIDTH // 2,
                    row * CELL_HEIGHT + CELL_HEIGHT // 2
                )
                cv2.circle(grid_display, center, 8, (0, 0, 0), 2)
                cv2.circle(grid_display, center, 6, marker_color, -1)

            message = frame_text if self.simple_mode else self.decoder.get_full_message()
            y_offset = 40
            x_offset = 10
            max_chars_per_line = 40
            lines = [message[i:i + max_chars_per_line] for i in range(0, len(message), max_chars_per_line)]

            for i, line in enumerate(lines[-10:]):  # Show last 10 lines
                y = y_offset + i * 25
                if y < DISPLAY_HEIGHT - 20:
                    cv2.putText(
                        grid_display, line,
                        (x_offset, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1
                    )

            if self.simple_mode:
                stats = {'frames': self.fps_counter, 'message_length': len(message)}
            else:
                stats = self.decoder.get_stats()
            fps = self.fps_counter / (time.time() - self.fps_time + 0.001)
            stats_text = (
                f"FPS: {fps:.1f} | Frames: {stats['frames']} | "
                f"Msg: {len(message)}ch | Detect: {'YES' if grid_detected else 'NO'}"
            )
            if reference_total:
                ref_text = f"Ref: {reference_matches}/{reference_total}"
                if reference_mismatches:
                    ref_text += f" ({','.join(reference_mismatches)})"
            else:
                ref_text = "Ref: n/a"
            cv2.putText(
                grid_display, stats_text,
                (x_offset, DISPLAY_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1
            )
            cv2.putText(
                grid_display, ref_text,
                (x_offset, DISPLAY_HEIGHT - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 255), 1
            )
            cv2.putText(
                grid_display, calibration_message,
                (x_offset, DISPLAY_HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1
            )

            combined = np.hstack([camera_display, grid_display])

            try:
                cv2.imshow(window_name, combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('c'):
                    with self.lock:
                        self.decoder.decoded_text = ""
                        self.last_frame_decoded_text = ""
                elif key == ord('k'):
                    with self.lock:
                        calib_frame = None if self.last_frame is None else self.last_frame.copy()
                        calib_bounds = self.last_grid_bounds
                    if calib_frame is None or calib_bounds is None:
                        success = False
                        msg = "Calibration skipped: grid not detected"
                    else:
                        success, msg = self.grid_detector.calibrate_colors(calib_frame, calib_bounds)
                    with self.lock:
                        self.calibration_message = msg
                        self.calibration_timestamp = time.time()
                    indicator = "✓" if success else "✗"
                    print(f"{indicator} {msg}")
            except Exception:
                print("⚠ Display lost; stopping receiver.")
                self.running = False
                break

            elapsed = time.time() - display_start
            sleep_time = max(0, FRAME_TIME_MS / 1000.0 - elapsed)
            time.sleep(sleep_time)

        if display_available:
            cv2.destroyAllWindows()

    def start(self):
        """Start receiver with desktop UI"""
        print(f"Starting receiver (desktop OpenCV mode) with camera {self.camera_id}")

        camera_t = threading.Thread(target=self.camera_thread, daemon=True, name="CameraThread")
        camera_t.start()

        try:
            self.display_loop()
        except KeyboardInterrupt:
            print("Interrupted, shutting down receiver...")
            self.running = False

        camera_t.join(timeout=2)


if __name__ == "__main__":
    app = ReceiverApp(camera_id=0)

    print("=" * 60)
    print("NO-MARGIN-VIS RECEIVER")
    print("=" * 60)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"FPS: {TARGET_FPS}")
    print(f"\nControls:")
    print(f"  q: Quit")
    print(f"  c: Clear decoded text")
    print(f"  k: Calibrate colors from reference markers")
    print("=" * 60)

    app.start()
