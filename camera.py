import dataclasses
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional

WIDTH = 64
HEIGHT = 32
RANGE = 7
COLOR_MAP = [
    (255, 255, 255),  # 0 - White
    (255, 0, 0),      # 1 - Red
    (0, 255, 0),      # 2 - Green
    (0, 0, 255),      # 3 - Blue
    (255, 255, 0),    # 4 - Yellow
    (255, 0, 255),    # 5 - Magenta
    (0, 255, 255),    # 6 - Cyan
]


@dataclasses.dataclass
class Frame:
    data: NDArray[np.int64]


class Camera:
    def __init__(self, test_mode: bool = False) -> None:
        self.test_mode = test_mode
        # Maintain square cells by deriving width from grid aspect ratio
        self.display_height = 1552
        self.display_width = int(self.display_height * WIDTH / HEIGHT)
        self.warp_height = 1552
        self.warp_width = int(self.warp_height * WIDTH / HEIGHT)
        self.capture_height = self.display_height
        self.capture_width = self.display_width
        self.locked_corners: Optional[NDArray[np.float32]] = None
        self.warp_matrix: Optional[NDArray[np.float32]] = None
        self.warp_matrix_inv: Optional[NDArray[np.float32]] = None
        self.test_camera_input: Optional[NDArray[np.uint8]] = None
        self.calibrated_colors: Optional[NDArray[np.float32]] = None

        # Temporal averaging for noise reduction
        self.prev_pixel_avg: Optional[NDArray[np.float32]] = None  # (HEIGHT, WIDTH, 3)
        self.prev_decoded_data: Optional[NDArray[np.int64]] = None  # (HEIGHT, WIDTH)

        # Calibration status
        self.receive_calibration_done = False
        self.transmit_calibration_done = False

        # Current display mode
        self.display_mode = "idle"  # idle, transmit_markers, transmit_colors, send_data
        self.display_data: Optional[Frame] = None
        self.debug_grid_overlay = False
        self.debug_window_name = 'Sampling Debug'
        self._debug_window_open = False

        if not test_mode:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
            self.capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.capture_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(
                f"Camera resolution: "
                f"{self.capture_width}x{self.capture_height}"
            )

        # Print instructions
        self._print_instructions()

        # Create unified window
        cv2.namedWindow('Camera Data Link')

    def _print_instructions(self) -> None:
        print("\n" + "="*70)
        print("CAMERA DATA LINK - Instructions")
        print("="*70)
        print("\nKEYBOARD CONTROLS:")
        print("  T - Transmit calibration (show markers + colors)")
        print("  R - Receive calibration (detect markers + colors from other side)")
        print("  S - Send random data (after both calibrations complete)")
        print("  R - Receive data (after both calibrations complete)")
        print("  Q - Quit")
        print("\nTYPICAL FLOW (two computers):")
        print("  1. Computer A: Press T (transmit calibration)")
        print("  2. Computer B: Press R (receive A's calibration)")
        print("  3. Computer B: Press T (transmit calibration)")
        print("  4. Computer A: Press R (receive B's calibration)")
        print("  5. Both computers have ✓✓ checkmarks")
        print("  6. Use S to send data, D to decode data")
        print("\nNOTE: You can press R before the other side presses T.")
        print("      It will wait until it sees the calibration pattern.")
        print("="*70 + "\n")

    def _render_instructions(self, in_data_mode: bool = False) -> NDArray[np.uint8]:
        """Renders the instructions screen"""
        height = self.display_height
        width = self.display_width
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Title
        cv2.putText(img, "CAMERA DATA LINK",
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

        y = 220

        if not in_data_mode:
            # Calibration mode
            check_rx = "[X]" if self.receive_calibration_done else "[ ]"
            check_tx = "[X]" if self.transmit_calibration_done else "[ ]"

            cv2.putText(img, "CALIBRATION STATUS:",
                       (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            y += 80
            cv2.putText(img, f"{check_rx} Receive Calibration",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            y += 70
            cv2.putText(img, f"{check_tx} Transmit Calibration",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            y += 100

            cv2.putText(img, "KEYBOARD CONTROLS:",
                       (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            y += 80
            cv2.putText(img, "T - Transmit calibration",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            y += 60
            cv2.putText(img, "R - Receive calibration",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            y += 60
            cv2.putText(img, "Q - Quit",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        else:
            # Data transmission mode
            cv2.putText(img, "DATA TRANSMISSION MODE",
                       (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 0), 2)
            y += 100

            cv2.putText(img, "KEYBOARD CONTROLS:",
                       (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            y += 80
            cv2.putText(img, "S - Send random data",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            y += 60
            cv2.putText(img, "R - Receive data",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            y += 60
            cv2.putText(img, "Q - Quit",
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        return img

    def _render_window(self, webcam_frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Renders the full window based on current display mode"""
        # Display based on mode
        if self.display_mode == "instructions":
            in_data_mode = self.receive_calibration_done and self.transmit_calibration_done
            return self._render_instructions(in_data_mode)
        elif self.display_mode == "transmit_markers":
            return self._render_calibration_boundary()
        elif self.display_mode == "transmit_colors":
            if hasattr(self, '_transmit_color_idx'):
                return self._render_color_calibration_pattern(self._transmit_color_idx)
            return self._render_instructions()
        elif self.display_mode == "receive_camera":
            # Show camera feed with native camera resolution (no stretching)
            return webcam_frame
        elif self.display_mode == "send_data":
            if self.display_data is not None:
                return self._render_data(self.display_data)
            return self._render_instructions(True)
        else:
            return self._render_instructions()

    def calibrate(self) -> bool:
        """Calibration loop - handles T/R keys until both transmit and receive are complete"""
        import time

        # Start with instructions
        self.display_mode = "instructions"

        # State for receive calibration
        rx_state = "idle"  # idle, waiting_markers, locked, receiving_colors
        rx_prev_samples = None
        rx_color_idx = 0
        rx_frames_stable = 0

        # State for transmit calibration
        tx_start_time = None

        while True:
            # Get webcam frame
            if self.test_mode:
                if self.test_camera_input is None:
                    webcam_frame = self._blank_capture_frame()
                else:
                    webcam_frame = self.test_camera_input
                    self._update_capture_dimensions(webcam_frame)
            else:
                ret, webcam_frame = self.cap.read()
                if not ret:
                    webcam_frame = self._blank_capture_frame()
                else:
                    self._update_capture_dimensions(webcam_frame)

            # Handle transmit calibration timing
            if self.display_mode == "transmit_markers":
                elapsed = time.time() - tx_start_time
                if elapsed >= 2.0:
                    # Switch to colors
                    self.display_mode = "transmit_colors"
                    self._transmit_color_idx = 0
            elif self.display_mode == "transmit_colors":
                elapsed = time.time() - tx_start_time - 2.0
                self._transmit_color_idx = min(int(elapsed), 6)
                if elapsed >= 7.0:
                    # Done transmitting - back to instructions
                    self.display_mode = "instructions"
                    self.transmit_calibration_done = True
                    print("✓ Transmit calibration complete!")

            # Handle receive calibration
            if rx_state == "waiting_markers":
                # Try to detect and lock markers
                detected = self._detect_boundary(webcam_frame)
                if detected is not None:
                    self.locked_corners = detected
                    dst = np.array([
                        [0, 0],
                        [self.warp_width - 1, 0],
                        [self.warp_width - 1, self.warp_height - 1],
                        [0, self.warp_height - 1]
                    ], dtype=np.float32)
                    matrix = cv2.getPerspectiveTransform(
                        self.locked_corners, dst)
                    self._set_warp_matrix(matrix)
                    print("✓ Markers detected and locked!")
                    rx_state = "locked"
                    self.calibrated_colors = np.zeros(
                        (HEIGHT, WIDTH, len(COLOR_MAP), 3), dtype=np.float32)
                    rx_prev_samples = None
                    rx_color_idx = 0
                    rx_frames_stable = 0
            elif rx_state == "locked":
                # Wait for markers to disappear (transition to colors)
                # Capture current samples and wait for a significant change
                curr_samples = self._capture_color_samples()

                if rx_prev_samples is None:
                    rx_prev_samples = curr_samples.copy()
                elif self._detect_color_change(rx_prev_samples, curr_samples):
                    # Markers have disappeared! Transition to receiving colors
                    rx_state = "receiving_colors"
                    rx_prev_samples = curr_samples.copy()
                    rx_frames_stable = 0
                    print("✓ Markers disappeared, capturing color patterns...")
                else:
                    rx_prev_samples = curr_samples.copy()

            elif rx_state == "receiving_colors":
                curr_samples = self._capture_color_samples()

                if rx_prev_samples is None:
                    rx_frames_stable = 0
                elif self._detect_color_change(rx_prev_samples, curr_samples):
                    rx_frames_stable = 0
                else:
                    rx_frames_stable += 1

                if rx_frames_stable == 3:  # STABILIZATION_FRAMES
                    self.calibrated_colors[:, :, rx_color_idx, :] = curr_samples
                    print(f"  ✓ Captured color {rx_color_idx}")
                    rx_color_idx += 1
                    rx_frames_stable = -1000

                    if rx_color_idx >= 7:
                        print("✓ Receive calibration complete!")
                        self.receive_calibration_done = True
                        rx_state = "idle"
                        # Back to instructions
                        self.display_mode = "instructions"

                rx_prev_samples = curr_samples.copy()

            # Check if both calibrations are complete
            if self.receive_calibration_done and self.transmit_calibration_done:
                print("\n✓✓ Calibration complete! Ready for data transmission.")
                self.display_mode = "instructions"
                return True

            # Render and display
            display = self._render_window(webcam_frame)
            cv2.imshow('Camera Data Link', display)
            self._show_sampling_debug(webcam_frame)

            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == ord('Q'):
                return False
            elif key == ord('t') or key == ord('T'):
                if not self.transmit_calibration_done:
                    print("Starting transmit calibration...")
                    self.display_mode = "transmit_markers"
                    tx_start_time = time.time()
                else:
                    print("Already completed transmit calibration")
            elif key == ord('r') or key == ord('R'):
                if not self.receive_calibration_done:
                    print("Starting receive calibration...")
                    print("Waiting for markers...")
                    self.display_mode = "receive_camera"
                    rx_state = "waiting_markers"
                else:
                    print("Already completed receive calibration")

    def _render_calibration_boundary(self) -> NDArray[np.uint8]:
        width = self.display_width
        height = self.display_height
        marker_size = int(min(width, height) * 0.15)  # 15% of smaller dimension
        padding_x = int(width * 0.1)
        padding_y = int(height * 0.1)
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)
        positions = [
            (padding_x, padding_y),
            (width - marker_size - padding_x, padding_y),
            (width - marker_size - padding_x, height - marker_size - padding_y),
            (padding_x, height - marker_size - padding_y),
        ]

        for marker_id, (x, y) in enumerate(positions):
            marker = cv2.aruco.generateImageMarker(
                aruco_dict, marker_id, marker_size)
            img[y:y+marker_size, x:x+marker_size] = cv2.cvtColor(
                marker, cv2.COLOR_GRAY2BGR)

        return img

    def _render_receiver_display(
            self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        display = frame.copy()

        if self.locked_corners is not None:
            corners = self.locked_corners.astype(np.int32)
            cv2.polylines(display, [corners], True, (255, 0, 255), 3)
            cv2.putText(
                display, "LOCKED",
                (int(corners[0][0]), int(corners[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        status_text = (
            "NOT CALIBRATED - Press SPACE to lock"
            if self.locked_corners is None
            else "CALIBRATED - Press SPACE to finish")
        status_color = (
            (0, 0, 255) if self.locked_corners is None else (0, 255, 0))
        cv2.putText(
            display, status_text,
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return display

    def _try_lock_corners(self, frame: NDArray[np.uint8]) -> bool:
        detected = self._detect_boundary(frame)
        if detected is not None:
            self.locked_corners = detected
            dst = np.array([
                [0, 0],
                [self.warp_width - 1, 0],
                [self.warp_width - 1, self.warp_height - 1],
                [0, self.warp_height - 1]
            ], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(
                self.locked_corners, dst)
            self._set_warp_matrix(matrix)
            return True
        return False

    def _detect_boundary(
            self, frame: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters())

        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )

        if ids is None or len(ids) < 4:
            return None

        marker_corners = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id < 4:
                corner_points = corners[i][0]
                marker_corners[marker_id] = corner_points[marker_id]

        if len(marker_corners) < 4:
            return None

        return np.array([
            marker_corners[0],
            marker_corners[1],
            marker_corners[2],
            marker_corners[3],
        ], dtype=np.float32)

    def _is_calibrated(self) -> bool:
        return (self.locked_corners is not None
                and self.warp_matrix is not None)

    def _compute_data_layout(self):
        """
        Returns (x_start, y_start, cell_size, data_width, data_height) for rendering.
        Ensures square cells with 10% vertical padding and centered horizontally.
        """
        pad_y = int(self.display_height * 0.1)
        usable_height = self.display_height - 2 * pad_y
        cell_size = usable_height / HEIGHT
        data_height = cell_size * HEIGHT
        data_width = cell_size * WIDTH
        pad_x = max(0, int(round((self.display_width - data_width) / 2)))
        return pad_x, pad_y, cell_size, data_width, data_height

    def _blank_capture_frame(self) -> NDArray[np.uint8]:
        return np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)

    def _update_capture_dimensions(self, frame: NDArray[np.uint8]) -> None:
        if frame is None:
            return
        height, width = frame.shape[:2]
        if height > 0 and width > 0:
            self.capture_height = height
            self.capture_width = width

    def _set_warp_matrix(self, matrix: Optional[NDArray[np.float32]]) -> None:
        if matrix is None:
            self.warp_matrix = None
            self.warp_matrix_inv = None
            return
        self.warp_matrix = matrix.astype(np.float32, copy=False)
        try:
            inv = np.linalg.inv(self.warp_matrix)
            self.warp_matrix_inv = inv.astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            self.warp_matrix_inv = None

    def _draw_sampling_overlay(self, img: NDArray[np.uint8]) -> None:
        """Draws grid lines showing sampling positions."""
        x_start, y_start, cell_size, data_width, data_height = self._compute_data_layout()
        x_end = min(self.display_width, int(round(x_start + data_width)))
        y_end = min(self.display_height, int(round(y_start + data_height)))
        color = (50, 50, 50)

        for row in range(HEIGHT + 1):
            y = int(round(y_start + row * cell_size))
            if 0 <= y < self.display_height:
                cv2.line(img, (int(x_start), y), (x_end, y), color, 1)

        for col in range(WIDTH + 1):
            x = int(round(x_start + col * cell_size))
            if 0 <= x < self.display_width:
                cv2.line(img, (x, int(y_start)), (x, y_end), color, 1)

    def _render_sampling_debug_frame(
            self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        overlay = frame.copy()
        if not self._is_calibrated() or self.warp_matrix_inv is None:
            return overlay

        cell_width = self.warp_width / WIDTH
        cell_height = self.warp_height / HEIGHT
        warp_to_cam = self.warp_matrix_inv.astype(np.float32, copy=False)

        # Draw rectangles for each cell showing the sampled area
        for row in range(HEIGHT):
            for col in range(WIDTH):
                # Get the four corners of this cell in warped space
                y1 = row * cell_height
                y2 = (row + 1) * cell_height
                x1 = col * cell_width
                x2 = (col + 1) * cell_width

                # Define the rectangle corners
                corners = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.float32).reshape(-1, 1, 2)

                # Transform to camera space
                cam_corners = cv2.perspectiveTransform(corners, warp_to_cam)

                # Convert to integer points
                pts = cam_corners.reshape(-1, 2).astype(np.int32)

                # Draw the rectangle
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        return overlay

    def _show_sampling_debug(self, frame: Optional[NDArray[np.uint8]]) -> None:
        if not self.debug_grid_overlay:
            if self._debug_window_open:
                cv2.destroyWindow(self.debug_window_name)
                self._debug_window_open = False
            return

        if frame is None:
            overlay = self._blank_capture_frame()
        else:
            overlay = self._render_sampling_debug_frame(frame)

        cv2.imshow(self.debug_window_name, overlay)
        self._debug_window_open = True

    def update_debug_window(self, frame: Optional[NDArray[np.uint8]]) -> None:
        """Public helper to sync the sampling debug overlay."""
        self._show_sampling_debug(frame)

    def _render_color_calibration_pattern(
            self, color_idx: int) -> NDArray[np.uint8]:
        """Renders a full grid with all cells showing the same color."""
        img = np.ones((self.display_height, self.display_width, 3), dtype=np.uint8) * 255

        x_start, y_start, cell_size, data_width, data_height = self._compute_data_layout()
        x_start_int = int(round(x_start))
        y_start_int = int(round(y_start))
        x_end = min(self.display_width, int(round(x_start + data_width)))
        y_end = min(self.display_height, int(round(y_start + data_height)))

        color = COLOR_MAP[color_idx]
        img[y_start_int:y_end, x_start_int:x_end] = color

        return img

    def _capture_color_samples(self) -> NDArray[np.float32]:
        """Captures BGR values for all grid positions from current frame.
        Samples ALL pixels in each cell and averages them for maximum noise reduction.
        Returns array of shape (HEIGHT, WIDTH, 3)."""
        if not self._is_calibrated():
            return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        if self.test_mode:
            frame = self.test_camera_input
            if frame is None:
                return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
            self._update_capture_dimensions(frame)
        else:
            ret, frame = self.cap.read()
            if not ret:
                return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
            self._update_capture_dimensions(frame)

        unwarped = cv2.warpPerspective(
            frame, self.warp_matrix, (self.warp_width, self.warp_height))

        cell_width = self.warp_width / WIDTH
        cell_height = self.warp_height / HEIGHT
        samples = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        for row in range(HEIGHT):
            for col in range(WIDTH):
                # Calculate pixel bounds for this cell
                y1 = int(row * cell_height)
                y2 = int((row + 1) * cell_height)
                x1 = int(col * cell_width)
                x2 = int((col + 1) * cell_width)

                # Ensure bounds are within the image
                y1 = max(0, y1)
                y2 = min(self.warp_height, y2)
                x1 = max(0, x1)
                x2 = min(self.warp_width, x2)

                # Extract all pixels in this cell and average them
                cell_pixels = unwarped[y1:y2, x1:x2]
                samples[row, col] = np.mean(cell_pixels, axis=(0, 1)).astype(np.float32)

        return samples

    def _detect_color_change(
            self,
            previous_samples: NDArray[np.float32],
            current_samples: NDArray[np.float32],
            threshold: int = 200) -> bool:
        """Detects if a major color change occurred across the grid.
        Returns True if more than 'threshold' squares changed significantly."""
        if previous_samples is None:
            return False

        # Calculate per-square color distance
        diff = np.sum((current_samples - previous_samples) ** 2, axis=2)

        # Count how many squares changed significantly (distance > 1000)
        changed_count = np.sum(diff > 1000)

        return changed_count > threshold

    def calibrate_colors_receiver(self) -> bool:
        """Automatically detects and captures color calibration.
        Monitors camera feed for color changes and captures all 7 colors.
        Returns True on success."""
        if not self._is_calibrated():
            print("Error: Must complete geometric calibration first")
            return False

        print("Starting color calibration (receiver side)...")
        print("Waiting for transmitter to show colors...")

        # Initialize calibration array: (HEIGHT, WIDTH, 7 colors, 3 BGR)
        self.calibrated_colors = np.zeros(
            (HEIGHT, WIDTH, len(COLOR_MAP), 3), dtype=np.float32)

        previous_samples = None
        current_color_idx = 0
        frames_since_change = 0
        STABILIZATION_FRAMES = 3  # Wait 3 stable frames before capturing

        while current_color_idx < len(COLOR_MAP):
            current_samples = self._capture_color_samples()

            if previous_samples is None:
                # First frame
                frames_since_change = 0
            elif self._detect_color_change(previous_samples, current_samples):
                # Change detected - reset counter
                frames_since_change = 0
            else:
                # No change - increment counter
                frames_since_change += 1

            # If we've been stable long enough, capture this color
            if frames_since_change == STABILIZATION_FRAMES:
                self.calibrated_colors[:, :, current_color_idx, :] = \
                    current_samples
                print(f"  ✓ Captured color {current_color_idx}")
                current_color_idx += 1
                # Set to -1000 so we don't capture again until next change
                frames_since_change = -1000

            previous_samples = current_samples.copy()
            cv2.waitKey(30)  # ~30ms between checks

        print("✓ Color calibration complete!")
        return True

    def send(self, frame: Frame) -> None:
        cv2.imshow('Transmitter: Data', self._render_data(frame))

    def _render_data(self, frame: Frame) -> NDArray[np.uint8]:
        img = np.ones((self.display_height, self.display_width, 3), dtype=np.uint8) * 255

        x_start, y_start, cell_size, _, _ = self._compute_data_layout()
        for row in range(HEIGHT):
            for col in range(WIDTH):
                value = frame.data[row, col]
                color = COLOR_MAP[value]
                y1 = min(self.display_height, int(round(y_start + row * cell_size)))
                y2 = min(self.display_height, int(round(y_start + (row + 1) * cell_size)))
                x1 = min(self.display_width, int(round(x_start + col * cell_size)))
                x2 = min(self.display_width, int(round(x_start + (col + 1) * cell_size)))
                img[y1:y2, x1:x2] = color
        if self.debug_grid_overlay:
            self._draw_sampling_overlay(img)
        return img

    def receive(self) -> Frame:
        if not self._is_calibrated():
            return Frame(data=np.zeros((HEIGHT, WIDTH), dtype=np.int64))

        if self.test_mode:
            frame = self.test_camera_input
            if frame is None:
                return Frame(data=np.zeros((HEIGHT, WIDTH), dtype=np.int64))
            self._update_capture_dimensions(frame)
        else:
            ret, frame = self.cap.read()
            if not ret:
                return Frame(data=np.zeros((HEIGHT, WIDTH), dtype=np.int64))
            self._update_capture_dimensions(frame)

        unwarped = cv2.warpPerspective(
            frame, self.warp_matrix, (self.warp_width, self.warp_height))

        cell_width = self.warp_width / WIDTH
        cell_height = self.warp_height / HEIGHT

        # Initialize temporal buffers on first call
        if self.prev_pixel_avg is None:
            self.prev_pixel_avg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
            self.prev_decoded_data = np.zeros((HEIGHT, WIDTH), dtype=np.int64)

        data = self.prev_decoded_data.copy()  # Start with previous frame
        current_pixel_avg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        # Use calibrated colors if available, otherwise use COLOR_MAP
        use_calibrated = self.calibrated_colors is not None

        # Threshold: only update if pixel changed by 75% or more
        # Max squared distance in BGR space: (255-0)^2 + (255-0)^2 + (255-0)^2 = 195,075
        # 75% of that is ~146,306
        CHANGE_THRESHOLD_SQ = 195075 * 0.75  # 75% of max squared distance

        for row in range(HEIGHT):
            for col in range(WIDTH):
                # Calculate pixel bounds for this cell
                y1 = int(row * cell_height)
                y2 = int((row + 1) * cell_height)
                x1 = int(col * cell_width)
                x2 = int((col + 1) * cell_width)

                # Ensure bounds are within the image
                y1 = max(0, y1)
                y2 = min(self.warp_height, y2)
                x1 = max(0, x1)
                x2 = min(self.warp_width, x2)

                # Extract all pixels in this cell and average them
                cell_pixels = unwarped[y1:y2, x1:x2]
                avg_pixel = np.mean(cell_pixels, axis=(0, 1)).astype(np.float32)
                current_pixel_avg[row, col] = avg_pixel

                # Calculate temporal difference from previous frame
                pixel_diff_sq = np.sum((avg_pixel - self.prev_pixel_avg[row, col]) ** 2)

                # Only update if change is significant (>75% different)
                if pixel_diff_sq > CHANGE_THRESHOLD_SQ:
                    # Find the closest color
                    min_dist = float('inf')
                    best_idx = 0

                    if use_calibrated:
                        # Compare to calibrated colors for this position
                        for idx in range(len(COLOR_MAP)):
                            color_ref = self.calibrated_colors[row, col, idx]
                            dist = np.sum((avg_pixel - color_ref) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                best_idx = idx
                    else:
                        # Compare to ideal COLOR_MAP
                        for idx, color in enumerate(COLOR_MAP):
                            dist = np.sum((avg_pixel - np.array(color)) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                best_idx = idx

                    data[row, col] = best_idx
                # else: keep previous decoded value (no update)

        # Update temporal buffers
        self.prev_pixel_avg = current_pixel_avg
        self.prev_decoded_data = data

        return Frame(data=data)


if __name__ == "__main__":
    cam = Camera(test_mode=False)

    # Run calibration
    if not cam.calibrate():
        print("Calibration cancelled")
        cv2.destroyAllWindows()
        cam.cap.release()
        exit(0)

    # Main data transmission loop
    print("\nEntering data transmission mode...")
    print("Press S to send random data, R to receive data, Q to quit")

    # Start with instructions
    cam.display_mode = "instructions"

    while True:
        # Get webcam frame
        ret, webcam_frame = cam.cap.read()
        if not ret:
            webcam_frame = cam._blank_capture_frame()
        else:
            cam._update_capture_dimensions(webcam_frame)

        # Render and display
        display = cam._render_window(webcam_frame)
        cv2.imshow('Camera Data Link', display)
        cam._show_sampling_debug(webcam_frame)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            # Send random data - switch to send_data mode and stay there
            data = np.random.randint(0, 7, (HEIGHT, WIDTH), dtype=np.int64)
            cam.display_data = Frame(data=data)
            cam.display_mode = "send_data"
            print(f"Sending data:\n{data}")
        elif key == ord('r') or key == ord('R'):
            # Receive data - don't change display mode
            received = cam.receive()
            print(f"Received data:\n{received.data}")

    # Cleanup
    cv2.destroyAllWindows()
    cam.cap.release()
