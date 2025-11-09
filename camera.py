import dataclasses
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional

WIDTH = 64
HEIGHT = 32
RANGE = 8
COLOR_MAP = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


@dataclasses.dataclass
class Frame:
    data: NDArray[np.int64]


class Camera:
    def __init__(self, test_mode: bool = False) -> None:
        self.test_mode = test_mode
        # Display dimensions - 2:1 aspect ratio to match 64:32 grid
        self.display_width = 3104  # 2x base size for 2:1 ratio
        self.display_height = 1552  # base size
        # Warp dimensions match display
        self.warp_width = 3104
        self.warp_height = 1552
        self.locked_corners: Optional[NDArray[np.float32]] = None
        self.warp_matrix: Optional[NDArray[np.float32]] = None
        self.test_camera_input: Optional[NDArray[np.uint8]] = None
        self.calibrated_colors: Optional[NDArray[np.float32]] = None

        # Calibration status
        self.receive_calibration_done = False
        self.transmit_calibration_done = False

        # Current display mode
        self.display_mode = "idle"  # idle, transmit_markers, transmit_colors, send_data
        self.display_data: Optional[Frame] = None

        if not test_mode:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1552)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1552)
            print(
                f"Camera resolution: "
                f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
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
        img = np.ones((self.display_height, self.display_width, 3), dtype=np.uint8) * 255

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
            # Show camera feed - already 1000x1000
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
                    webcam_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                else:
                    webcam_frame = self.test_camera_input
            else:
                ret, webcam_frame = self.cap.read()
                if not ret:
                    webcam_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

            # Handle transmit calibration timing
            if self.display_mode == "transmit_markers":
                elapsed = time.time() - tx_start_time
                if elapsed >= 2.0:
                    # Switch to colors
                    self.display_mode = "transmit_colors"
                    self._transmit_color_idx = 0
            elif self.display_mode == "transmit_colors":
                elapsed = time.time() - tx_start_time - 2.0
                self._transmit_color_idx = min(int(elapsed), 7)
                if elapsed >= 8.0:
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
                    self.warp_matrix = cv2.getPerspectiveTransform(
                        self.locked_corners, dst)
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

                    if rx_color_idx >= 8:
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
        # Create wider image for 2:1 aspect ratio
        img = np.ones((self.display_height, self.display_width, 3), dtype=np.uint8) * 255
        marker_size = int(self.display_height * 0.15)  # 15% of height

        # Data area boundaries (same as color pattern and data rendering)
        data_start_y = int(self.display_height * 0.1)
        data_start_x = int(self.display_width * 0.1)
        data_height = int(self.display_height * 0.8)
        data_width = int(self.display_width * 0.8)

        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)

        # Position markers so their detected corners align with data area corners
        # Marker 0: top-left corner at data top-left
        # Marker 1: top-right corner at data top-right
        # Marker 2: bottom-right corner at data bottom-right
        # Marker 3: bottom-left corner at data bottom-left
        positions = [
            (data_start_x, data_start_y),  # marker 0 top-left
            (data_start_x + data_width - marker_size, data_start_y),  # marker 1 top-left
            (data_start_x + data_width - marker_size, data_start_y + data_height - marker_size),  # marker 2 top-left
            (data_start_x, data_start_y + data_height - marker_size),  # marker 3 top-left
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
            self.warp_matrix = cv2.getPerspectiveTransform(
                self.locked_corners, dst)
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

    def _render_color_calibration_pattern(
            self, color_idx: int) -> NDArray[np.uint8]:
        """Renders a full grid with all cells showing the same color."""
        img = np.ones((self.display_height, self.display_width, 3), dtype=np.uint8) * 255

        data_start_y = int(self.display_height * 0.1)  # 10% padding
        data_start_x = int(self.display_width * 0.1)  # 10% padding
        data_height = int(self.display_height * 0.8)  # 80% for data
        data_width = int(self.display_width * 0.8)  # 80% for data

        color = COLOR_MAP[color_idx]
        img[data_start_y:data_start_y+data_height,
            data_start_x:data_start_x+data_width] = color

        return img

    def _capture_color_samples(self) -> NDArray[np.float32]:
        """Captures BGR values for all grid positions from current frame.
        Returns array of shape (HEIGHT, WIDTH, 3)."""
        if not self._is_calibrated():
            return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        if self.test_mode:
            frame = self.test_camera_input
        else:
            ret, frame = self.cap.read()
            if not ret:
                return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        unwarped = cv2.warpPerspective(
            frame, self.warp_matrix, (self.warp_width, self.warp_height))

        cell_width = self.warp_width / WIDTH
        cell_height = self.warp_height / HEIGHT
        samples = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        for row in range(HEIGHT):
            for col in range(WIDTH):
                y = int((row + 0.5) * cell_height)
                x = int((col + 0.5) * cell_width)
                samples[row, col] = unwarped[y, x].astype(np.float32)

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
        Monitors camera feed for color changes and captures all 8 colors.
        Returns True on success."""
        if not self._is_calibrated():
            print("Error: Must complete geometric calibration first")
            return False

        print("Starting color calibration (receiver side)...")
        print("Waiting for transmitter to show colors...")

        # Initialize calibration array: (HEIGHT, WIDTH, 8 colors, 3 BGR)
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

        data_start_y = int(self.display_height * 0.1)  # 10% padding
        data_start_x = int(self.display_width * 0.1)  # 10% padding
        data_height = int(self.display_height * 0.8)  # 80% for data
        data_width = int(self.display_width * 0.8)  # 80% for data

        cell_width = data_width / WIDTH
        cell_height = data_height / HEIGHT
        for row in range(HEIGHT):
            for col in range(WIDTH):
                value = frame.data[row, col]
                color = COLOR_MAP[value]
                y1 = int(data_start_y + row * cell_height)
                y2 = int(data_start_y + (row + 1) * cell_height)
                x1 = int(data_start_x + col * cell_width)
                x2 = int(data_start_x + (col + 1) * cell_width)
                img[y1:y2, x1:x2] = color
        return img

    def receive(self) -> Frame:
        if not self._is_calibrated():
            return Frame(data=np.zeros((HEIGHT, WIDTH), dtype=np.int64))

        if self.test_mode:
            frame = self.test_camera_input
        else:
            ret, frame = self.cap.read()
            if not ret:
                return Frame(data=np.zeros((HEIGHT, WIDTH), dtype=np.int64))

        unwarped = cv2.warpPerspective(
            frame, self.warp_matrix, (self.warp_width, self.warp_height))

        cell_width = self.warp_width / WIDTH
        cell_height = self.warp_height / HEIGHT
        data = np.zeros((HEIGHT, WIDTH), dtype=np.int64)

        # Use calibrated colors if available, otherwise use COLOR_MAP
        use_calibrated = self.calibrated_colors is not None

        for row in range(HEIGHT):
            for col in range(WIDTH):
                y = int((row + 0.5) * cell_height)
                x = int((col + 0.5) * cell_width)
                pixel = unwarped[y, x].astype(np.float32)

                min_dist = float('inf')
                best_idx = 0

                if use_calibrated:
                    # Compare to calibrated colors for this position
                    for idx in range(len(COLOR_MAP)):
                        color_ref = self.calibrated_colors[row, col, idx]
                        dist = np.sum((pixel - color_ref) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx
                else:
                    # Compare to ideal COLOR_MAP
                    for idx, color in enumerate(COLOR_MAP):
                        dist = np.sum((pixel - np.array(color)) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx

                data[row, col] = best_idx

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
            webcam_frame = np.zeros((cam.display_height, cam.display_width, 3), dtype=np.uint8)

        # Render and display
        display = cam._render_window(webcam_frame)
        cv2.imshow('Camera Data Link', display)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            # Send random data - switch to send_data mode and stay there
            data = np.random.randint(0, 8, (HEIGHT, WIDTH), dtype=np.int64)
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
