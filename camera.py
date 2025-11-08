import dataclasses
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional

WIDTH = 16
HEIGHT = 16
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
        self.warp_size = 600
        self.locked_corners: Optional[NDArray[np.float32]] = None
        self.warp_matrix: Optional[NDArray[np.float32]] = None
        self.test_camera_input: Optional[NDArray[np.uint8]] = None
        self.calibrated_colors: Optional[NDArray[np.float32]] = None

        if not test_mode:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print(
                f"Camera resolution: "
                f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )

    def calibrate(self) -> bool:
        print("Press SPACE to lock, then SPACE again to start color calibration")

        # Phase 1: Geometric calibration
        while True:
            if self.test_mode:
                if self.test_camera_input is None:
                    print("No test camera input")
                    return False
                frame = self.test_camera_input
            else:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    return False

            cv2.imshow(
                'Transmitter: Calibration Pattern',
                self._render_calibration_boundary()
            )
            cv2.imshow(
                'Receiver: Camera View',
                self._render_receiver_display(frame)
            )

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                if self._is_calibrated():
                    # Geometric calibration done, move to color calibration
                    break
                else:
                    if self._try_lock_corners(frame):
                        print("\n✓ Corners locked!")
                        print(
                            f"    Corners: "
                            f"{self.locked_corners.astype(int).tolist()}")
                        print("\n    Press SPACE again to start color calibration.")
                    else:
                        print(
                            "✗ No boundary detected. "
                            "Make sure other side is showing boundary."
                        )

        # Phase 2: Color calibration
        print("\n" + "="*60)
        print("COLOR CALIBRATION")
        print("="*60)
        print("Starting automatic color calibration (both sides)...")
        print("This will take ~8 seconds...")

        import time

        # Initialize calibration array
        self.calibrated_colors = np.zeros(
            (HEIGHT, WIDTH, len(COLOR_MAP), 3), dtype=np.float32)

        # Track receiver state
        prev_samples = None
        color_idx = 0
        frames_stable = 0
        STABILIZATION_FRAMES = 3

        start_time = time.time()

        while color_idx < 8:
            elapsed = time.time() - start_time
            current_color = min(int(elapsed), 7)

            # Read camera frame
            if self.test_mode:
                frame = self.test_camera_input
            else:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    return False

            # Transmit current color
            pattern = self._render_color_calibration_pattern(current_color)
            cv2.imshow('Transmitter: Calibration Pattern', pattern)
            cv2.imshow('Receiver: Camera View', frame)
            cv2.waitKey(30)

            # Receive and detect
            curr_samples = self._capture_color_samples()
            if prev_samples is None:
                frames_stable = 0
            elif self._detect_color_change(prev_samples, curr_samples):
                frames_stable = 0
            else:
                frames_stable += 1

            if frames_stable == STABILIZATION_FRAMES:
                self.calibrated_colors[:, :, color_idx, :] = curr_samples
                print(f"  ✓ Captured color {color_idx}")
                color_idx += 1
                frames_stable = -1000

            prev_samples = curr_samples.copy()

            # Timeout after 10 seconds
            if elapsed > 10:
                break

        if color_idx == 8:
            print("\n✓ Calibration complete!")
            return True
        else:
            print(f"\n✗ Calibration incomplete ({color_idx}/8 colors)")
            return False

    def _render_calibration_boundary(self) -> NDArray[np.uint8]:
        size = 600
        marker_size = 100
        padding = 50
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)
        positions = [
            (padding, padding),
            (size - marker_size - padding, padding),
            (size - marker_size - padding, size - marker_size - padding),
            (padding, size - marker_size - padding),
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
                [self.warp_size - 1, 0],
                [self.warp_size - 1, self.warp_size - 1],
                [0, self.warp_size - 1]
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
        size = 600
        img = np.ones((size, size, 3), dtype=np.uint8) * 255

        data_start = 50
        data_size = 500

        color = COLOR_MAP[color_idx]
        img[data_start:data_start+data_size,
            data_start:data_start+data_size] = color

        return img

    def _capture_color_samples(self) -> NDArray[np.float32]:
        """Captures BGR values for all 256 grid positions from current frame.
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
            frame, self.warp_matrix, (self.warp_size, self.warp_size))

        cell_size = self.warp_size / WIDTH
        samples = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

        for row in range(HEIGHT):
            for col in range(WIDTH):
                y = int((row + 0.5) * cell_size)
                x = int((col + 0.5) * cell_size)
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
        size = 600
        img = np.ones((size, size, 3), dtype=np.uint8) * 255

        data_start = 50
        data_size = 500

        cell_size = data_size / WIDTH
        for row in range(HEIGHT):
            for col in range(WIDTH):
                value = frame.data[row, col]
                color = COLOR_MAP[value]
                y1 = int(data_start + row * cell_size)
                y2 = int(data_start + (row + 1) * cell_size)
                x1 = int(data_start + col * cell_size)
                x2 = int(data_start + (col + 1) * cell_size)
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
            frame, self.warp_matrix, (self.warp_size, self.warp_size))

        cell_size = self.warp_size / WIDTH
        data = np.zeros((HEIGHT, WIDTH), dtype=np.int64)

        # Use calibrated colors if available, otherwise use COLOR_MAP
        use_calibrated = self.calibrated_colors is not None

        for row in range(HEIGHT):
            for col in range(WIDTH):
                y = int((row + 0.5) * cell_size)
                x = int((col + 0.5) * cell_size)
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
    if cam.calibrate():
        print("\n✓ Calibration successful!")
        print("Ready for data transmission...")
    else:
        print("\n✗ Calibration failed")
