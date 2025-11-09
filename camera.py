import dataclasses
import enum
import time
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional

ROWS = 30
COLS = 50
SQUARE_SIZE = 20
PADDING = 25

DATA_WIDTH = COLS * SQUARE_SIZE
DATA_HEIGHT = ROWS * SQUARE_SIZE
WINDOW_WIDTH = DATA_WIDTH + 2 * PADDING
WINDOW_HEIGHT = DATA_HEIGHT + 2 * PADDING

COLORS = [
    (0, 0, 0),       # 000 - black
    (0, 0, 255),     # 001 - blue
    (0, 255, 0),     # 010 - green
    (0, 255, 255),   # 011 - cyan
    (255, 0, 0),     # 100 - red
    (255, 0, 255),   # 101 - magenta
    (255, 255, 0),   # 110 - yellow
    (255, 255, 255), # 111 - white
]
SECONDS_PER_FRAME = 0.15


class CalibrationState(enum.Enum):
    INSTRUCTIONS = "instructions"
    TRANSMIT_CALIBRATION = "transmit_calibration"
    RECEIVE_WAITING_MARKERS = "receive_waiting_markers"
    RECEIVE_COLORS = "receive_colors"


@dataclasses.dataclass
class Frame:
    data: NDArray[np.int64]


class Camera:
    def __init__(self) -> None:
        self.warp_matrix: Optional[NDArray[np.float32]] = None
        self.calibrated_colors: NDArray[np.float32] = np.zeros((ROWS, COLS, len(COLORS), 3), dtype=np.float32)
        self.curent_transmission: Optional[Frame] = None

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cv2.namedWindow('low margins', cv2.WINDOW_AUTOSIZE)

    def calibrate(self) -> bool:
        state = CalibrationState.INSTRUCTIONS

        receive_calibration_done = False
        transmit_calibration_done = False
        transmit_start_time = None

        receiving_color_index = 0
        receiving_frames_stable = 0
        receiving_prev_frame = None

        while True:
            if receive_calibration_done and transmit_calibration_done:
                print(
                    "\n✓✓ Calibration complete! Ready for data transmission."
                )
                return True

            # State machine
            match state:
                case CalibrationState.INSTRUCTIONS:
                    display = self._render_instructions(
                        receive_calibration_done, transmit_calibration_done)
                    cv2.imshow('low margins', display)

                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('t') or key == ord('T'):
                        if not transmit_calibration_done:
                            print("Starting transmit calibration...")
                            state = CalibrationState.TRANSMIT_CALIBRATION
                            transmit_start_time = time.time()
                        else:
                            print("Already completed transmit calibration")
                    elif key == ord('r') or key == ord('R'):
                        if not receive_calibration_done:
                            print("Starting receive calibration...")
                            print("Waiting for markers...")
                            state = CalibrationState.RECEIVE_WAITING_MARKERS
                        else:
                            print("Already completed receive calibration")

                case CalibrationState.TRANSMIT_CALIBRATION:
                    elapsed = time.time() - transmit_start_time
                    if elapsed >= 2 + len(COLORS):
                        state = CalibrationState.INSTRUCTIONS
                        transmit_calibration_done = True
                        print("✓ Transmit calibration complete!")

                    display = self._render_transmit_calibration(elapsed)
                    cv2.imshow('low margins', display)
                    cv2.waitKey(30)

                case CalibrationState.RECEIVE_WAITING_MARKERS:
                    ret, webcam_frame = self.cap.read()
                    if not ret:
                        print("ERROR: Webcam failed to capture frame")
                        return False

                    detected = self._detect_boundary(webcam_frame)
                    if detected is not None:
                        dst = np.array([
                            [0, 0],
                            [DATA_WIDTH - 1, 0],
                            [DATA_WIDTH - 1, DATA_HEIGHT - 1],
                            [0, DATA_HEIGHT - 1]
                        ], dtype=np.float32)
                        self.warp_matrix = cv2.getPerspectiveTransform(detected, dst)
                        state = CalibrationState.RECEIVE_COLORS
                        receiving_frames_stable = 4
                        print("✓ Markers detected and locked!")
                        print("Waiting for color patterns...")

                    cv2.imshow('low margins', webcam_frame)
                    cv2.waitKey(30)

                case CalibrationState.RECEIVE_COLORS:
                    ret, webcam_frame = self.cap.read()
                    if not ret:
                        print("ERROR: Webcam failed to capture frame")
                        return False

                    curr_samples = self._capture_color_samples()

                    if self._detect_color_change(receiving_prev_frame, curr_samples):
                        receiving_frames_stable = 0
                    else:
                        receiving_frames_stable += 1

                    if receiving_frames_stable == 3:
                        self.calibrated_colors[:, :, receiving_color_index, :] = curr_samples
                        print(f"  ✓ Captured color {receiving_color_index}")
                        receiving_color_index += 1

                        if receiving_color_index >= len(COLORS):
                            print("✓ Receive calibration complete!")
                            receive_calibration_done = True
                            state = CalibrationState.INSTRUCTIONS

                    receiving_prev_frame = curr_samples

                    cv2.imshow('low margins', webcam_frame)
                    cv2.waitKey(30)

                case _:
                    raise RuntimeError(
                        f"Unexpected state during calibration: {state}")

    def _render_instructions(
            self,
            receive_done: bool = False,
            transmit_done: bool = False) -> NDArray[np.uint8]:
        img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        y = 50

        rx = "X" if receive_done else " "
        tx = "X" if transmit_done else " "
        cv2.putText(img, f"[{rx}] Receive  [{tx}] Transmit", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y += 80
        cv2.putText(img, "T - Transmit calibration", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y += 40
        cv2.putText(img, "R - Receive calibration", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return img

    def _render_transmit_calibration(self, elapsed: float) -> NDArray[np.uint8]:
        img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255

        if elapsed < 2.0:
            # Render markers
            marker_size = 100
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            positions = [
                (PADDING, PADDING),
                (WINDOW_WIDTH - marker_size - PADDING, PADDING),
                (WINDOW_WIDTH - marker_size - PADDING, WINDOW_HEIGHT - marker_size - PADDING),
                (PADDING, WINDOW_HEIGHT - marker_size - PADDING),
            ]
            for marker_id, (x, y) in enumerate(positions):
                marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
                img[y:y+marker_size, x:x+marker_size] = cv2.cvtColor(
                    marker, cv2.COLOR_GRAY2BGR)
        else:
            # Render color pattern
            color_idx = min(int(elapsed - 2.0), len(COLORS) - 1)
            color = COLORS[color_idx]
            img[PADDING:PADDING+DATA_HEIGHT, PADDING:PADDING+DATA_WIDTH] = color

        return img

    def _detect_boundary(
            self, frame: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters())

        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )

        if ids is None or len(ids) != 4:
            return None

        marker_corners = {}
        for i, marker_id in enumerate(ids.flatten()):
            corner_points = corners[i][0]
            # ArUco corners are in order: top-left, top-right, bottom-right, bottom-left
            # We want the corner of each marker that corresponds to its position:
            # marker 0 (top-left): use top-left corner [0]
            # marker 1 (top-right): use top-right corner [1]
            # marker 2 (bottom-right): use bottom-right corner [2]
            # marker 3 (bottom-left): use bottom-left corner [3]
            if 0 <= marker_id <= 3:
                marker_corners[marker_id] = corner_points[marker_id]
            else:
                return None

        return np.array([
            marker_corners[0],
            marker_corners[1],
            marker_corners[2],
            marker_corners[3],
        ], dtype=np.float32)

    def _capture_color_samples(self) -> NDArray[np.float32]:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Webcam failed to capture frame")

        unwarped = cv2.warpPerspective(
            frame, self.warp_matrix, (DATA_WIDTH, DATA_HEIGHT))

        samples = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        for row in range(ROWS):
            for col in range(COLS):
                y = int((row + 0.5) * SQUARE_SIZE)
                x = int((col + 0.5) * SQUARE_SIZE)
                samples[row, col] = unwarped[y, x].astype(np.float32)

        return samples

    def _detect_color_change(
            self,
            previous_samples: Optional[NDArray[np.float32]],
            current_samples: NDArray[np.float32]) -> bool:
        if previous_samples is None:
            return False

        diff = np.sum((current_samples - previous_samples) ** 2, axis=2)
        changed_count = np.sum(diff > 1000)
        return changed_count > 200

    def update(self) -> None:
        if self.curent_transmission is not None:
            img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255

            for row in range(ROWS):
                for col in range(COLS):
                    value = self.curent_transmission.data[row, col]
                    color = COLORS[value]
                    y1 = PADDING + row * SQUARE_SIZE
                    y2 = PADDING + (row + 1) * SQUARE_SIZE
                    x1 = PADDING + col * SQUARE_SIZE
                    x2 = PADDING + (col + 1) * SQUARE_SIZE
                    img[y1:y2, x1:x2] = color
            display = img
        else:
            display = self._render_instructions(True, True)
        cv2.imshow('low margins', display)

    def transmit(self, frame: Frame) -> None:
        self.curent_transmission = frame

    def receive(self) -> Frame:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Webcam failed to capture frame")

        unwarped = cv2.warpPerspective(
            frame, self.warp_matrix, (DATA_WIDTH, DATA_HEIGHT))

        data = np.zeros((ROWS, COLS), dtype=np.int64)
        for row in range(ROWS):
            for col in range(COLS):
                y = int((row + 0.5) * SQUARE_SIZE)
                x = int((col + 0.5) * SQUARE_SIZE)
                pixel = unwarped[y, x].astype(np.float32)

                min_dist = float('inf')
                best_color_index = 0
                for i in range(len(COLORS)):
                    color_ref = self.calibrated_colors[row, col, i]
                    dist = np.sum((pixel - color_ref) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_color_index = i

                data[row, col] = best_color_index

        return Frame(data=data)
