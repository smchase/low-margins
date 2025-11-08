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
    data: NDArray[np.integer]


class Camera:
    def __init__(self, test_mode: bool = False) -> None:
        self.test_mode = test_mode
        self.warp_size = 600
        self.locked_corners: Optional[NDArray[np.float32]] = None
        self.warp_matrix: Optional[NDArray[np.float32]] = None
        self.test_camera_input: Optional[NDArray[np.uint8]] = None

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
        print("Press SPACE to lock, then SPACE again to finish")

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
                frame = cv2.flip(frame, 1)

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
                    print("\n✓ Calibration complete!")
                    return True
                else:
                    if self._try_lock_corners(frame):
                        print("\n✓ Corners locked!")
                        print(
                            f"    Corners: "
                            f"{self.locked_corners.astype(int).tolist()}")
                        print("\n    Press SPACE again to finish.")
                    else:
                        print(
                            "✗ No boundary detected. "
                            "Make sure other side is showing boundary."
                        )

    def _render_calibration_boundary(self) -> NDArray[np.uint8]:
        size = 600
        marker_size = 50
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)
        positions = [
            (0, 0),
            (size - marker_size, 0),
            (size - marker_size, size - marker_size),
            (0, size - marker_size),
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

    def send(self, frame: Frame) -> None:
        pass

    def receive(self) -> Frame:
        pass


if __name__ == "__main__":
    cam = Camera(test_mode=False)
    if cam.calibrate():
        print("\n✓ Calibration successful!")
        print("Ready for data transmission...")
    else:
        print("\n✗ Calibration failed")
