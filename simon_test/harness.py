import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, ROWS, COLS, COLORS  # noqa: E402


if __name__ == "__main__":
    cam = Camera()

    if not cam.calibrate():
        print("Calibration did not complete")
        exit(0)

    while True:
        cam.update()

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('t') or key == ord('T'):
            data = np.random.randint(0, len(COLORS), (ROWS, COLS), dtype=np.int64)
            frame = Frame(data=data)
            cam.transmit(frame)
            print(f"Transmitting data:\n{data}")
        elif key == ord('r') or key == ord('R'):
            received = cam.receive()
            print(f"Received data:\n{received.data}")
