import sys
from pathlib import Path
import json

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, ROWS, COLS, COLORS  # noqa: E402


def load_test_data(filename: str = "test_data.json"):
    filepath = Path(__file__).parent / filename
    with open(filepath, 'r') as f:
        test_cases = json.load(f)
    return [np.array(case, dtype=np.int64) for case in test_cases]


if __name__ == "__main__":
    cam = Camera()

    if not cam.calibrate():
        print("Calibration did not complete")
        exit(0)

    test_cases = load_test_data()
    current_test_idx = 0

    while True:
        cam.update()

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('t') or key == ord('T'):
            if current_test_idx >= len(test_cases):
                print("No more test cases available. Breaking.")
                break
            
            data = test_cases[current_test_idx]
            frame = Frame(data=data)
            cam.transmit(frame)
            print(f"Transmitting test case {current_test_idx}:\n{data}")
            current_test_idx += 1
        elif key == ord('r') or key == ord('R'):
            received = cam.receive()
            
            match_found = False
            for idx, test_case in enumerate(test_cases):
                if np.array_equal(received.data, test_case):
                    match_found = True
                    print(f"Match: True (test case {idx})")
                    break
            
            if not match_found:
                print("Match: False")
            
            print(f"Received data:\n{received.data}")
