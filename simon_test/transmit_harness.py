import sys
import time
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
    print(f"\nLoaded {len(test_cases)} test cases")
    print("\n" + "="*60)
    print("Press SPACE to start transmission (will wait for odd second)")
    print("="*60)
    
    while True:
        cam.update()
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            break
    
    while int(time.time()) % 2 == 0:
        cam.update()
        cv2.waitKey(30)
    
    print(f"GO! Transmitting on odd seconds...")
    
    current_test_idx = 0
    last_second = -1
    
    while current_test_idx < len(test_cases):
        cam.update()
        
        current_time = time.time()
        current_second = int(current_time)
        
        # Transmit on odd seconds
        if current_second != last_second and current_second % 2 == 1:
            data = test_cases[current_test_idx]
            frame = Frame(data=data)
            cam.transmit(frame)
            print(f"[ODD {current_second}] Transmitted test case {current_test_idx}")
            current_test_idx += 1
            last_second = current_second
    
    if current_test_idx >= len(test_cases):
        print(f"\n✓ All {len(test_cases)} test cases transmitted!")
        print("Keep window open until receiver confirms all cases received...") 

