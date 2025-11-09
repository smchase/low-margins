import sys
import time
from pathlib import Path
import json

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, ROWS, COLS, COLORS, SECONDS_PER_FRAME  # noqa: E402


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
    print(f"Frame rate: {1/SECONDS_PER_FRAME:.2f} fps ({SECONDS_PER_FRAME}s per frame)")
    print("\n" + "="*60)
    print("Press SPACE to start transmission (will sync to 5-second boundary)")
    print("="*60)

    while True:
        cam.update()
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            break

    # Wait for the next 5-second boundary
    # If we're currently at a 5-second boundary, wait for it to pass first
    if int(time.time()) % 5 == 0:
        current = int(time.time())
        while int(time.time()) == current:
            cam.update()
            cv2.waitKey(30)

    # Now wait for the next 5-second boundary
    while int(time.time()) % 5 != 0:
        cam.update()
        cv2.waitKey(30)

    # We're at a 5-second boundary - record start time
    start_time = time.time()
    print(f"GO! Starting transmission at t={start_time:.3f} (unix second {int(start_time)})")

    frame_number = 0

    while frame_number < len(test_cases):
        cam.update()

        elapsed = time.time() - start_time
        target_time = frame_number * SECONDS_PER_FRAME

        # Transmit when we've reached or passed the target time for this frame
        if elapsed >= target_time:
            data = test_cases[frame_number]
            frame = Frame(data=data)
            cam.transmit(frame)
            print(f"[t={elapsed:.3f}s] Transmitted frame {frame_number} (target: {target_time:.3f}s)")
            frame_number += 1

        cv2.waitKey(30)
    
    if frame_number >= len(test_cases):
        print(f"\n✓ All {len(test_cases)} test cases transmitted!")
        print(f"Waiting {SECONDS_PER_FRAME * 2:.1f}s for receiver to capture last frame...")

        # Wait for 2 frame periods to ensure receiver gets the last frame
        end_time = time.time() + (SECONDS_PER_FRAME * 2)
        while time.time() < end_time:
            cam.update()
            cv2.waitKey(30)

        print("Done!") 

