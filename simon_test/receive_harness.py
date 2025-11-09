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
    print("Press SPACE to start reception (will sync to 5-second boundary)")
    print("="*60)

    while True:
        cam.update()
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            break

    # Wait for the next 5-second boundary (same logic as transmitter)
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
    print(f"GO! Starting reception at t={start_time:.3f} (unix second {int(start_time)})")
    print(f"Will receive at midpoint of each frame period")

    frame_number = 0

    matched_count = 0
    total_pixels_wrong = 0
    total_pixels = ROWS * COLS * len(test_cases)

    results = []

    while frame_number < len(test_cases):
        cam.update()

        elapsed = time.time() - start_time
        # Receive at the midpoint of each frame period
        target_time = (frame_number + 0.5) * SECONDS_PER_FRAME

        # Receive when we've reached or passed the target time for this frame
        if elapsed >= target_time:
            received = cam.receive()
            expected = test_cases[frame_number]

            # Check if they match
            matches = np.array_equal(received.data, expected)
            if matches:
                matched_count += 1

            # Count wrong pixels
            pixels_wrong = np.sum(received.data != expected)
            total_pixels_wrong += pixels_wrong

            results.append({
                'test_idx': frame_number,
                'matches': matches,
                'pixels_wrong': pixels_wrong
            })

            print(f"[t={elapsed:.3f}s] Frame {frame_number} (target: {target_time:.3f}s): "
                  f"Match={matches}, Pixels wrong={pixels_wrong}/{ROWS*COLS}")

            frame_number += 1

        cv2.waitKey(30)
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Test cases received: {frame_number}")
    print(f"Matched: {matched_count}/{frame_number}")
    print(f"Failed: {frame_number - matched_count}/{frame_number}")
    print(f"\nTotal pixels wrong: {total_pixels_wrong}")
    print(f"Total pixels checked: {ROWS * COLS * frame_number}")

    if frame_number > 0:
        accuracy = 100.0 * (1 - total_pixels_wrong / (ROWS * COLS * frame_number))
        print(f"Pixel accuracy: {accuracy:.2f}%")

    print("\nDetailed results:")
    for result in results:
        status = "✓" if result['matches'] else "✗"
        print(f"  {status} Frame {result['test_idx']}: {result['pixels_wrong']} pixels wrong")
