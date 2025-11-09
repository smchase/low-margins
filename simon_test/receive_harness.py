import sys
import time
from pathlib import Path
import json

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, ROWS, COLS, COLORS, SECONDS_PER_FRAME, RECEIVE_OFFSET  # noqa: E402


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
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break

    # Wait for the next 5-second boundary (same logic as transmitter)
    # If we're currently at a 5-second boundary, wait for it to pass first
    if int(time.time()) % 5 == 0:
        current = int(time.time())
        while int(time.time()) == current:
            cam.update()
            cv2.waitKey(1)

    # Now wait for the next 5-second boundary
    while int(time.time()) % 5 != 0:
        cam.update()
        cv2.waitKey(1)

    # We're at a 5-second boundary - record start time
    start_time = time.time()
    print(f"GO! Starting reception at t={start_time:.3f} (unix second {int(start_time)})")
    if RECEIVE_OFFSET > 0:
        print(f"Will receive {RECEIVE_OFFSET}s after each frame starts")
    elif RECEIVE_OFFSET < 0:
        print(f"Will receive {-RECEIVE_OFFSET}s before each frame starts")
    else:
        print(f"Will receive exactly when each frame starts")

    frame_number = 0

    matched_count = 0
    total_pixels_wrong = 0
    total_pixels = ROWS * COLS * len(test_cases)

    results = []

    while frame_number < len(test_cases):
        cam.update()

        elapsed = time.time() - start_time
        # Receive at: frame_time + RECEIVE_OFFSET
        # Positive RECEIVE_OFFSET = after frame, Negative = before frame
        target_time = frame_number * SECONDS_PER_FRAME + RECEIVE_OFFSET
        # Ensure we don't have negative target times (clamp to 0)
        if target_time < 0:
            target_time = 0

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
            
            # Check for timing issues - did we capture the wrong frame?
            timing_info = ""
            if not matches:
                # Check if it matches the previous frame (captured too early)
                if frame_number > 0:
                    prev_expected = test_cases[frame_number - 1]
                    if np.array_equal(received.data, prev_expected):
                        timing_info = " [TIMING: EARLY - got previous frame]"
                
                # Check if it matches the next frame (captured too late)
                if not timing_info and frame_number < len(test_cases) - 1:
                    next_expected = test_cases[frame_number + 1]
                    if np.array_equal(received.data, next_expected):
                        timing_info = " [TIMING: LATE - got next frame]"

            results.append({
                'test_idx': frame_number,
                'matches': matches,
                'pixels_wrong': pixels_wrong,
                'timing_info': timing_info
            })

            print(f"[t={elapsed:.3f}s] Frame {frame_number} (target: {target_time:.3f}s): "
                  f"Match={matches}, Pixels wrong={pixels_wrong}/{ROWS*COLS}{timing_info}")

            frame_number += 1

        cv2.waitKey(1)
    
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
        timing = result.get('timing_info', '')
        print(f"  {status} Frame {result['test_idx']}: {result['pixels_wrong']} pixels wrong{timing}")
