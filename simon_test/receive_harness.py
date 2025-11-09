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
    print("Press SPACE to start reception (will wait for even second)")
    print("="*60)
    
    # Wait for user to press space
    started = False
    while not started:
        cam.update()
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            started = True
            print("Starting... waiting for sync with transmitter")
        elif key == ord('q') or key == ord('Q'):
            print("Exiting...")
            exit(0)
    
    # Wait for the next ODD second (same as transmitter)
    while int(time.time()) % 2 == 0:
        cam.update()
        cv2.waitKey(30)
    
    print("On odd second boundary... waiting 1 more second for even")
    
    # Now wait for the next EVEN second (transmitter will have just sent on the odd second)
    while int(time.time()) % 2 == 1:
        cam.update()
        cv2.waitKey(30)
    
    print(f"GO! Receiving on even seconds...")
    
    current_test_idx = 0
    last_second = -1
    
    matched_count = 0
    total_pixels_wrong = 0
    total_pixels = ROWS * COLS * len(test_cases)
    
    results = []
    
    while current_test_idx < len(test_cases):
        cam.update()
        
        current_time = time.time()
        current_second = int(current_time)
        
        # Receive on even seconds
        if current_second != last_second and current_second % 2 == 0:
            received = cam.receive()
            expected = test_cases[current_test_idx]
            
            # Check if they match
            matches = np.array_equal(received.data, expected)
            if matches:
                matched_count += 1
            
            # Count wrong pixels
            pixels_wrong = np.sum(received.data != expected)
            total_pixels_wrong += pixels_wrong
            
            results.append({
                'test_idx': current_test_idx,
                'matches': matches,
                'pixels_wrong': pixels_wrong
            })
            
            print(f"[EVEN {current_second}] Test case {current_test_idx}: "
                  f"Match={matches}, Pixels wrong={pixels_wrong}/{ROWS*COLS}")
            
            current_test_idx += 1
            last_second = current_second
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\nExiting early...")
            break
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Test cases received: {current_test_idx}")
    print(f"Matched: {matched_count}/{current_test_idx}")
    print(f"Failed: {current_test_idx - matched_count}/{current_test_idx}")
    print(f"\nTotal pixels wrong: {total_pixels_wrong}")
    print(f"Total pixels checked: {ROWS * COLS * current_test_idx}")
    
    if current_test_idx > 0:
        accuracy = 100.0 * (1 - total_pixels_wrong / (ROWS * COLS * current_test_idx))
        print(f"Pixel accuracy: {accuracy:.2f}%")
    
    print("\nDetailed results:")
    for result in results:
        status = "✓" if result['matches'] else "✗"
        print(f"  {status} Test {result['test_idx']}: {result['pixels_wrong']} pixels wrong")
    
    print("\nPress Q to exit...")
    while True:
        cam.update()
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

