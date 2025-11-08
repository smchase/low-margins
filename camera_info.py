#!/usr/bin/env python3
"""
Camera Information Tool
Opens the camera and displays video dimensions and actual framerate.
"""

import cv2
import time
import sys


def main():
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n{'='*50}")
    print(f"Camera Information")
    print(f"{'='*50}")
    print(f"Resolution: {width}x{height}")
    print(f"Reported FPS: {reported_fps}")
    print(f"{'='*50}\n")

    # Variables for measuring actual FPS
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 1.0  # Update FPS every second
    last_fps_update = start_time
    current_fps = 0

    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        frame_count += 1
        current_time = time.time()

        # Calculate actual FPS
        elapsed = current_time - last_fps_update
        if elapsed >= fps_update_interval:
            current_fps = frame_count / elapsed
            frame_count = 0
            last_fps_update = current_time

            # Display FPS in terminal
            print(f"\rActual FPS: {current_fps:.2f}", end='', flush=True)

        # Display the frame with info overlay
        info_text = [
            f"Resolution: {width}x{height}",
            f"Reported FPS: {reported_fps:.2f}",
            f"Actual FPS: {current_fps:.2f}"
        ]

        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Camera Feed - Press Q to quit', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    total_time = time.time() - start_time
    print(f"\n\nSession Statistics:")
    print(f"Total runtime: {total_time:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
