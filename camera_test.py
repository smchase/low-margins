#!/usr/bin/env python3
import cv2
import numpy as np
import select
import sys
from numpy.typing import NDArray
from camera import Camera, Frame


def overlay_centered(
        background: NDArray[np.uint8],
        foreground: NDArray[np.uint8]) -> NDArray[np.uint8]:
    result = background.copy()
    h, w = foreground.shape[:2]
    y_offset = (background.shape[0] - h) // 2
    x_offset = (background.shape[1] - w) // 2
    result[y_offset:y_offset+h, x_offset:x_offset+w] = foreground
    return result


def test_harness() -> None:
    print("\n" + "="*60)
    print("TEST HARNESS - Two Virtual Cameras")
    print("="*60)
    print("\nSimulates two computers facing each other.")
    print("\nFlow:")
    print("  1. Press Enter to calibrate both sides")
    print("  2. Press Enter to send test pattern")
    print("  3. Press Enter to receive and decode")
    print("="*60 + "\n")

    cam_a = Camera(test_mode=True)
    cam_b = Camera(test_mode=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open webcam, using black background\n")
        cap = None
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow('Receiver Side A')
    cv2.namedWindow('Receiver Side B')

    # Show calibration patterns in loop
    print("Showing calibration patterns...")

    while True:
        if cap is not None:
            ret, webcam_frame = cap.read()
            if not ret:
                webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        transmitter_a = cam_a._render_calibration_boundary()
        transmitter_b = cam_b._render_calibration_boundary()

        receiver_a_feed = overlay_centered(webcam_frame, transmitter_b)
        receiver_b_feed = overlay_centered(webcam_frame, transmitter_a)

        cam_a.test_camera_input = receiver_a_feed
        cam_b.test_camera_input = receiver_b_feed

        cv2.imshow('Receiver Side A', receiver_a_feed)
        cv2.imshow('Receiver Side B', receiver_b_feed)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            return

        # Check if stdin has input (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break

    # Lock both sides
    print("\nLocking calibration...")
    if cam_a._try_lock_corners(receiver_a_feed):
        print(f"✓✓✓ Side A: CALIBRATED!")
        print(f"    Corners: {cam_a.locked_corners.astype(int).tolist()}")
    else:
        print(f"✗ Side A: No boundary detected")
        return

    if cam_b._try_lock_corners(receiver_b_feed):
        print(f"✓✓✓ Side B: CALIBRATED!")
        print(f"    Corners: {cam_b.locked_corners.astype(int).tolist()}")
    else:
        print(f"✗ Side B: No boundary detected")
        return

    print("\n✓✓✓ BOTH SIDES CALIBRATED!")

    # Show locked rectangles
    receiver_a_display = cam_a._render_receiver_display(receiver_a_feed)
    receiver_b_display = cam_b._render_receiver_display(receiver_b_feed)
    cv2.imshow('Receiver Side A', receiver_a_display)
    cv2.imshow('Receiver Side B', receiver_b_display)
    cv2.waitKey(1)

    # Color calibration phase
    input("\nPress Enter to start color calibration...")

    print("\n" + "="*60)
    print("COLOR CALIBRATION")
    print("="*60)

    import time

    # Initialize calibration arrays
    cam_a.calibrated_colors = np.zeros(
        (16, 16, len([0,1,2,3,4,5,6,7]), 3), dtype=np.float32)
    cam_b.calibrated_colors = np.zeros(
        (16, 16, len([0,1,2,3,4,5,6,7]), 3), dtype=np.float32)

    # Track state for both receivers
    prev_samples_a = None
    prev_samples_b = None
    color_idx_a = 0
    color_idx_b = 0
    frames_stable_a = 0
    frames_stable_b = 0
    STABILIZATION_FRAMES = 3

    # Calibrate both sides simultaneously
    print("\nCalibrating both sides...")
    start_time = time.time()

    while color_idx_a < 8 or color_idx_b < 8:
        elapsed = time.time() - start_time
        current_color = min(int(elapsed), 7)  # 0-7 based on seconds elapsed

        if cap is not None:
            ret, webcam_frame = cap.read()
            if not ret:
                webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Side A transmits, B receives
        pattern_a = cam_a._render_color_calibration_pattern(current_color)
        receiver_b_feed = overlay_centered(webcam_frame, pattern_a)
        cam_b.test_camera_input = receiver_b_feed

        # Side B transmits, A receives
        pattern_b = cam_b._render_color_calibration_pattern(current_color)
        receiver_a_feed = overlay_centered(webcam_frame, pattern_b)
        cam_a.test_camera_input = receiver_a_feed

        # Display both
        cv2.imshow('Receiver Side A', receiver_a_feed)
        cv2.imshow('Receiver Side B', receiver_b_feed)
        cv2.waitKey(30)

        # Process Side A receiver
        if color_idx_a < 8:
            curr_samples_a = cam_a._capture_color_samples()
            if prev_samples_a is None:
                frames_stable_a = 0
            elif cam_a._detect_color_change(prev_samples_a, curr_samples_a):
                frames_stable_a = 0
            else:
                frames_stable_a += 1

            if frames_stable_a == STABILIZATION_FRAMES:
                cam_a.calibrated_colors[:, :, color_idx_a, :] = curr_samples_a
                print(f"  ✓ Side A captured color {color_idx_a}")
                color_idx_a += 1
                frames_stable_a = -1000

            prev_samples_a = curr_samples_a.copy()

        # Process Side B receiver
        if color_idx_b < 8:
            curr_samples_b = cam_b._capture_color_samples()
            if prev_samples_b is None:
                frames_stable_b = 0
            elif cam_b._detect_color_change(prev_samples_b, curr_samples_b):
                frames_stable_b = 0
            else:
                frames_stable_b += 1

            if frames_stable_b == STABILIZATION_FRAMES:
                cam_b.calibrated_colors[:, :, color_idx_b, :] = curr_samples_b
                print(f"  ✓ Side B captured color {color_idx_b}")
                color_idx_b += 1
                frames_stable_b = -1000

            prev_samples_b = curr_samples_b.copy()

        # Stop after 10 seconds max
        if elapsed > 10:
            break

    if color_idx_a == 8 and color_idx_b == 8:
        print("\n✓✓✓ COLOR CALIBRATION COMPLETE!")
    else:
        print(f"\n✗ Color calibration incomplete (A: {color_idx_a}/8, B: {color_idx_b}/8)")
        return

    # Data transmission phase
    input("\nPress Enter to send test pattern...")

    test_data_a = Frame(data=np.random.randint(0, 8, (16, 16)))
    test_data_b = Frame(data=np.random.randint(0, 8, (16, 16)))

    print("\n--- Sending data ---")
    print(f"Side A sending:\n{test_data_a.data}")
    print(f"\nSide B sending:\n{test_data_b.data}")

    if cap is not None:
        ret, webcam_frame = cap.read()
        if not ret:
            webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    else:
        webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    tx_a_img = cam_a._render_data(test_data_a)
    tx_b_img = cam_b._render_data(test_data_b)

    receiver_a_feed = overlay_centered(webcam_frame, tx_b_img)
    receiver_b_feed = overlay_centered(webcam_frame, tx_a_img)

    cam_a.test_camera_input = receiver_a_feed
    cam_b.test_camera_input = receiver_b_feed

    cv2.imshow('Receiver Side A', receiver_a_feed)
    cv2.imshow('Receiver Side B', receiver_b_feed)
    cv2.waitKey(1)

    input("\nPress Enter to receive and decode...")

    print("\n--- Receiving data ---")
    received_a = cam_a.receive()
    received_b = cam_b.receive()

    print(f"Side A received:\n{received_a.data}")
    print(f"\nSide B received:\n{received_b.data}")

    print(f"\nSide A match: {np.array_equal(received_a.data, test_data_b.data)}")
    print(f"Side B match: {np.array_equal(received_b.data, test_data_a.data)}")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    test_harness()
