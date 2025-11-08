#!/usr/bin/env python3
import cv2
import numpy as np
from numpy.typing import NDArray
from camera import Camera


def overlay_centered(
        background: NDArray[np.uint8],
        foreground: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Overlay foreground image centered on background."""
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
    print("\nControls:")
    print("  - Click window to select active side")
    print("  - Press SPACE to lock corners on active side")
    print("  - Press 'q' to quit")
    print("="*60 + "\n")

    cam_a = Camera(test_mode=True)
    cam_b = Camera(test_mode=True)

    # Try to open webcam for background
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open webcam, using black background\n")
        cap = None
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Track which side is active
    active_side = 'a'

    def set_active_a(event, x, y, flags, param):
        nonlocal active_side
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
            if active_side != 'a':
                active_side = 'a'
                print("Active: Side A")

    def set_active_b(event, x, y, flags, param):
        nonlocal active_side
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
            if active_side != 'b':
                active_side = 'b'
                print("Active: Side B")

    # Create windows and set callbacks
    cv2.namedWindow('Transmitter Side A')
    cv2.namedWindow('Receiver Side A')
    cv2.namedWindow('Transmitter Side B')
    cv2.namedWindow('Receiver Side B')

    cv2.setMouseCallback('Transmitter Side A', set_active_a)
    cv2.setMouseCallback('Receiver Side A', set_active_a)
    cv2.setMouseCallback('Transmitter Side B', set_active_b)
    cv2.setMouseCallback('Receiver Side B', set_active_b)

    print("Both sides ready. Click on a window to select it.\n")

    while True:
        # Get webcam frame or black background
        if cap is not None:
            ret, webcam_frame = cap.read()
            if not ret:
                webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            webcam_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Get transmitter outputs
        transmitter_a = cam_a._render_calibration_boundary()
        transmitter_b = cam_b._render_calibration_boundary()

        # Each camera sees the other's transmitter overlaid on webcam
        receiver_a_feed = overlay_centered(webcam_frame, transmitter_b)
        receiver_b_feed = overlay_centered(webcam_frame, transmitter_a)

        # Update camera inputs
        cam_a.test_camera_input = receiver_a_feed
        cam_b.test_camera_input = receiver_b_feed

        # Create displays with calibration overlays
        receiver_a_display = cam_a._render_receiver_display(receiver_a_feed)
        receiver_b_display = cam_b._render_receiver_display(receiver_b_feed)

        # Add active indicator
        if active_side == 'a':
            cv2.putText(
                transmitter_a, "ACTIVE",
                (transmitter_a.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(
                receiver_a_display, "ACTIVE",
                (receiver_a_display.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(
                transmitter_b, "ACTIVE",
                (transmitter_b.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(
                receiver_b_display, "ACTIVE",
                (receiver_b_display.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display all windows
        cv2.imshow('Transmitter Side A', transmitter_a)
        cv2.imshow('Receiver Side A', receiver_a_display)
        cv2.imshow('Transmitter Side B', transmitter_b)
        cv2.imshow('Receiver Side B', receiver_b_display)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord(' '):
            # Try to lock corners on active camera
            cam = cam_a if active_side == 'a' else cam_b
            other_cam = cam_b if active_side == 'a' else cam_a
            feed = receiver_a_feed if active_side == 'a' else receiver_b_feed
            side_name = "Side A" if active_side == 'a' else "Side B"

            if not cam._is_calibrated():
                if cam._try_lock_corners(feed):
                    print(f"✓✓✓ {side_name}: CALIBRATED!")
                    print(
                        f"    Corners: "
                        f"{cam.locked_corners.astype(int).tolist()}")
                    if not other_cam._is_calibrated():
                        print("    Now calibrate the other side.")
                else:
                    print(f"✗ {side_name}: No boundary detected")
            else:
                print(f"{side_name} already calibrated")

            # Check if both are done
            if cam_a._is_calibrated() and cam_b._is_calibrated():
                print("\n✓✓✓ BOTH SIDES CALIBRATED!")
                print(
                    f"Side A corners: "
                    f"{cam_a.locked_corners.astype(int).tolist()}")
                print(
                    f"Side B corners: "
                    f"{cam_b.locked_corners.astype(int).tolist()}")
                print("\nCalibration complete. Press 'q' to quit.")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_harness()
