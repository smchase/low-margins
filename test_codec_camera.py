import numpy as np
import cv2
import time
from camera import Camera, Frame, HEIGHT, WIDTH
from codec import codec


def test_send_receive():
    """
    Test sending a 16x16 FP16 tensor through the camera using codec.
    One side sends, other side continuously receives and detects frame changes.
    """
    # Initialize camera
    cam = Camera(test_mode=False)

    # Run calibration
    print("\n=== CALIBRATION PHASE ===")
    if not cam.calibrate():
        print("Calibration cancelled")
        cv2.destroyAllWindows()
        cam.cap.release()
        return

    print("\n=== CODEC SETUP ===")
    # Create codec for 16x16 tensor with range 0-7 (matching camera's 8 colors)
    c = codec(rows=HEIGHT, cols=WIDTH, min_val=0, max_val=7)
    print(f"Codec: {HEIGHT}x{WIDTH} grid, range [0,7], {c.grids_needed()} grids needed")

    # Create a test 16x16 FP16 tensor
    test_tensor = np.random.randn(HEIGHT, WIDTH).astype(np.float16)
    print(f"Test tensor shape: {test_tensor.shape}, dtype: {test_tensor.dtype}")

    # Encode the tensor
    grids = c.encode(test_tensor)
    print(f"Encoded into {grids.shape[0]} grids of shape {grids.shape[1:]} with dtype {grids.dtype}")

    print("\n=== TRANSMISSION PHASE ===")
    print("Choose mode:")
    print("  S - SEND mode (transmit the tensor)")
    print("  R - RECEIVE mode (receive the tensor)")
    print("  Q - Quit")

    # Wait for mode selection
    mode = None
    while mode is None:
        # Show instructions
        cam.display_mode = "instructions"
        if not cam.test_mode:
            ret, webcam_frame = cam.cap.read()
            if not ret:
                webcam_frame = np.zeros((cam.display_size, cam.display_size, 3), dtype=np.uint8)
        else:
            webcam_frame = np.zeros((cam.display_size, cam.display_size, 3), dtype=np.uint8)

        display = cam._render_window(webcam_frame)
        cv2.imshow('Camera Data Link', display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('s') or key == ord('S'):
            mode = 'send'
            print("Selected SEND mode")
        elif key == ord('r') or key == ord('R'):
            mode = 'receive'
            print("Selected RECEIVE mode")
        elif key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            if not cam.test_mode:
                cam.cap.release()
            return

    if mode == 'send':
        send_mode(cam, grids)
    else:
        receive_mode(cam, c)

    # Cleanup
    cv2.destroyAllWindows()
    cam.cap.release()


def send_mode(cam: Camera, grids: np.ndarray):
    """
    Sender: Display frames at 2fps.
    Sequence: empty frame -> grid[0] -> grid[1] -> ... -> grid[D-1] -> repeat
    """
    print("\n=== SEND MODE ===")
    print(f"Sending {grids.shape[0]} grids at 2fps (0.5s per frame)")
    print("Press Q to quit")

    frame_interval = 0.5  # 2fps = 0.5 seconds per frame
    frame_idx = -1  # Start with -1 for empty frame
    last_frame_time = time.time()

    # Initialize with empty frame
    data = np.ones((HEIGHT, WIDTH), dtype=np.int64)

    while True:
        current_time = time.time()

        # Check if it's time for next frame
        if current_time - last_frame_time >= frame_interval:
            frame_idx += 1
            if frame_idx >= grids.shape[0]:
                frame_idx = -1  # Loop back to empty frame
            last_frame_time = current_time

            if frame_idx == -1:
                # Empty frame (all green = color index 1)
                print(f"[{current_time:.2f}] Showing EMPTY frame (all green)")
                data = np.ones((HEIGHT, WIDTH), dtype=np.int64)
            else:
                # Show codec grid
                print(f"[{current_time:.2f}] Showing grid {frame_idx}/{grids.shape[0]-1}")
                data = grids[frame_idx].astype(np.int64)

        # Display current frame
        cam.display_data = Frame(data=data)
        cam.display_mode = "send_data"
        display = cam._render_data(cam.display_data)
        cv2.imshow('Camera Data Link', display)

        # Handle keyboard
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break


def receive_mode(cam: Camera, c: codec):
    """
    Receiver: Continuously capture frames and detect changes.
    When a change is detected, capture the stable frame.
    """
    print("\n=== RECEIVE MODE ===")
    print(f"Expecting {c.grids_needed()} grids")
    print("Continuously monitoring for frame changes...")
    print("Press SPACE to start/restart collection, Q to quit")

    collected_grids = []
    prev_frame_data = None
    frames_stable = 0
    STABILITY_THRESHOLD = 3  # Need 3 stable frames before capturing
    CHANGE_THRESHOLD = 50  # Number of cells that must differ to detect change
    collecting = False

    while True:
        # Capture current frame
        received = cam.receive()
        current_data = received.data

        # Check for frame change if collecting
        if collecting and prev_frame_data is not None:
            diff_count = np.sum(current_data != prev_frame_data)

            if diff_count > CHANGE_THRESHOLD:
                # Major change detected - new frame incoming
                frames_stable = 0
                print(f"[CHANGE DETECTED] {diff_count} cells changed")
            else:
                # Frame is stable
                frames_stable += 1

                # If stable enough and collecting, capture it
                if frames_stable == STABILITY_THRESHOLD:
                    # Check if this is not the empty frame (all ones)
                    if not np.all(current_data == 1):
                        collected_grids.append(current_data.copy())
                        print(f"✓ Captured grid {len(collected_grids)}/{c.grids_needed()}")
                        print(f"  Stats: min={current_data.min()}, max={current_data.max()}, unique={len(np.unique(current_data))}")

                        # Check if we have all grids
                        if len(collected_grids) == c.grids_needed():
                            print("\n=== DECODING ===")
                            try:
                                # Stack grids and decode
                                grids_array = np.stack(collected_grids, axis=0)
                                decoded_tensor = c.decode(grids_array)
                                print(f"✓ Successfully decoded tensor!")
                                print(f"  Shape: {decoded_tensor.shape}, dtype: {decoded_tensor.dtype}")
                                print(f"  Stats: min={decoded_tensor.min():.4f}, max={decoded_tensor.max():.4f}")
                                print("\nPress SPACE to collect again, Q to quit")
                                collecting = False
                                collected_grids = []
                            except Exception as e:
                                print(f"✗ Decode failed: {e}")
                                collected_grids = []
                                collecting = False
                        # Reset stable counter after successful capture
                        frames_stable = 0
                    else:
                        print(f"[SKIPPED] Empty frame detected (all green)")
                        frames_stable = 0

        prev_frame_data = current_data.copy()

        # Display received frame
        cam.display_data = Frame(data=current_data)
        cam.display_mode = "send_data"

        if not cam.test_mode:
            ret, webcam_frame = cam.cap.read()
            if not ret:
                webcam_frame = np.zeros((cam.display_size, cam.display_size, 3), dtype=np.uint8)
        else:
            webcam_frame = np.zeros((cam.display_size, cam.display_size, 3), dtype=np.uint8)

        display = cam._render_window(webcam_frame)
        cv2.imshow('Camera Data Link', display)

        # Handle keyboard
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting receive mode...")
            break
        elif key == ord(' '):
            # Start/restart collection
            collected_grids = []
            frames_stable = 0
            collecting = True
            print("\n[COLLECTING] Started frame collection")


if __name__ == "__main__":
    test_send_receive()
