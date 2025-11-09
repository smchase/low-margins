"""
Test for codec + camera integration.

Workflow:
1. Both computers: Complete calibration (T to transmit, R to receive)
2. Sender: Press S for send mode
   - Shows GREEN start signal for 5 seconds
   - Then cycles through encoded grids at 2fps
3. Receiver: Press R for receive mode, then SPACE to start
   - Waits for GREEN start signal
   - Detects when start signal changes
   - Collects all grids automatically
   - Decodes and compares with expected tensor
   - Shows diff report (perfect match or error details)

Note: Both TX and RX use the same hardcoded tensor (seed=42) for verification.
Grid size: 64x64 pixels
"""
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

    # Create a HARDCODED test 16x16 FP16 tensor (same on both TX and RX)
    np.random.seed(42)  # Fixed seed for reproducibility
    test_tensor = np.random.randn(HEIGHT, WIDTH).astype(np.float16)
    print(f"Test tensor shape: {test_tensor.shape}, dtype: {test_tensor.dtype}")
    print(f"Test tensor stats: min={test_tensor.min():.4f}, max={test_tensor.max():.4f}, mean={test_tensor.mean():.4f}")

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
        receive_mode(cam, c, test_tensor)

    # Cleanup
    cv2.destroyAllWindows()
    cam.cap.release()


def send_mode(cam: Camera, grids: np.ndarray):
    """
    Sender: Display frames at 2fps.
    Sequence: green start signal (5s) -> grid[0] -> grid[1] -> ... -> grid[D-1] -> repeat
    """
    print("\n=== SEND MODE ===")
    print(f"Sending {grids.shape[0]} grids at 2fps (0.5s per frame)")
    print("Starting with 5-second GREEN start signal...")
    print("Press Q to quit")

    start_time = time.time()
    frame_interval = 0.5  # 2fps = 0.5 seconds per frame
    frame_idx = -1  # Will be set after start signal
    last_frame_time = None
    in_start_signal = True

    # Initialize with green start signal (index 3 = green)
    data = np.full((HEIGHT, WIDTH), 3, dtype=np.int64)

    while True:
        current_time = time.time()

        # Handle start signal period (5 seconds of green)
        if in_start_signal:
            elapsed = current_time - start_time
            if elapsed >= 5.0:
                # Start signal done, begin transmission
                in_start_signal = False
                last_frame_time = current_time
                frame_idx = 0
                print(f"[{current_time:.2f}] Start signal complete, beginning transmission...")
                data = grids[frame_idx].astype(np.int64)
                print(f"[{current_time:.2f}] Showing grid {frame_idx}/{grids.shape[0]-1}")
        else:
            # Check if it's time for next frame
            if current_time - last_frame_time >= frame_interval:
                frame_idx += 1
                if frame_idx >= grids.shape[0]:
                    frame_idx = 0  # Loop back to first grid
                last_frame_time = current_time

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


def receive_mode(cam: Camera, c: codec, expected_tensor: np.ndarray):
    """
    Receiver: Continuously capture frames and detect changes.
    Wait for green start signal to end, then collect grids.
    """
    print("\n=== RECEIVE MODE ===")
    print(f"Expecting {c.grids_needed()} grids")

    # Check if color calibration was done
    if cam.calibrated_colors is None:
        print("⚠️  WARNING: No color calibration detected!")
        print("   Colors may be inaccurate. Make sure you completed RECEIVE calibration")
        print("   by capturing the other computer's color pattern.")
    else:
        print("✓ Using calibrated colors for decoding")

    print("Press SPACE to start waiting for GREEN start signal, Q to quit")

    collected_grids = []
    prev_frame_data = None
    frames_stable = 0
    STABILITY_THRESHOLD = 3  # Need 3 stable frames before capturing
    CHANGE_THRESHOLD = 50  # Number of cells that must differ to detect change

    # State machine: idle -> waiting_for_start -> collecting -> done
    state = "idle"
    last_debug_print = 0

    while True:
        # Capture current frame
        received = cam.receive()
        current_data = received.data

        # State machine
        if state == "idle":
            # Waiting for user to press space
            pass

        elif state == "waiting_for_start":
            # Check if current frame is the green start signal (all 3s)
            is_start_signal = np.all(current_data == 3)

            if is_start_signal:
                # Still in start signal - print every 30 frames (~1 second)
                current_time = time.time()
                if current_time - last_debug_print > 1.0:
                    print("[WAITING] Green start signal detected, waiting for it to change...")
                    last_debug_print = current_time
                prev_frame_data = current_data.copy()
            else:
                # Start signal changed! Begin collecting
                print("✓ Start signal ended, beginning collection!")
                state = "collecting"
                collected_grids = []
                frames_stable = 0
                prev_frame_data = current_data.copy()

        elif state == "collecting":
            # Check for frame change
            if prev_frame_data is not None:
                diff_count = np.sum(current_data != prev_frame_data)

                if diff_count > CHANGE_THRESHOLD:
                    # Major change detected - new frame incoming
                    frames_stable = 0
                    unique, counts = np.unique(current_data, return_counts=True)
                    color_dist = dict(zip(unique, counts))
                    print(f"[CHANGE DETECTED] {diff_count} cells changed, colors: {color_dist}")
                else:
                    # Frame is stable
                    frames_stable += 1

                    # If stable enough, capture it
                    if frames_stable == STABILITY_THRESHOLD:
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

                                # Compare with expected tensor
                                print("\n=== COMPARISON ===")
                                # Compare as uint16 for exact bit-level comparison
                                expected_u16 = expected_tensor.view(np.uint16)
                                decoded_u16 = decoded_tensor.view(np.uint16)

                                exact_match = np.array_equal(expected_u16, decoded_u16)
                                if exact_match:
                                    print("✓✓✓ PERFECT MATCH! Transmission successful!")
                                else:
                                    diff_mask = expected_u16 != decoded_u16
                                    num_errors = np.sum(diff_mask)
                                    error_rate = num_errors / expected_u16.size * 100
                                    print(f"✗ {num_errors}/{expected_u16.size} values differ ({error_rate:.2f}% error rate)")

                                    # Show some examples of differences
                                    error_positions = np.argwhere(diff_mask)
                                    num_to_show = min(5, len(error_positions))
                                    print(f"\nFirst {num_to_show} errors:")
                                    for i in range(num_to_show):
                                        row, col = error_positions[i]
                                        exp_val = expected_tensor[row, col]
                                        got_val = decoded_tensor[row, col]
                                        print(f"  [{row},{col}] expected={exp_val:.6f}, got={got_val:.6f}")

                                # Print both tensors
                                print("\n=== EXPECTED TENSOR ===")
                                print(expected_tensor)
                                print("\n=== DECODED TENSOR ===")
                                print(decoded_tensor)

                                print("\nPress SPACE to collect again, Q to quit")
                                state = "idle"
                                collected_grids = []
                            except Exception as e:
                                print(f"✗ Decode failed: {e}")
                                import traceback
                                traceback.print_exc()
                                state = "idle"
                                collected_grids = []
                        # Reset stable counter after successful capture
                        frames_stable = 0

            prev_frame_data = current_data.copy()

        elif state == "done":
            # Just display, waiting for user input
            pass

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
            if state == "idle":
                # Start waiting for green start signal
                state = "waiting_for_start"
                collected_grids = []
                frames_stable = 0
                prev_frame_data = None
                print("\n[WAITING] Looking for GREEN start signal...")


if __name__ == "__main__":
    test_send_receive()
