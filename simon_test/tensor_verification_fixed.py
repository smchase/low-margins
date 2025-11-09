import sys
import time
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, ROWS, COLS, SECONDS_PER_FRAME, RECEIVE_OFFSET  # noqa: E402
from simon_test.tensor_verification_test import (  # noqa: E402
    generate_deterministic_tensor, tensor_to_frames_simple, frames_to_tensor_simple,
    apply_transform, verify_tensors, TENSOR_SIZE, FRAMES_PER_TENSOR, 
    ROOT_SEED, CYCLE_TIME
)


def log_status(message: str):
    """Log status with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def transmit_frames_properly(cam: Camera, frames: list[Frame], phase_start_time: float):
    """Transmit frames ensuring each is displayed properly."""
    for i, frame in enumerate(frames):
        # Wait for the right time to transmit this frame
        target_transmit_time = i * SECONDS_PER_FRAME
        
        # Transmit the frame
        cam.transmit(frame)
        log_status(f"  Transmitted frame {i+1}/{len(frames)}")
        
        # Keep displaying this frame until it's time for the next one
        next_frame_time = (i + 1) * SECONDS_PER_FRAME
        while (time.time() - phase_start_time) < next_frame_time:
            cam.update()  # This keeps the frame visible!
            cv2.waitKey(10)


def receive_frames_properly(cam: Camera, phase_start_time: float) -> list[Frame]:
    """Receive frames with proper timing."""
    received_frames = []
    
    for i in range(FRAMES_PER_TENSOR):
        # Wait with configured offset for frame to be displayed
        target_time = i * SECONDS_PER_FRAME + RECEIVE_OFFSET
        while (time.time() - phase_start_time) < target_time:
            cam.update()
            cv2.waitKey(10)
        
        # Capture frame
        frame = cam.receive()
        received_frames.append(frame)
        log_status(f"  Received frame {i+1}/{FRAMES_PER_TENSOR}")
    
    return received_frames


def wait_until(cam: Camera, target_time: float):
    """Wait until target time while keeping camera updated."""
    while time.time() < target_time:
        cam.update()
        cv2.waitKey(30)


def test_root_node():
    """Root node test - fixed transmission."""
    cam = Camera()
    
    log_status("=== TENSOR VERIFICATION TEST (FIXED) - ROOT NODE ===")
    log_status("Testing with proper frame display timing")
    
    if not cam.calibrate():
        log_status("ERROR: Calibration failed")
        return
    
    log_status("Calibration complete. Press SPACE to start...")
    
    while True:
        cam.update()
        if (cv2.waitKey(30) & 0xFF) == ord(' '):
            break
    
    # Wait for next cycle boundary
    current_time = time.time()
    next_cycle = ((int(current_time) // CYCLE_TIME) + 1) * CYCLE_TIME
    wait_time = next_cycle - current_time
    log_status(f"Starting in {wait_time:.1f} seconds...")
    
    wait_until(cam, next_cycle)
    
    cycle_start = time.time()
    cycle_num = 0
    
    # Generate deterministic tensor for this cycle
    original_tensor = generate_deterministic_tensor(ROOT_SEED, cycle_num)
    log_status(f"Generated tensor: min={np.min(original_tensor):.3f}, max={np.max(original_tensor):.3f}, mean={np.mean(original_tensor):.3f}")
    
    # What we expect to receive back (transformed)
    expected_response = apply_transform(original_tensor)
    
    # Phase 1: Compute (0-0.5s)
    log_status("Phase 1: Compute")
    wait_until(cam, cycle_start + 0.5)
    
    # Phase 2: Transmit to worker (0.5-5.5s)
    log_status("Phase 2: Transmitting to worker...")
    frames = tensor_to_frames_simple(original_tensor)
    phase_start = time.time()
    transmit_frames_properly(cam, frames, phase_start)
    
    # Wait for phase end
    wait_until(cam, cycle_start + 5.5)
    
    # Phase 3: Worker compute (5.5-6s)
    log_status("Phase 3: Worker computing...")
    wait_until(cam, cycle_start + 6.0)
    
    # Phase 4: Receive from worker (6-11s)
    log_status("Phase 4: Receiving from worker...")
    phase_start = time.time()
    received_frames = receive_frames_properly(cam, phase_start)
    
    # Decode received tensor
    try:
        received_tensor = frames_to_tensor_simple(received_frames)
        log_status(f"Decoded tensor: min={np.min(received_tensor):.3f}, max={np.max(received_tensor):.3f}, mean={np.mean(received_tensor):.3f}")
        
        # Verify against expected
        matches, num_diff, max_error = verify_tensors(expected_response, received_tensor)
        
        log_status("\n" + "="*60)
        log_status("VERIFICATION RESULTS:")
        log_status("="*60)
        log_status(f"Original tensor sent: shape={original_tensor.shape}, mean={np.mean(original_tensor):.6f}")
        log_status(f"Expected response:    shape={expected_response.shape}, mean={np.mean(expected_response):.6f}")
        log_status(f"Received tensor:      shape={received_tensor.shape}, mean={np.mean(received_tensor):.6f}")
        log_status("")
        
        if matches:
            log_status("✓ SUCCESS: Received tensor matches expected transform!")
        else:
            log_status(f"✗ FAILURE: {num_diff}/{TENSOR_SIZE} values differ")
            log_status(f"  Maximum error: {max_error:.6f}")
            
            # Show examples of mismatches
            if num_diff > 0 and num_diff <= 10:
                diff_indices = np.where(~np.isclose(expected_response, received_tensor, rtol=1e-3, atol=1e-3))[0]
                log_status("\n  Examples of mismatches:")
                for i in range(min(5, len(diff_indices))):
                    idx = diff_indices[i]
                    log_status(f"    [{idx}] Expected: {expected_response[idx]:.6f}, Received: {received_tensor[idx]:.6f}")
        
    except Exception as e:
        log_status(f"✗ ERROR decoding frames: {e}")
    
    # Keep displaying
    time.sleep(2)
    cv2.destroyAllWindows()


def test_worker_node():
    """Worker node test - fixed transmission."""
    cam = Camera()
    
    log_status("=== TENSOR VERIFICATION TEST (FIXED) - WORKER NODE ===")
    log_status("Testing with proper frame display timing")
    
    if not cam.calibrate():
        log_status("ERROR: Calibration failed")
        return
    
    log_status("Calibration complete. Press SPACE to start...")
    
    while True:
        cam.update()
        if (cv2.waitKey(30) & 0xFF) == ord(' '):
            break
    
    # Wait for next cycle boundary
    current_time = time.time()
    next_cycle = ((int(current_time) // CYCLE_TIME) + 1) * CYCLE_TIME
    wait_time = next_cycle - current_time
    log_status(f"Starting in {wait_time:.1f} seconds...")
    
    wait_until(cam, next_cycle)
    
    cycle_start = time.time()
    cycle_num = 0
    
    # Calculate what we expect to receive from root
    expected_from_root = generate_deterministic_tensor(ROOT_SEED, cycle_num)
    log_status(f"Expecting tensor: min={np.min(expected_from_root):.3f}, max={np.max(expected_from_root):.3f}, mean={np.mean(expected_from_root):.3f}")
    
    # Phase 1: Root compute (0-0.5s)
    log_status("Phase 1: Root computing...")
    wait_until(cam, cycle_start + 0.5)
    
    # Phase 2: Receive from root (0.5-5.5s)
    log_status("Phase 2: Receiving from root...")
    phase_start = time.time()
    received_frames = receive_frames_properly(cam, phase_start)
    
    # Decode and verify
    received_tensor = None
    transform_tensor = None
    
    try:
        received_tensor = frames_to_tensor_simple(received_frames)
        log_status(f"Decoded tensor: min={np.min(received_tensor):.3f}, max={np.max(received_tensor):.3f}, mean={np.mean(received_tensor):.3f}")
        
        # Verify against expected
        matches, num_diff, max_error = verify_tensors(expected_from_root, received_tensor)
        
        if matches:
            log_status("✓ SUCCESS: Received exact tensor from root!")
            transform_tensor = apply_transform(received_tensor)
        else:
            log_status(f"✗ FAILURE: {num_diff}/{TENSOR_SIZE} values differ from expected")
            log_status(f"  Maximum error: {max_error:.6f}")
            # Use dummy tensor if verification failed
            from simon_test.tensor_verification_test import WORKER_SEED
            transform_tensor = generate_deterministic_tensor(WORKER_SEED, cycle_num)
        
    except Exception as e:
        log_status(f"✗ ERROR decoding frames: {e}")
        from simon_test.tensor_verification_test import WORKER_SEED
        transform_tensor = generate_deterministic_tensor(WORKER_SEED, cycle_num)
    
    # Wait for phase end
    wait_until(cam, cycle_start + 5.5)
    
    # Phase 3: Compute (5.5-6s)
    log_status("Phase 3: Computing transform...")
    if received_tensor is not None and matches:
        log_status(f"  Transformed tensor: mean={np.mean(transform_tensor):.6f}")
    else:
        log_status(f"  Using dummy tensor due to validation failure")
    
    wait_until(cam, cycle_start + 6.0)
    
    # Phase 4: Transmit to root (6-11s) - FIXED TRANSMISSION
    log_status("Phase 4: Transmitting to root...")
    frames = tensor_to_frames_simple(transform_tensor)
    phase_start = time.time()
    transmit_frames_properly(cam, frames, phase_start)
    
    log_status("\n" + "="*60)
    log_status("WORKER VERIFICATION RESULTS:")
    log_status("="*60)
    log_status(f"Expected from root: shape={expected_from_root.shape}, mean={np.mean(expected_from_root):.6f}")
    if received_tensor is not None:
        log_status(f"Received tensor:    shape={received_tensor.shape}, mean={np.mean(received_tensor):.6f}")
        log_status(f"Verification: {'PASSED' if matches else f'FAILED ({num_diff} differences)'}")
    else:
        log_status("Verification: FAILED (decoding error)")
    
    # Keep displaying for a moment
    wait_until(cam, cycle_start + 11)
    
    time.sleep(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fixed tensor verification test')
    parser.add_argument('mode', choices=['root', 'worker'], help='Run as root or worker node')
    args = parser.parse_args()
    
    if args.mode == 'root':
        test_root_node()
    else:
        test_worker_node()
