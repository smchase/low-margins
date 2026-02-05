import sys
import time
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, ROWS, COLS, SECONDS_PER_FRAME  # noqa: E402
from codec import codec  # noqa: E402

# Protocol constants
CYCLE_TIME = 11  # Total cycle duration in seconds
TENSOR_SIZE = 670  # Number of float16 values (matching model size)
GRIDS_PER_FLOAT16 = 6  # Number of grids needed per float16 value (with K=8)
PIXELS_PER_FRAME = ROWS * COLS  # 450 pixels
FRAMES_PER_TENSOR = 9  # ceil(670 * 6 / 450) = 9 frames

# Seeds for deterministic generation
ROOT_SEED = 42
WORKER_SEED = 123


def log_status(message: str):
    """Log status with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def generate_deterministic_tensor(seed: int, cycle: int) -> np.ndarray:
    """Generate a deterministic tensor based on seed and cycle."""
    rng = np.random.RandomState(seed + cycle)
    tensor = rng.uniform(-1, 1, TENSOR_SIZE).astype(np.float16)
    return tensor


def tensor_to_frames_simple(tensor: np.ndarray) -> list[Frame]:
    """Convert tensor to frames using a simple, verifiable method."""
    if tensor.shape != (TENSOR_SIZE,):
        raise ValueError(f"Expected shape ({TENSOR_SIZE},), got {tensor.shape}")
    
    frames = []
    values_per_frame = PIXELS_PER_FRAME // GRIDS_PER_FLOAT16  # 75 values per frame
    
    for frame_idx in range(FRAMES_PER_TENSOR):
        frame_data = np.zeros((ROWS, COLS), dtype=np.int64)
        
        start_idx = frame_idx * values_per_frame
        end_idx = min(start_idx + values_per_frame, TENSOR_SIZE)
        
        # Simple encoding: use codec for each value individually
        for i in range(end_idx - start_idx):
            value_idx = start_idx + i
            pixel_start = i * GRIDS_PER_FLOAT16
            
            # Create a 1x1 codec for this single value
            c = codec(1, 1, 0, 7)  # K=8 (values 0-7)
            value_array = tensor[value_idx:value_idx+1].reshape(1, 1)
            grids = c.encode(value_array)  # Returns (6, 1, 1)
            
            # Place the 6 grid values in the frame
            for g in range(GRIDS_PER_FLOAT16):
                pixel_idx = pixel_start + g
                row = pixel_idx // COLS
                col = pixel_idx % COLS
                if row < ROWS:
                    frame_data[row, col] = int(grids[g, 0, 0])
        
        frames.append(Frame(data=frame_data))
    
    return frames


def frames_to_tensor_simple(frames: list[Frame]) -> np.ndarray:
    """Convert frames back to tensor using simple method."""
    if len(frames) != FRAMES_PER_TENSOR:
        raise ValueError(f"Expected {FRAMES_PER_TENSOR} frames, got {len(frames)}")
    
    values = []
    values_per_frame = PIXELS_PER_FRAME // GRIDS_PER_FLOAT16
    
    for frame_idx, frame in enumerate(frames):
        frame_data = frame.data
        
        start_idx = frame_idx * values_per_frame
        end_idx = min(start_idx + values_per_frame, TENSOR_SIZE)
        
        for i in range(end_idx - start_idx):
            pixel_start = i * GRIDS_PER_FLOAT16
            
            # Extract 6 grid values
            grids = np.zeros((GRIDS_PER_FLOAT16, 1, 1), dtype=np.int64)
            for g in range(GRIDS_PER_FLOAT16):
                pixel_idx = pixel_start + g
                row = pixel_idx // COLS
                col = pixel_idx % COLS
                if row < ROWS:
                    grids[g, 0, 0] = frame_data[row, col]
            
            # Decode single value
            c = codec(1, 1, 0, 7)
            decoded = c.decode(grids)  # Returns (1, 1) float16
            values.append(decoded[0, 0])
    
    return np.array(values[:TENSOR_SIZE], dtype=np.float16)


def apply_transform(tensor: np.ndarray) -> np.ndarray:
    """Apply a simple transformation (multiply by 2)."""
    # Clip to prevent overflow
    result = tensor * 2.0
    return np.clip(result, -65504, 65504).astype(np.float16)


def verify_tensors(expected: np.ndarray, received: np.ndarray) -> tuple[bool, int, float]:
    """Verify if two tensors match."""
    if expected.shape != received.shape:
        return False, -1, float('inf')
    
    # Use allclose for float comparisons with small tolerance
    close_mask = np.isclose(expected, received, rtol=1e-3, atol=1e-3, equal_nan=False)
    all_close = np.all(close_mask)
    
    if all_close:
        return True, 0, 0.0
    
    # Count differences
    diff_mask = ~close_mask
    num_diff = np.sum(diff_mask)
    
    # Max absolute error
    if num_diff > 0:
        max_error = np.max(np.abs(expected[diff_mask] - received[diff_mask]))
    else:
        max_error = 0.0
    
    return False, int(num_diff), float(max_error)


def test_root_node():
    """Root node test - sends tensor, receives transformed tensor."""
    cam = Camera()
    
    log_status("=== TENSOR VERIFICATION TEST - ROOT NODE ===")
    log_status("Testing 1 cycle of deterministic tensor exchange")
    
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
    
    while time.time() < next_cycle:
        cam.update()
        cv2.waitKey(30)
    
    cycle_start = time.time()
    cycle_num = 0
    
    # Generate deterministic tensor for this cycle
    original_tensor = generate_deterministic_tensor(ROOT_SEED, cycle_num)
    log_status(f"Generated tensor: min={np.min(original_tensor):.3f}, max={np.max(original_tensor):.3f}, mean={np.mean(original_tensor):.3f}")
    
    # What we expect to receive back (transformed)
    expected_response = apply_transform(original_tensor)
    
    # Phase 1: Compute (0-0.5s)
    log_status("Phase 1: Compute")
    time.sleep(0.5)
    
    # Phase 2: Transmit to worker (0.5-5.5s)
    log_status("Phase 2: Transmitting to worker...")
    frames = tensor_to_frames_simple(original_tensor)
    
    phase_start = time.time()
    for i, frame in enumerate(frames):
        # Wait for frame timing
        while (time.time() - phase_start) < (i * SECONDS_PER_FRAME):
            cam.update()
            cv2.waitKey(10)
        
        cam.transmit(frame)
        log_status(f"  Transmitted frame {i+1}/{len(frames)}")
    
    # Wait for phase end
    while (time.time() - cycle_start) < 5.5:
        cam.update()
        cv2.waitKey(30)
    
    # Phase 3: Worker compute (5.5-6s)
    log_status("Phase 3: Worker computing...")
    while (time.time() - cycle_start) < 6.0:
        cam.update()
        cv2.waitKey(30)
    
    # Phase 4: Receive from worker (6-11s)
    log_status("Phase 4: Receiving from worker...")
    received_frames = []
    phase_start = time.time()
    
    for i in range(FRAMES_PER_TENSOR):
        # Wait for frame midpoint
        target_time = (i + 0.5) * SECONDS_PER_FRAME
        while (time.time() - phase_start) < target_time:
            cam.update()
            cv2.waitKey(10)
        
        frame = cam.receive()
        received_frames.append(frame)
        log_status(f"  Received frame {i+1}/{FRAMES_PER_TENSOR}")
    
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
    """Worker node test - receives tensor, sends back transformed tensor."""
    cam = Camera()
    
    log_status("=== TENSOR VERIFICATION TEST - WORKER NODE ===")
    log_status("Testing 1 cycle of deterministic tensor exchange")
    
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
    
    while time.time() < next_cycle:
        cam.update()
        cv2.waitKey(30)
    
    cycle_start = time.time()
    cycle_num = 0
    
    # Calculate what we expect to receive from root
    expected_from_root = generate_deterministic_tensor(ROOT_SEED, cycle_num)
    log_status(f"Expecting tensor: min={np.min(expected_from_root):.3f}, max={np.max(expected_from_root):.3f}, mean={np.mean(expected_from_root):.3f}")
    
    # Phase 1: Root compute (0-0.5s)
    log_status("Phase 1: Root computing...")
    while (time.time() - cycle_start) < 0.5:
        cam.update()
        cv2.waitKey(30)
    
    # Phase 2: Receive from root (0.5-5.5s)
    log_status("Phase 2: Receiving from root...")
    received_frames = []
    phase_start = time.time()
    
    for i in range(FRAMES_PER_TENSOR):
        # Wait for frame midpoint
        target_time = (i + 0.5) * SECONDS_PER_FRAME
        while (time.time() - phase_start) < target_time:
            cam.update()
            cv2.waitKey(10)
        
        frame = cam.receive()
        received_frames.append(frame)
        log_status(f"  Received frame {i+1}/{FRAMES_PER_TENSOR}")
    
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
            transform_tensor = generate_deterministic_tensor(WORKER_SEED, cycle_num)
        
    except Exception as e:
        log_status(f"✗ ERROR decoding frames: {e}")
        transform_tensor = generate_deterministic_tensor(WORKER_SEED, cycle_num)
    
    # Wait for phase end
    while (time.time() - cycle_start) < 5.5:
        cam.update()
        cv2.waitKey(30)
    
    # Phase 3: Compute (5.5-6s)
    log_status("Phase 3: Computing transform...")
    if received_tensor is not None and matches:
        log_status(f"  Applying transform to validated tensor")
    else:
        log_status(f"  Using dummy tensor due to validation failure")
    
    while (time.time() - cycle_start) < 6.0:
        cam.update()
        cv2.waitKey(30)
    
    # Phase 4: Transmit to root (6-11s)
    log_status("Phase 4: Transmitting to root...")
    frames = tensor_to_frames_simple(transform_tensor)
    
    phase_start = time.time()
    for i, frame in enumerate(frames):
        # Wait for frame timing
        while (time.time() - phase_start) < (i * SECONDS_PER_FRAME):
            cam.update()
            cv2.waitKey(10)
        
        cam.transmit(frame)
        log_status(f"  Transmitted frame {i+1}/{len(frames)}")
    
    log_status("\n" + "="*60)
    log_status("WORKER VERIFICATION RESULTS:")
    log_status("="*60)
    log_status(f"Expected from root: shape={expected_from_root.shape}, mean={np.mean(expected_from_root):.6f}")
    if received_tensor is not None:
        log_status(f"Received tensor:    shape={received_tensor.shape}, mean={np.mean(received_tensor):.6f}")
        log_status(f"Verification: {'PASSED' if matches else f'FAILED ({num_diff} differences)'}")
    else:
        log_status("Verification: FAILED (decoding error)")
    
    # Keep displaying
    time.sleep(2)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Tensor verification test')
    parser.add_argument('mode', choices=['root', 'worker'], help='Run as root or worker node')
    args = parser.parse_args()
    
    if args.mode == 'root':
        test_root_node()
    else:
        test_worker_node()
