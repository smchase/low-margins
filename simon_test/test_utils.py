import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Frame, ROWS, COLS  # noqa: E402
from codec import codec  # noqa: E402

# Protocol constants
CYCLE_TIME = 11  # Total cycle duration in seconds
COMPUTE_PHASE_DURATION = 0.5  # Duration of each compute phase
TRANSMIT_PHASE_DURATION = 5.0  # Duration of each transmit phase

# Tensor specifications
TENSOR_SIZE = 670  # Number of float16 values (matching model size)
GRIDS_PER_FLOAT16 = 6  # Number of grids needed per float16 value (with K=8)
PIXELS_PER_FRAME = ROWS * COLS  # 450 pixels
FRAMES_PER_TENSOR = 9  # ceil(670 * 6 / 450) = 9 frames

# Codec parameters
CODEC_MIN_VAL = 0
CODEC_MAX_VAL = 7  # K=8 for 3-bit values


class PhaseType:
    """Enum for phase types in the protocol"""
    COMPUTE_ROOT = "compute_root"
    TRANSMIT_ROOT_TO_WORKER = "transmit_root_to_worker"
    COMPUTE_WORKER = "compute_worker"
    TRANSMIT_WORKER_TO_ROOT = "transmit_worker_to_root"


def get_current_phase_info() -> Tuple[PhaseType, float, float]:
    """
    Get the current phase based on Unix time.
    
    Returns:
        Tuple of (phase_type, seconds_into_phase, seconds_remaining)
    """
    current_time = time.time()
    cycle_position = current_time % CYCLE_TIME
    
    if cycle_position < COMPUTE_PHASE_DURATION:
        phase = PhaseType.COMPUTE_ROOT
        seconds_in = cycle_position
        seconds_remain = COMPUTE_PHASE_DURATION - seconds_in
    elif cycle_position < COMPUTE_PHASE_DURATION + TRANSMIT_PHASE_DURATION:
        phase = PhaseType.TRANSMIT_ROOT_TO_WORKER
        seconds_in = cycle_position - COMPUTE_PHASE_DURATION
        seconds_remain = TRANSMIT_PHASE_DURATION - seconds_in
    elif cycle_position < COMPUTE_PHASE_DURATION + TRANSMIT_PHASE_DURATION + COMPUTE_PHASE_DURATION:
        phase = PhaseType.COMPUTE_WORKER
        seconds_in = cycle_position - COMPUTE_PHASE_DURATION - TRANSMIT_PHASE_DURATION
        seconds_remain = COMPUTE_PHASE_DURATION - seconds_in
    else:
        phase = PhaseType.TRANSMIT_WORKER_TO_ROOT
        seconds_in = cycle_position - COMPUTE_PHASE_DURATION - TRANSMIT_PHASE_DURATION - COMPUTE_PHASE_DURATION
        seconds_remain = TRANSMIT_PHASE_DURATION - seconds_in
    
    return phase, seconds_in, seconds_remain


def wait_for_phase(target_phase: PhaseType) -> float:
    """
    Wait until the specified phase begins.
    
    Returns:
        The Unix timestamp when the phase started
    """
    while True:
        phase, _, _ = get_current_phase_info()
        if phase == target_phase:
            return time.time()
        time.sleep(0.01)  # Small sleep to avoid busy waiting


def generate_test_tensor(seed: int, cycle_number: int) -> np.ndarray:
    """
    Generate a deterministic test tensor based on seed and cycle number.
    
    Args:
        seed: Random seed for the node (different for root/worker)
        cycle_number: Current cycle number
        
    Returns:
        numpy array of shape (TENSOR_SIZE,) with float16 values
    """
    rng = np.random.RandomState(seed + cycle_number)
    # Generate random float16 values in range [-1, 1]
    tensor = rng.uniform(-1, 1, TENSOR_SIZE).astype(np.float16)
    return tensor


def tensor_to_frames(tensor: np.ndarray) -> List[Frame]:
    """
    Convert a tensor of float16 values to a list of camera frames.
    
    Args:
        tensor: numpy array of shape (TENSOR_SIZE,) with float16 values
        
    Returns:
        List of Frame objects ready for transmission
    """
    if tensor.shape != (TENSOR_SIZE,):
        raise ValueError(f"Expected tensor shape ({TENSOR_SIZE},), got {tensor.shape}")
    if tensor.dtype != np.float16:
        raise ValueError(f"Expected dtype float16, got {tensor.dtype}")
    
    frames = []
    
    # Calculate how many float16 values we can fit per frame
    # Each float16 needs GRIDS_PER_FLOAT16 grids, each grid is ROWS x COLS
    # But we need to pack multiple float16s into the same frame
    floats_per_frame = PIXELS_PER_FRAME // GRIDS_PER_FLOAT16  # 450 / 6 = 75
    
    for frame_idx in range(FRAMES_PER_TENSOR):
        start_idx = frame_idx * floats_per_frame
        end_idx = min(start_idx + floats_per_frame, TENSOR_SIZE)
        
        # Get chunk of tensor for this frame
        chunk = tensor[start_idx:end_idx]
        chunk_size = len(chunk)
        
        # For this frame, we'll encode each float16 separately and then pack
        # We need to pack the grids into a single ROWS x COLS frame
        frame_data = np.zeros((ROWS, COLS), dtype=np.int64)
        
        # We can fit 75 float16 values in one frame (450 pixels / 6 grids per float)
        # Arrange them in a way that uses all pixels
        pixels_per_float = GRIDS_PER_FLOAT16
        
        for i in range(min(chunk_size, floats_per_frame)):
            # Create a tiny 1x1 "image" for this single float16
            single_value = chunk[i:i+1].reshape(1, 1)
            
            # Use a mini codec just for this value
            mini_codec = codec(1, 1, CODEC_MIN_VAL, CODEC_MAX_VAL)
            grids = mini_codec.encode(single_value)  # Shape: (6, 1, 1)
            
            # Place these 6 values into the frame
            # Calculate position in the frame
            pixel_offset = i * pixels_per_float
            row = pixel_offset // COLS
            col = pixel_offset % COLS
            
            # Place the 6 grid values sequentially
            for g in range(GRIDS_PER_FLOAT16):
                if col >= COLS:
                    row += 1
                    col = 0
                if row < ROWS:
                    frame_data[row, col] = grids[g, 0, 0]
                    col += 1
        
        frames.append(Frame(data=frame_data))
    
    return frames


def frames_to_tensor(frames: List[Frame]) -> np.ndarray:
    """
    Convert a list of camera frames back to a tensor of float16 values.
    
    Args:
        frames: List of Frame objects
        
    Returns:
        numpy array of shape (TENSOR_SIZE,) with float16 values
    """
    if len(frames) != FRAMES_PER_TENSOR:
        raise ValueError(f"Expected {FRAMES_PER_TENSOR} frames, got {len(frames)}")
    
    floats_per_frame = PIXELS_PER_FRAME // GRIDS_PER_FLOAT16
    decoded_values = []
    
    for frame_idx, frame in enumerate(frames):
        frame_data = frame.data
        
        # Extract float16 values from this frame
        pixels_per_float = GRIDS_PER_FLOAT16
        max_floats = floats_per_frame
        if frame_idx == len(frames) - 1:
            # Last frame might have fewer values
            max_floats = TENSOR_SIZE - frame_idx * floats_per_frame
        
        for i in range(max_floats):
            # Extract the 6 grid values for this float16
            pixel_offset = i * pixels_per_float
            row = pixel_offset // COLS
            col = pixel_offset % COLS
            
            # Collect the 6 grid values
            grids = np.zeros((GRIDS_PER_FLOAT16, 1, 1), dtype=np.int64)
            for g in range(GRIDS_PER_FLOAT16):
                if col >= COLS:
                    row += 1
                    col = 0
                if row < ROWS:
                    grids[g, 0, 0] = frame_data[row, col]
                    col += 1
            
            # Decode this single float16 value
            mini_codec = codec(1, 1, CODEC_MIN_VAL, CODEC_MAX_VAL)
            decoded = mini_codec.decode(grids)  # Shape: (1, 1) with float16
            decoded_values.append(decoded[0, 0])
    
    return np.array(decoded_values, dtype=np.float16)


def verify_tensor_integrity(original: np.ndarray, received: np.ndarray) -> Tuple[bool, int, float]:
    """
    Verify the integrity of a received tensor against the original.
    
    Args:
        original: Original tensor
        received: Received tensor
        
    Returns:
        Tuple of (all_match, num_differences, max_absolute_error)
    """
    if original.shape != received.shape:
        return False, -1, float('inf')
    
    differences = original != received
    num_diff = np.sum(differences)
    
    if num_diff == 0:
        return True, 0, 0.0
    
    # Calculate maximum absolute error for float comparisons
    max_error = np.max(np.abs(original - received))
    
    return False, int(num_diff), float(max_error)


def apply_test_transform(tensor: np.ndarray, transform_type: str = "multiply") -> np.ndarray:
    """
    Apply a simple transformation to a tensor for testing.
    
    Args:
        tensor: Input tensor
        transform_type: Type of transformation ("multiply", "add", "negate")
        
    Returns:
        Transformed tensor
    """
    if transform_type == "multiply":
        # Multiply by 2, clipping to valid float16 range
        result = tensor * 2.0
        # Clip to prevent overflow
        result = np.clip(result, np.finfo(np.float16).min, np.finfo(np.float16).max)
        return result.astype(np.float16)
    elif transform_type == "add":
        # Add 0.5
        result = tensor + 0.5
        result = np.clip(result, np.finfo(np.float16).min, np.finfo(np.float16).max)
        return result.astype(np.float16)
    elif transform_type == "negate":
        return -tensor
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def get_cycle_timing_stats() -> dict:
    """Get current timing statistics for logging."""
    current_time = time.time()
    cycle_number = int(current_time // CYCLE_TIME)
    phase, seconds_in, seconds_remain = get_current_phase_info()
    
    return {
        "unix_time": current_time,
        "cycle_number": cycle_number,
        "phase": phase,
        "seconds_into_phase": seconds_in,
        "seconds_remaining": seconds_remain,
        "cycle_position": current_time % CYCLE_TIME
    }


def compute_tensor_checksum(tensor: np.ndarray) -> int:
    """
    Compute a deterministic checksum for a tensor.
    Uses the raw bytes of the float16 values.
    
    Args:
        tensor: Float16 numpy array
        
    Returns:
        Integer checksum
    """
    # Convert to uint16 view to get raw bits
    as_u16 = tensor.view(np.uint16)
    # Simple but effective checksum: sum of all bits XOR'd with positions
    checksum = 0
    for i, val in enumerate(as_u16.flat):
        checksum ^= (int(val) * (i + 1))
    return checksum & 0xFFFFFFFF  # Keep as 32-bit


def verify_tensor_bitwise(original: np.ndarray, received: np.ndarray) -> Tuple[bool, int, str]:
    """
    Verify two tensors are bit-exact matches (for float16).
    
    Args:
        original: Original tensor (float16)
        received: Received tensor (float16)
        
    Returns:
        Tuple of (matches, num_bit_differences, details_string)
    """
    if original.shape != received.shape:
        return False, -1, f"Shape mismatch: {original.shape} vs {received.shape}"
    
    if original.dtype != np.float16 or received.dtype != np.float16:
        return False, -1, f"Dtype mismatch: {original.dtype} vs {received.dtype}"
    
    # Compare as uint16 for bit-exact comparison
    orig_u16 = original.view(np.uint16)
    recv_u16 = received.view(np.uint16)
    
    diff_mask = orig_u16 != recv_u16
    num_diff = np.sum(diff_mask)
    
    if num_diff == 0:
        orig_checksum = compute_tensor_checksum(original)
        return True, 0, f"Perfect match! Checksum: 0x{orig_checksum:08X}"
    
    # Find how many bits differ in total
    xor_vals = orig_u16[diff_mask] ^ recv_u16[diff_mask]
    total_bit_flips = sum(bin(int(x)).count('1') for x in xor_vals)
    
    details = f"{num_diff} values differ ({num_diff/len(orig_u16.flat)*100:.2f}%), {total_bit_flips} total bit flips"
    return False, int(num_diff), details


def log_tensor_details(tensor: np.ndarray, label: str) -> List[str]:
    """
    Generate detailed logging information about a tensor.
    
    Args:
        tensor: Tensor to log
        label: Label for this tensor
        
    Returns:
        List of log strings
    """
    lines = []
    lines.append(f"{label}:")
    lines.append(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
    lines.append(f"  Stats: min={np.min(tensor):.6f}, max={np.max(tensor):.6f}, mean={np.mean(tensor):.6f}, std={np.std(tensor):.6f}")
    lines.append(f"  Checksum: 0x{compute_tensor_checksum(tensor):08X}")
    
    # Show first few values
    flat = tensor.flat
    preview = [f"{flat[i]:.6f}" for i in range(min(5, len(flat)))]
    lines.append(f"  First values: [{', '.join(preview)}...]")
    
    return lines
