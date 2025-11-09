import sys
import time
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Frame, ROWS, COLS, SECONDS_PER_FRAME  # noqa: E402
from codec import codec  # noqa: E402


# Distributed training states
class DistributedState(Enum):
    COMPUTE_G = "compute_g"
    COMMUNICATE_G = "communicate_g"
    COMPUTE_THETA = "compute_theta"
    COMMUNICATE_THETA = "communicate_theta"
    COMPUTE_EVAL = "compute_eval"


# State durations in seconds (adjusted for camera timing)
STATE_DURATIONS = {
    DistributedState.COMPUTE_G: 0.5,          # Compute gradients
    DistributedState.COMMUNICATE_G: 3.0,      # Send gradients (2 frames @ 1s each + margin)
    DistributedState.COMPUTE_THETA: 0.5,      # Update parameters
    DistributedState.COMMUNICATE_THETA: 3.0,  # Send parameters (2 frames @ 1s each + margin)
    DistributedState.COMPUTE_EVAL: 1.0,       # Evaluate
}

CYCLE_DURATION = sum(STATE_DURATIONS.values())  # 8 seconds total

# Camera parameters
PIXELS_PER_FRAME = ROWS * COLS  # 2800 pixels
GRIDS_PER_FLOAT16 = 6  # With K=8
FLOATS_PER_FRAME = PIXELS_PER_FRAME // GRIDS_PER_FLOAT16  # 466 floats per frame

# Seeds for reproducibility
ROOT_SEED = 42
WORKER_SEED = 123


def get_current_state_from_time() -> Tuple[DistributedState, float]:
    """Determine current state based on unix time modulo cycle duration."""
    current_time = time.time()
    cycle_time = current_time % CYCLE_DURATION
    
    t1 = STATE_DURATIONS[DistributedState.COMPUTE_G]
    t2 = t1 + STATE_DURATIONS[DistributedState.COMMUNICATE_G]
    t3 = t2 + STATE_DURATIONS[DistributedState.COMPUTE_THETA]
    t4 = t3 + STATE_DURATIONS[DistributedState.COMMUNICATE_THETA]
    
    if cycle_time < t1:
        return DistributedState.COMPUTE_G, cycle_time
    elif cycle_time < t2:
        return DistributedState.COMMUNICATE_G, cycle_time - t1
    elif cycle_time < t3:
        return DistributedState.COMPUTE_THETA, cycle_time - t2
    elif cycle_time < t4:
        return DistributedState.COMMUNICATE_THETA, cycle_time - t3
    else:
        return DistributedState.COMPUTE_EVAL, cycle_time - t4


def wait_for_state(target_state: DistributedState) -> float:
    """Wait until the specified state begins."""
    while True:
        state, _ = get_current_state_from_time()
        if state == target_state:
            return time.time()
        time.sleep(0.01)


def tensors_to_frames(tensors: List[torch.Tensor]) -> List[Frame]:
    """Convert list of torch tensors to camera frames."""
    # Flatten all tensors into a single array
    flat_values = []
    for tensor in tensors:
        if tensor is not None:
            # Detach if tensor requires grad
            t = tensor.detach() if tensor.requires_grad else tensor
            flat_values.extend(t.cpu().numpy().flatten().astype(np.float16))
    
    total_floats = len(flat_values)
    num_frames = (total_floats + FLOATS_PER_FRAME - 1) // FLOATS_PER_FRAME
    
    frames = []
    for frame_idx in range(num_frames):
        frame_data = np.zeros((ROWS, COLS), dtype=np.int64)
        
        start_idx = frame_idx * FLOATS_PER_FRAME
        end_idx = min(start_idx + FLOATS_PER_FRAME, total_floats)
        
        # Encode values for this frame
        for i in range(end_idx - start_idx):
            value_idx = start_idx + i
            pixel_start = i * GRIDS_PER_FLOAT16
            
            # Create codec for single value
            c = codec(1, 1, 0, 7)  # K=8
            value_array = np.array([[flat_values[value_idx]]], dtype=np.float16)
            grids = c.encode(value_array)
            
            # Place grids in frame
            for g in range(GRIDS_PER_FLOAT16):
                pixel_idx = pixel_start + g
                row = pixel_idx // COLS
                col = pixel_idx % COLS
                if row < ROWS and col < COLS:
                    frame_data[row, col] = int(grids[g, 0, 0])
        
        frames.append(Frame(data=frame_data))
    
    return frames


def frames_to_tensors(frames: List[Frame], tensor_shapes: List[Tuple]) -> List[torch.Tensor]:
    """Convert camera frames back to torch tensors with original shapes."""
    # Decode all values
    all_values = []
    
    for frame in frames:
        frame_data = frame.data
        
        # Extract values from frame
        for i in range(FLOATS_PER_FRAME):
            pixel_start = i * GRIDS_PER_FLOAT16
            
            # Check if we have enough pixels left
            if pixel_start + GRIDS_PER_FLOAT16 > PIXELS_PER_FRAME:
                break
            
            # Extract grids
            grids = np.zeros((GRIDS_PER_FLOAT16, 1, 1), dtype=np.int64)
            valid = True
            
            for g in range(GRIDS_PER_FLOAT16):
                pixel_idx = pixel_start + g
                row = pixel_idx // COLS
                col = pixel_idx % COLS
                
                if row < ROWS and col < COLS:
                    val = frame_data[row, col]
                    if val < 0 or val > 7:
                        valid = False
                        break
                    grids[g, 0, 0] = val
            
            if valid:
                try:
                    c = codec(1, 1, 0, 7)
                    decoded = c.decode(grids)
                    all_values.append(decoded[0, 0])
                except:
                    all_values.append(0.0)
    
    # Reshape into original tensor shapes
    tensors = []
    value_idx = 0
    
    for shape in tensor_shapes:
        num_elements = np.prod(shape)
        if value_idx + num_elements <= len(all_values):
            tensor_values = all_values[value_idx:value_idx + num_elements]
            tensor = torch.tensor(tensor_values, dtype=torch.float32).reshape(shape)
            tensors.append(tensor)
            value_idx += num_elements
        else:
            # Not enough values, create zero tensor
            tensors.append(torch.zeros(shape, dtype=torch.float32))
    
    return tensors


def get_parameter_shapes(model) -> List[Tuple]:
    """Get shapes of all model parameters."""
    shapes = []
    for param in model.parameters():
        shapes.append(tuple(param.shape))
    return shapes


def log_timing(message: str, state: DistributedState = None):
    """Log with timestamp and optional state info."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if state:
        current_state, time_in_state = get_current_state_from_time()
        print(f"[{timestamp}] {message} (state: {state.value}, t={time_in_state:.1f}s)")
    else:
        print(f"[{timestamp}] {message}")
