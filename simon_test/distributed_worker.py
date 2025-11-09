import sys
import time
from pathlib import Path
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, SECONDS_PER_FRAME, RECEIVE_OFFSET  # noqa: E402
from train.model import MLP  # noqa: E402
from train.worker import Worker  # noqa: E402
from simon_test.distributed_utils import (  # noqa: E402
    DistributedState, get_current_state_from_time, wait_for_state,
    tensors_to_frames, frames_to_tensors, get_parameter_shapes,
    log_timing, CYCLE_DURATION, WORKER_SEED
)


def transmit_frames_properly(cam: Camera, frames: list[Frame], phase_start_time: float):
    """Transmit frames ensuring each is displayed properly."""
    for i, frame in enumerate(frames):
        # Transmit the frame
        cam.transmit(frame)
        log_timing(f"  Transmitted frame {i+1}/{len(frames)}")
        
        # Keep displaying this frame until it's time for the next one
        next_frame_time = (i + 1) * SECONDS_PER_FRAME
        while (time.time() - phase_start_time) < next_frame_time:
            cam.update()  # This keeps the frame visible!
            cv2.waitKey(10)


def receive_frames_properly(cam: Camera, phase_start_time: float, num_frames: int) -> list[Frame]:
    """Receive frames with proper timing."""
    received_frames = []
    
    for i in range(num_frames):
        # Wait with configured offset for frame to be displayed
        target_time = i * SECONDS_PER_FRAME + RECEIVE_OFFSET
        while (time.time() - phase_start_time) < target_time:
            cam.update()
            cv2.waitKey(10)
        
        # Capture frame
        frame = cam.receive()
        received_frames.append(frame)
        log_timing(f"  Received frame {i+1}/{num_frames}")
    
    return received_frames


def wait_until(cam: Camera, target_time: float):
    """Wait until target time while keeping camera updated."""
    while time.time() < target_time:
        cam.update()
        cv2.waitKey(30)


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Distributed training worker node')
    parser.add_argument('--num-steps', type=int, default=10, help='Number of training steps')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()
    
    # Initialize camera
    cam = Camera()
    
    log_timing("=== DISTRIBUTED TRAINING - WORKER NODE ===")
    log_timing(f"Training for {args.num_steps} steps")
    log_timing(f"Cycle duration: {CYCLE_DURATION}s")
    
    if not cam.calibrate():
        log_timing("ERROR: Calibration failed")
        return
    
    log_timing("Calibration complete. Press SPACE to start...")
    
    while True:
        cam.update()
        if (cv2.waitKey(30) & 0xFF) == ord(' '):
            break
    
    # Set random seed (use ROOT_SEED to start with same model)
    torch.manual_seed(ROOT_SEED)
    
    # Create model
    device = 'cpu'
    model = MLP().bfloat16().to(device)
    param_shapes = get_parameter_shapes(model)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Worker gets second half of data
    data_indices = list(range(len(train_dataset) // 2, len(train_dataset)))
    data_subset = Subset(train_dataset, data_indices)
    
    worker = Worker(
        model=model,
        data_subset=data_subset,
        batch_size=1024,
        learning_rate=args.learning_rate,
        device=device,
        root=False,
        worker_id=1
    )
    
    # Calculate frames needed
    test_params = [p.clone() for p in model.parameters()]
    param_frames = tensors_to_frames(test_params)
    num_param_frames = len(param_frames)
    log_timing(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    log_timing(f"Need {num_param_frames} frames for parameters/gradients")
    
    # Wait to start at beginning of COMPUTE_G
    log_timing("Waiting to synchronize to global clock...")
    wait_for_state(DistributedState.COMPUTE_G)
    
    start_time = time.time()
    log_timing(f"Starting training at unix time {start_time:.6f}")
    
    local_gradients = None
    total_train_loss = 0.0
    
    for step in range(args.num_steps):
        cycle_start = time.time()
        
        # ========== State 1: COMPUTE_G ==========
        state_start = wait_for_state(DistributedState.COMPUTE_G)
        log_timing(f"Step {step+1}: Computing gradients", DistributedState.COMPUTE_G)
        
        # Compute local gradients
        local_gradients, local_loss = worker.run_step()
        total_train_loss += local_loss
        
        grad_norm = sum(torch.norm(g).item() for g in local_gradients if g is not None)
        log_timing(f"  Computed gradients with norm: {grad_norm:.4f}, loss: {local_loss:.4f}")
        
        # ========== State 2: COMMUNICATE_G ==========
        state_start = wait_for_state(DistributedState.COMMUNICATE_G)
        log_timing(f"Step {step+1}: Transmitting gradients to root", DistributedState.COMMUNICATE_G)
        
        # Send gradients to root
        gradient_frames = tensors_to_frames(local_gradients)
        
        phase_start = time.time()
        transmit_frames_properly(cam, gradient_frames, phase_start)
        
        # ========== State 3: COMPUTE_THETA ==========
        state_start = wait_for_state(DistributedState.COMPUTE_THETA)
        log_timing(f"Step {step+1}: Root computing parameter update", DistributedState.COMPUTE_THETA)
        
        # Worker just waits during this phase
        wait_until(cam, time.time() + 0.4)  # Most of compute phase
        
        # ========== State 4: COMMUNICATE_THETA ==========
        state_start = wait_for_state(DistributedState.COMMUNICATE_THETA)
        log_timing(f"Step {step+1}: Receiving updated parameters", DistributedState.COMMUNICATE_THETA)
        
        # Receive updated parameters from root
        phase_start = time.time()
        received_frames = receive_frames_properly(cam, phase_start, num_param_frames)
        
        try:
            updated_params = frames_to_tensors(received_frames, param_shapes)
            worker.replace_parameters(updated_params)
            
            param_norm = torch.norm(updated_params[0]).item()
            log_timing(f"  Received parameters with norm: {param_norm:.4f}")
        except Exception as e:
            log_timing(f"  ERROR receiving parameters: {e}")
            # Continue with current parameters if reception failed
        
        # ========== State 5: COMPUTE_EVAL ==========
        state_start = wait_for_state(DistributedState.COMPUTE_EVAL)
        log_timing(f"Step {step+1}: Evaluation phase", DistributedState.COMPUTE_EVAL)
        
        # Worker logs its training progress
        avg_loss = total_train_loss / (step + 1)
        log_timing(f"  Step {step+1}/{args.num_steps}: Worker avg loss={avg_loss:.4f}")
        
        # Worker just waits during eval
        wait_until(cam, time.time() + 0.8)  # Most of eval phase
    
    # Training complete
    final_avg_loss = total_train_loss / args.num_steps
    log_timing(f"\nTraining complete! Worker average loss: {final_avg_loss:.4f}")
    
    # Save final model state
    torch.save(model.state_dict(), 'distributed_model_worker.pth')
    log_timing("Model saved to distributed_model_worker.pth")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
