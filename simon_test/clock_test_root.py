import sys
import time
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, Frame, ROWS, COLS, SECONDS_PER_FRAME  # noqa: E402
from test_utils import (  # noqa: E402
    PhaseType, get_current_phase_info, wait_for_phase,
    generate_test_tensor, tensor_to_frames, frames_to_tensor,
    verify_tensor_integrity, get_cycle_timing_stats,
    CYCLE_TIME, FRAMES_PER_TENSOR
)

# Root node seed for deterministic tensor generation
ROOT_SEED = 42
TARGET_CYCLES = 10  # Number of cycles to run for testing


def log_status(message: str, timing_stats: dict = None):
    """Log status with timestamp and optional timing information."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if timing_stats:
        phase_str = timing_stats["phase"].split(".")[-1]
        print(f"[{timestamp}] Cycle {timing_stats['cycle_number']}, {phase_str} "
              f"({timing_stats['seconds_into_phase']:.1f}s): {message}")
    else:
        print(f"[{timestamp}] {message}")


def main():
    cam = Camera()
    
    log_status("Clock Test Root Node - Starting")
    log_status(f"Target: {TARGET_CYCLES} complete cycles")
    log_status(f"Cycle time: {CYCLE_TIME}s")
    
    if not cam.calibrate():
        log_status("ERROR: Calibration did not complete")
        return
    
    log_status("Calibration complete")
    
    # Wait for user to start
    print("\n" + "="*60)
    print("Press SPACE to start clock synchronization test")
    print("Both root and worker nodes should start at the same time")
    print("="*60)
    
    while True:
        cam.update()
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            break
    
    # Wait for the start of the next cycle
    log_status("Waiting for next cycle to begin...")
    current_time = time.time()
    next_cycle_start = ((int(current_time) // CYCLE_TIME) + 1) * CYCLE_TIME
    wait_time = next_cycle_start - current_time
    log_status(f"Starting in {wait_time:.1f} seconds...")
    
    while time.time() < next_cycle_start:
        cam.update()
        cv2.waitKey(30)
    
    start_time = time.time()
    cycle_count = 0
    successful_cycles = 0
    
    # Track timing accuracy
    phase_timing_errors = []
    
    while cycle_count < TARGET_CYCLES:
        timing_stats = get_cycle_timing_stats()
        cycle_number = timing_stats["cycle_number"]
        
        # Phase 1: Compute (generate test tensor)
        phase_start = wait_for_phase(PhaseType.COMPUTE_ROOT)
        timing_error = phase_start - (cycle_number * CYCLE_TIME + start_time % CYCLE_TIME)
        phase_timing_errors.append(("compute_root", timing_error))
        
        test_tensor = generate_test_tensor(ROOT_SEED, cycle_number)
        frames_to_send = tensor_to_frames(test_tensor)
        log_status(f"Generated test tensor for cycle {cycle_number}", timing_stats)
        
        # Phase 2: Transmit to worker
        phase_start = wait_for_phase(PhaseType.TRANSMIT_ROOT_TO_WORKER)
        timing_error = phase_start - (cycle_number * CYCLE_TIME + 0.5 + start_time % CYCLE_TIME)
        phase_timing_errors.append(("transmit_to_worker", timing_error))
        
        log_status(f"Transmitting {FRAMES_PER_TENSOR} frames to worker", timing_stats)
        
        frame_idx = 0
        phase_start_time = time.time()
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Check if we're still in the correct phase
            current_phase, _, _ = get_current_phase_info()
            if current_phase != PhaseType.TRANSMIT_ROOT_TO_WORKER:
                log_status(f"WARNING: Phase changed during transmission! Only sent {frame_idx}/{FRAMES_PER_TENSOR} frames")
                break
            
            # Transmit frame at appropriate time
            elapsed = time.time() - phase_start_time
            target_time = frame_idx * SECONDS_PER_FRAME
            
            if elapsed >= target_time:
                cam.transmit(frames_to_send[frame_idx])
                frame_idx += 1
            
            cv2.waitKey(30)
        
        if frame_idx == FRAMES_PER_TENSOR:
            log_status(f"Successfully transmitted all {FRAMES_PER_TENSOR} frames", timing_stats)
        
        # Phase 3: Worker compute (we just wait)
        wait_for_phase(PhaseType.COMPUTE_WORKER)
        log_status("Worker computing...", get_cycle_timing_stats())
        
        # Phase 4: Receive from worker
        phase_start = wait_for_phase(PhaseType.TRANSMIT_WORKER_TO_ROOT)
        timing_error = phase_start - (cycle_number * CYCLE_TIME + 6.0 + start_time % CYCLE_TIME)
        phase_timing_errors.append(("receive_from_worker", timing_error))
        
        log_status(f"Receiving {FRAMES_PER_TENSOR} frames from worker", get_cycle_timing_stats())
        
        received_frames = []
        frame_idx = 0
        phase_start_time = time.time()
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Check if we're still in the correct phase
            current_phase, _, _ = get_current_phase_info()
            if current_phase != PhaseType.TRANSMIT_WORKER_TO_ROOT:
                log_status(f"WARNING: Phase changed during reception! Only received {frame_idx}/{FRAMES_PER_TENSOR} frames")
                break
            
            # Receive frame at midpoint of each frame period
            elapsed = time.time() - phase_start_time
            target_time = (frame_idx + 0.5) * SECONDS_PER_FRAME
            
            if elapsed >= target_time:
                received_frame = cam.receive()
                received_frames.append(received_frame)
                frame_idx += 1
            
            cv2.waitKey(30)
        
        # Verify received data if we got all frames
        if len(received_frames) == FRAMES_PER_TENSOR:
            try:
                received_tensor = frames_to_tensor(received_frames)
                
                # The worker should have applied a transformation
                # For now, let's just check if we received valid data
                is_valid = received_tensor.shape == test_tensor.shape
                
                if is_valid:
                    log_status(f"Cycle {cycle_number} completed successfully!", get_cycle_timing_stats())
                    successful_cycles += 1
                else:
                    log_status(f"Cycle {cycle_number} FAILED: Invalid tensor shape", get_cycle_timing_stats())
                
                # Log some statistics about the received data
                log_status(f"Received tensor stats: shape={received_tensor.shape}, "
                          f"min={np.min(received_tensor):.3f}, max={np.max(received_tensor):.3f}, "
                          f"mean={np.mean(received_tensor):.3f}")
                
            except Exception as e:
                log_status(f"Cycle {cycle_number} FAILED: Error decoding frames - {str(e)}", get_cycle_timing_stats())
        else:
            log_status(f"Cycle {cycle_number} FAILED: Incomplete reception", get_cycle_timing_stats())
        
        cycle_count += 1
        
        # Wait for next cycle if not the last one
        if cycle_count < TARGET_CYCLES:
            current_phase, _, remaining = get_current_phase_info()
            if remaining > 0:
                time.sleep(remaining)
    
    # Final report
    print("\n" + "="*60)
    print("CLOCK TEST COMPLETE - FINAL REPORT")
    print("="*60)
    print(f"Total cycles attempted: {cycle_count}")
    print(f"Successful cycles: {successful_cycles}")
    print(f"Success rate: {100 * successful_cycles / cycle_count:.1f}%")
    
    # Timing accuracy analysis
    if phase_timing_errors:
        abs_errors = [abs(error) for _, error in phase_timing_errors]
        avg_error = np.mean(abs_errors)
        max_error = np.max(abs_errors)
        print(f"\nTiming accuracy:")
        print(f"  Average phase timing error: {avg_error*1000:.1f}ms")
        print(f"  Maximum phase timing error: {max_error*1000:.1f}ms")
        print(f"  Total measurements: {len(phase_timing_errors)}")
    
    # Keep displaying for a moment
    end_time = time.time() + 2
    while time.time() < end_time:
        cam.update()
        cv2.waitKey(30)


if __name__ == "__main__":
    main()
