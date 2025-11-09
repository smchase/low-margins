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
    verify_tensor_integrity, apply_test_transform, get_cycle_timing_stats,
    CYCLE_TIME, FRAMES_PER_TENSOR
)

# Worker node configuration
WORKER_SEED = 123  # Different from root for different patterns
TARGET_CYCLES = 10  # Should match root node
TRANSFORM_TYPE = "multiply"  # Transform to apply to received tensors


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
    
    log_status("Clock Test Worker Node - Starting")
    log_status(f"Target: {TARGET_CYCLES} complete cycles")
    log_status(f"Cycle time: {CYCLE_TIME}s")
    log_status(f"Transform: {TRANSFORM_TYPE}")
    
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
    
    # Wait for the start of the next cycle (synchronized with root)
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
    
    # Track reception quality
    reception_stats = []
    
    while cycle_count < TARGET_CYCLES:
        timing_stats = get_cycle_timing_stats()
        cycle_number = timing_stats["cycle_number"]
        
        # Update overlay display
        phase_name = timing_stats["phase"].split(".")[-1].replace("_", " ").title()
        overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | {phase_name} | {timing_stats['seconds_remaining']:.1f}s"
        cam.set_overlay_text(overlay_text)
        
        # Phase 1: Root compute (we just wait)
        wait_for_phase(PhaseType.COMPUTE_ROOT)
        timing_stats = get_cycle_timing_stats()
        overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Root Compute | {timing_stats['seconds_remaining']:.1f}s"
        cam.set_overlay_text(overlay_text)
        log_status("Root computing...", timing_stats)
        
        # Phase 2: Receive from root
        phase_start = wait_for_phase(PhaseType.TRANSMIT_ROOT_TO_WORKER)
        timing_stats = get_cycle_timing_stats()
        overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Receive from Root | {timing_stats['seconds_remaining']:.1f}s"
        cam.set_overlay_text(overlay_text)
        log_status(f"Receiving {FRAMES_PER_TENSOR} frames from root", timing_stats)
        
        received_frames = []
        frame_idx = 0
        phase_start_time = time.time()
        pixel_errors = []
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Update overlay with current status
            timing_stats = get_cycle_timing_stats()
            overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Receive [{frame_idx+1}/{FRAMES_PER_TENSOR}] | {timing_stats['seconds_remaining']:.1f}s"
            cam.set_overlay_text(overlay_text)
            
            # Check if we're still in the correct phase
            current_phase, _, _ = get_current_phase_info()
            if current_phase != PhaseType.TRANSMIT_ROOT_TO_WORKER:
                log_status(f"WARNING: Phase changed during reception! Only received {frame_idx}/{FRAMES_PER_TENSOR} frames")
                break
            
            # Receive frame at midpoint of each frame period
            elapsed = time.time() - phase_start_time
            target_time = (frame_idx + 0.5) * SECONDS_PER_FRAME
            
            if elapsed >= target_time:
                received_frame = cam.receive()
                received_frames.append(received_frame)
                
                # Check frame validity (all values should be 0-7)
                invalid_pixels = np.sum((received_frame.data < 0) | (received_frame.data > 7))
                pixel_errors.append(invalid_pixels)
                
                if invalid_pixels > 0:
                    log_status(f"  Frame {frame_idx}: {invalid_pixels} invalid pixels")
                
                frame_idx += 1
            
            cv2.waitKey(30)
        
        # Process received data
        received_tensor = None
        if len(received_frames) == FRAMES_PER_TENSOR:
            try:
                received_tensor = frames_to_tensor(received_frames)
                log_status(f"Successfully received and decoded tensor", timing_stats)
                
                # Log reception quality
                total_pixel_errors = sum(pixel_errors)
                total_pixels = FRAMES_PER_TENSOR * ROWS * COLS
                pixel_accuracy = 100.0 * (1 - total_pixel_errors / total_pixels)
                reception_stats.append({
                    'cycle': cycle_number,
                    'pixel_errors': total_pixel_errors,
                    'pixel_accuracy': pixel_accuracy
                })
                
                log_status(f"Reception quality: {pixel_accuracy:.1f}% pixel accuracy "
                          f"({total_pixel_errors}/{total_pixels} errors)")
                
            except Exception as e:
                log_status(f"ERROR: Failed to decode frames - {str(e)}", timing_stats)
        else:
            log_status(f"ERROR: Incomplete reception ({len(received_frames)}/{FRAMES_PER_TENSOR} frames)", timing_stats)
        
        # Phase 3: Compute (transform received tensor)
        wait_for_phase(PhaseType.COMPUTE_WORKER)
        timing_stats = get_cycle_timing_stats()
        overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Worker Compute | {timing_stats['seconds_remaining']:.1f}s"
        cam.set_overlay_text(overlay_text)
        log_status("Computing response...", timing_stats)
        
        if received_tensor is not None:
            # Apply transformation
            response_tensor = apply_test_transform(received_tensor, TRANSFORM_TYPE)
            frames_to_send = tensor_to_frames(response_tensor)
            
            log_status(f"Applied '{TRANSFORM_TYPE}' transform", get_cycle_timing_stats())
            log_status(f"Response tensor stats: min={np.min(response_tensor):.3f}, "
                      f"max={np.max(response_tensor):.3f}, mean={np.mean(response_tensor):.3f}")
        else:
            # If we didn't receive properly, generate a dummy response
            log_status("Generating dummy response due to reception failure", get_cycle_timing_stats())
            response_tensor = generate_test_tensor(WORKER_SEED, cycle_number)
            frames_to_send = tensor_to_frames(response_tensor)
        
        # Phase 4: Transmit to root
        phase_start = wait_for_phase(PhaseType.TRANSMIT_WORKER_TO_ROOT)
        timing_stats = get_cycle_timing_stats()
        overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Transmit to Root | {timing_stats['seconds_remaining']:.1f}s"
        cam.set_overlay_text(overlay_text)
        log_status(f"Transmitting {FRAMES_PER_TENSOR} frames to root", timing_stats)
        
        frame_idx = 0
        phase_start_time = time.time()
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Update overlay with current status
            timing_stats = get_cycle_timing_stats()
            overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Transmit [{frame_idx+1}/{FRAMES_PER_TENSOR}] | {timing_stats['seconds_remaining']:.1f}s"
            cam.set_overlay_text(overlay_text)
            
            # Check if we're still in the correct phase
            current_phase, _, _ = get_current_phase_info()
            if current_phase != PhaseType.TRANSMIT_WORKER_TO_ROOT:
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
            log_status(f"Successfully transmitted all {FRAMES_PER_TENSOR} frames", get_cycle_timing_stats())
            if received_tensor is not None:
                successful_cycles += 1
        
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
    
    # Reception quality statistics
    if reception_stats:
        avg_accuracy = np.mean([s['pixel_accuracy'] for s in reception_stats])
        min_accuracy = np.min([s['pixel_accuracy'] for s in reception_stats])
        max_accuracy = np.max([s['pixel_accuracy'] for s in reception_stats])
        
        print(f"\nReception quality:")
        print(f"  Average pixel accuracy: {avg_accuracy:.2f}%")
        print(f"  Min pixel accuracy: {min_accuracy:.2f}%")
        print(f"  Max pixel accuracy: {max_accuracy:.2f}%")
        
        # Show per-cycle stats if there were errors
        if any(s['pixel_errors'] > 0 for s in reception_stats):
            print(f"\nPer-cycle reception errors:")
            for stat in reception_stats:
                if stat['pixel_errors'] > 0:
                    print(f"  Cycle {stat['cycle']}: {stat['pixel_errors']} errors "
                          f"({stat['pixel_accuracy']:.1f}% accuracy)")
    
    # Keep displaying for a moment
    end_time = time.time() + 2
    while time.time() < end_time:
        cam.update()
        cv2.waitKey(30)


if __name__ == "__main__":
    main()
