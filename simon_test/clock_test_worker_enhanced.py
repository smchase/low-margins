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
ROOT_SEED = 42  # Must match root's seed to verify what we should receive
WORKER_SEED = 123  # Different from root for different patterns
TARGET_CYCLES = 1  # Should match root node
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
    
    log_status("Clock Test Worker Node (Enhanced) - Starting")
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
    successful_receptions = 0
    successful_validations = 0
    
    # Track reception quality
    reception_stats = []
    tensor_integrity_results = []
    
    while cycle_count < TARGET_CYCLES:
        timing_stats = get_cycle_timing_stats()
        cycle_number = timing_stats["cycle_number"]
        
        # Update overlay display (disabled for cleaner view)
        # phase_name = timing_stats["phase"].split(".")[-1].replace("_", " ").title()
        # overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | {phase_name} | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        
        # Calculate what we expect to receive from root
        expected_from_root = generate_test_tensor(ROOT_SEED, cycle_number)
        
        # Phase 1: Root compute (we just wait)
        wait_for_phase(PhaseType.COMPUTE_ROOT)
        timing_stats = get_cycle_timing_stats()
        # overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Root Compute | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        log_status("Root computing...", timing_stats)
        
        # Phase 2: Receive from root
        phase_start = wait_for_phase(PhaseType.TRANSMIT_ROOT_TO_WORKER)
        timing_stats = get_cycle_timing_stats()
        # overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Receive from Root | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        log_status(f"Receiving {FRAMES_PER_TENSOR} frames from root", timing_stats)
        
        received_frames = []
        frame_idx = 0
        phase_start_time = time.time()
        pixel_errors = []
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Update overlay with current status (disabled)
            # timing_stats = get_cycle_timing_stats()
            # overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Receive [{frame_idx+1}/{FRAMES_PER_TENSOR}] | {timing_stats['seconds_remaining']:.1f}s"
            # cam.set_overlay_text(overlay_text)
            
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
        reception_complete = len(received_frames) == FRAMES_PER_TENSOR
        validation_passed = False
        
        if reception_complete:
            try:
                received_tensor = frames_to_tensor(received_frames)
                log_status(f"Successfully received and decoded tensor", timing_stats)
                
                # REAL VERIFICATION: Check if we received what root should have sent
                matches, num_diff, max_error = verify_tensor_integrity(expected_from_root, received_tensor)
                
                tensor_integrity_results.append({
                    'cycle': cycle_count,
                    'matches': matches,
                    'num_diff': num_diff,
                    'max_error': max_error
                })
                
                if matches:
                    validation_passed = True
                    successful_validations += 1
                    log_status(f"✓ Reception VALIDATED: Received exact tensor from root!", timing_stats)
                else:
                    log_status(f"✗ Reception VALIDATION FAILED: {num_diff} values differ, max error: {max_error}", timing_stats)
                    # Show some examples of mismatches
                    if num_diff > 0 and num_diff < 100:
                        diffs = np.where(expected_from_root != received_tensor)[0]
                        for i in range(min(3, len(diffs))):
                            idx = diffs[i]
                            log_status(f"    Index {idx}: expected {expected_from_root[idx]:.6f}, got {received_tensor[idx]:.6f}")
                
                # Log reception quality
                total_pixel_errors = sum(pixel_errors)
                total_pixels = FRAMES_PER_TENSOR * ROWS * COLS
                pixel_accuracy = 100.0 * (1 - total_pixel_errors / total_pixels)
                reception_stats.append({
                    'cycle': cycle_number,
                    'pixel_errors': total_pixel_errors,
                    'pixel_accuracy': pixel_accuracy,
                    'tensor_valid': validation_passed
                })
                
                log_status(f"Reception quality: {pixel_accuracy:.1f}% pixel accuracy "
                          f"({total_pixel_errors}/{total_pixels} errors)")
                
                successful_receptions += 1
                
            except Exception as e:
                log_status(f"ERROR: Failed to decode frames - {str(e)}", timing_stats)
        else:
            log_status(f"ERROR: Incomplete reception ({len(received_frames)}/{FRAMES_PER_TENSOR} frames)", timing_stats)
        
        # Phase 3: Compute (transform received tensor)
        wait_for_phase(PhaseType.COMPUTE_WORKER)
        timing_stats = get_cycle_timing_stats()
        # overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Worker Compute | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        log_status("Computing response...", timing_stats)
        
        if received_tensor is not None and validation_passed:
            # Apply transformation only if we received valid data
            response_tensor = apply_test_transform(received_tensor, TRANSFORM_TYPE)
            frames_to_send = tensor_to_frames(response_tensor)
            
            log_status(f"Applied '{TRANSFORM_TYPE}' transform", timing_stats)
            log_status(f"Response tensor stats: min={np.min(response_tensor):.3f}, "
                      f"max={np.max(response_tensor):.3f}, mean={np.mean(response_tensor):.3f}")
        else:
            # If we didn't receive properly, generate a dummy response
            log_status("Generating dummy response due to reception failure or validation error", timing_stats)
            response_tensor = generate_test_tensor(WORKER_SEED, cycle_number)
            frames_to_send = tensor_to_frames(response_tensor)
        
        # Phase 4: Transmit to root
        phase_start = wait_for_phase(PhaseType.TRANSMIT_WORKER_TO_ROOT)
        timing_stats = get_cycle_timing_stats()
        # overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Transmit to Root | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        log_status(f"Transmitting {FRAMES_PER_TENSOR} frames to root", timing_stats)
        
        frame_idx = 0
        phase_start_time = time.time()
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Update overlay with current status (disabled)
            # timing_stats = get_cycle_timing_stats()
            # overlay_text = f"WORKER | Cycle {cycle_count+1}/{TARGET_CYCLES} | Transmit [{frame_idx+1}/{FRAMES_PER_TENSOR}] | {timing_stats['seconds_remaining']:.1f}s"
            # cam.set_overlay_text(overlay_text)
            
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
            log_status(f"Successfully transmitted all {FRAMES_PER_TENSOR} frames", timing_stats)
        
        cycle_count += 1
        
        # Wait for next cycle if not the last one
        if cycle_count < TARGET_CYCLES:
            current_phase, _, remaining = get_current_phase_info()
            if remaining > 0:
                time.sleep(remaining)
    
    # Final report
    print("\n" + "="*60)
    print("CLOCK TEST COMPLETE - ENHANCED FINAL REPORT")
    print("="*60)
    print(f"Total cycles attempted: {cycle_count}")
    print(f"Successful receptions: {successful_receptions}")
    print(f"Successful validations (exact tensor match): {successful_validations}")
    print(f"Reception success rate: {100 * successful_receptions / cycle_count:.1f}%")
    print(f"VALIDATION SUCCESS RATE: {100 * successful_validations / cycle_count:.1f}%")
    
    # Data integrity analysis
    if tensor_integrity_results:
        print(f"\nData Integrity Details (Root→Worker):")
        perfect_cycles = sum(1 for r in tensor_integrity_results if r['matches'])
        print(f"  Perfect matches: {perfect_cycles}/{len(tensor_integrity_results)}")
        
        error_cycles = [r for r in tensor_integrity_results if not r['matches'] and r['num_diff'] > 0]
        if error_cycles:
            avg_errors = np.mean([r['num_diff'] for r in error_cycles])
            max_error_overall = max(r['max_error'] for r in error_cycles)
            print(f"  Average values with errors: {avg_errors:.1f} per cycle")
            print(f"  Maximum error magnitude: {max_error_overall}")
    
    # Reception quality statistics
    if reception_stats:
        avg_accuracy = np.mean([s['pixel_accuracy'] for s in reception_stats])
        min_accuracy = np.min([s['pixel_accuracy'] for s in reception_stats])
        max_accuracy = np.max([s['pixel_accuracy'] for s in reception_stats])
        
        print(f"\nReception quality (pixel-level):")
        print(f"  Average pixel accuracy: {avg_accuracy:.2f}%")
        print(f"  Min pixel accuracy: {min_accuracy:.2f}%")
        print(f"  Max pixel accuracy: {max_accuracy:.2f}%")
        
        # Show correlation between pixel errors and tensor validation
        valid_with_errors = [s for s in reception_stats if s['pixel_errors'] > 0 and s['tensor_valid']]
        invalid_with_no_errors = [s for s in reception_stats if s['pixel_errors'] == 0 and not s['tensor_valid']]
        
        if valid_with_errors:
            print(f"\n  NOTE: {len(valid_with_errors)} cycles had pixel errors but still validated correctly")
        if invalid_with_no_errors:
            print(f"  WARNING: {len(invalid_with_no_errors)} cycles had no pixel errors but failed validation!")
    
    # Keep displaying for a moment
    end_time = time.time() + 2
    while time.time() < end_time:
        cam.update()
        cv2.waitKey(30)


if __name__ == "__main__":
    main()
