import sys
import time
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera, SECONDS_PER_FRAME  # noqa: E402
from test_utils import (  # noqa: E402
    PhaseType, get_current_phase_info, wait_for_phase,
    generate_test_tensor, tensor_to_frames, frames_to_tensor,
    verify_tensor_integrity, apply_test_transform, get_cycle_timing_stats,
    compute_tensor_checksum, verify_tensor_bitwise,
    CYCLE_TIME, FRAMES_PER_TENSOR
)

# Root node seed for deterministic tensor generation
ROOT_SEED = 42
WORKER_SEED = 123  # Must match worker's seed
TARGET_CYCLES = 1  # Number of cycles to run for testing
EXPECTED_TRANSFORM = "multiply"  # Must match worker's transform


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
    
    log_status("Clock Test Root Node (Enhanced) - Starting")
    log_status(f"Target: {TARGET_CYCLES} complete cycles")
    log_status(f"Cycle time: {CYCLE_TIME}s")
    log_status(f"Expected transform from worker: {EXPECTED_TRANSFORM}")
    
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
    successful_transmissions = 0
    successful_validations = 0
    
    # Track timing accuracy
    phase_timing_errors = []
    
    # Track data integrity
    tensor_integrity_results = []
    
    while cycle_count < TARGET_CYCLES:
        timing_stats = get_cycle_timing_stats()
        cycle_number = timing_stats["cycle_number"]
        
        # Update overlay display (disabled for cleaner view)
        # phase_name = timing_stats["phase"].split(".")[-1].replace("_", " ").title()
        # overlay_text = f"ROOT | Cycle {cycle_count+1}/{TARGET_CYCLES} | {phase_name} | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        
        # Phase 1: Compute (generate test tensor)
        phase_start = wait_for_phase(PhaseType.COMPUTE_ROOT)
        timing_error = phase_start - (cycle_number * CYCLE_TIME + start_time % CYCLE_TIME)
        phase_timing_errors.append(("compute_root", timing_error))
        
        # Generate the test tensor for this cycle (DETERMINISTIC)
        test_tensor = generate_test_tensor(ROOT_SEED, cycle_number)
        frames_to_send = tensor_to_frames(test_tensor)
        sent_checksum = compute_tensor_checksum(test_tensor)
        
        log_status(f"Generated test tensor for cycle {cycle_number}", timing_stats)
        log_status(f"  Shape: {test_tensor.shape}, dtype: {test_tensor.dtype}")
        log_status(f"  Stats: min={np.min(test_tensor):.6f}, max={np.max(test_tensor):.6f}, "
                  f"mean={np.mean(test_tensor):.6f}, std={np.std(test_tensor):.6f}")
        log_status(f"  Checksum: 0x{sent_checksum:08X}")
        
        # Calculate what we expect to receive back (DETERMINISTIC)
        expected_response = apply_test_transform(test_tensor, EXPECTED_TRANSFORM)
        expected_response_checksum = compute_tensor_checksum(expected_response)
        log_status(f"  Expected response checksum: 0x{expected_response_checksum:08X}")
        
        # Phase 2: Transmit to worker
        phase_start_time = wait_for_phase(PhaseType.TRANSMIT_ROOT_TO_WORKER)
        timing_error = phase_start_time - (cycle_number * CYCLE_TIME + 0.5 + start_time % CYCLE_TIME)
        phase_timing_errors.append(("transmit_to_worker", timing_error))
        
        # Update overlay for transmit phase (disabled)
        # timing_stats = get_cycle_timing_stats()
        # overlay_text = f"ROOT | Cycle {cycle_count+1}/{TARGET_CYCLES} | Transmit to Worker | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        
        log_status(f"Transmitting {FRAMES_PER_TENSOR} frames to worker", timing_stats)
        
        frame_idx = 0
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Update overlay with current status (disabled)
            # timing_stats = get_cycle_timing_stats()
            # overlay_text = f"ROOT | Cycle {cycle_count+1}/{TARGET_CYCLES} | Transmit [{frame_idx+1}/{FRAMES_PER_TENSOR}] | {timing_stats['seconds_remaining']:.1f}s"
            # cam.set_overlay_text(overlay_text)
            
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
        
        transmission_complete = (frame_idx == FRAMES_PER_TENSOR)
        if transmission_complete:
            log_status(f"Successfully transmitted all {FRAMES_PER_TENSOR} frames", timing_stats)
        
        # Phase 3: Worker compute (we just wait)
        wait_for_phase(PhaseType.COMPUTE_WORKER)
        timing_stats = get_cycle_timing_stats()
        # overlay_text = f"ROOT | Cycle {cycle_count+1}/{TARGET_CYCLES} | Worker Compute | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        log_status("Worker computing...", timing_stats)
        
        # Phase 4: Receive from worker
        phase_start_time = wait_for_phase(PhaseType.TRANSMIT_WORKER_TO_ROOT)
        timing_error = phase_start_time - (cycle_number * CYCLE_TIME + 6.0 + start_time % CYCLE_TIME)
        phase_timing_errors.append(("receive_from_worker", timing_error))
        
        # Update overlay for receive phase (disabled)
        # timing_stats = get_cycle_timing_stats()
        # overlay_text = f"ROOT | Cycle {cycle_count+1}/{TARGET_CYCLES} | Receive from Worker | {timing_stats['seconds_remaining']:.1f}s"
        # cam.set_overlay_text(overlay_text)
        
        log_status(f"Receiving {FRAMES_PER_TENSOR} frames from worker", get_cycle_timing_stats())
        
        received_frames = []
        frame_idx = 0
        
        while frame_idx < FRAMES_PER_TENSOR:
            cam.update()
            
            # Update overlay with current status (disabled)
            # timing_stats = get_cycle_timing_stats()
            # overlay_text = f"ROOT | Cycle {cycle_count+1}/{TARGET_CYCLES} | Receive [{frame_idx+1}/{FRAMES_PER_TENSOR}] | {timing_stats['seconds_remaining']:.1f}s"
            # cam.set_overlay_text(overlay_text)
            
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
        reception_complete = len(received_frames) == FRAMES_PER_TENSOR
        validation_passed = False
        
        if transmission_complete and reception_complete:
            successful_transmissions += 1
            
            try:
                received_tensor = frames_to_tensor(received_frames)
                received_checksum = compute_tensor_checksum(received_tensor)
                
                log_status(f"Received tensor checksum: 0x{received_checksum:08X}", get_cycle_timing_stats())
                
                # BIT-EXACT VERIFICATION: Check if received tensor matches expected transformation
                bitwise_match, num_diff, details = verify_tensor_bitwise(expected_response, received_tensor)
                
                # Also do float comparison for backward compatibility
                float_matches, float_num_diff, float_max_error = verify_tensor_integrity(expected_response, received_tensor)
                
                tensor_integrity_results.append({
                    'cycle': cycle_count,
                    'matches': bitwise_match,
                    'num_diff': num_diff,
                    'max_error': float_max_error if not bitwise_match else 0.0
                })
                
                if bitwise_match:
                    successful_validations += 1
                    log_status(f"✓ Cycle {cycle_number} VALIDATED (BIT-EXACT): {details}", get_cycle_timing_stats())
                    log_status(f"  Checksums match: 0x{expected_response_checksum:08X} == 0x{received_checksum:08X}")
                else:
                    log_status(f"✗ Cycle {cycle_number} VALIDATION FAILED (BIT-LEVEL): {details}", get_cycle_timing_stats())
                    log_status(f"  Expected checksum: 0x{expected_response_checksum:08X}")
                    log_status(f"  Received checksum: 0x{received_checksum:08X}")
                    
                    # Show some examples of mismatches
                    if num_diff > 0 and num_diff < 100:
                        expected_u16 = expected_response.view(np.uint16)
                        received_u16 = received_tensor.view(np.uint16)
                        diff_mask = expected_u16 != received_u16
                        diff_indices = np.where(diff_mask)[0]
                        
                        for i in range(min(5, len(diff_indices))):
                            idx = diff_indices[i]
                            exp_bits = expected_u16[idx]
                            rec_bits = received_u16[idx]
                            exp_float = expected_response.flat[idx]
                            rec_float = received_tensor.flat[idx]
                            xor = exp_bits ^ rec_bits
                            bit_flips = bin(xor).count('1')
                            log_status(f"    [{idx}] Expected: {exp_float:.6f} (0x{exp_bits:04X}) | "
                                      f"Got: {rec_float:.6f} (0x{rec_bits:04X}) | "
                                      f"{bit_flips} bits differ")
                
                # Log tensor statistics
                log_status(f"  Expected: min={np.min(expected_response):.6f}, "
                          f"max={np.max(expected_response):.6f}, mean={np.mean(expected_response):.6f}")
                log_status(f"  Received: min={np.min(received_tensor):.6f}, "
                          f"max={np.max(received_tensor):.6f}, mean={np.mean(received_tensor):.6f}")
                
            except Exception as e:
                log_status(f"Cycle {cycle_number} FAILED: Error decoding frames - {str(e)}", get_cycle_timing_stats())
        else:
            log_status(f"Cycle {cycle_number} FAILED: Incomplete transmission/reception", get_cycle_timing_stats())
            tensor_integrity_results.append({
                'cycle': cycle_count,
                'matches': False,
                'num_diff': -1,
                'max_error': float('inf')
            })
        
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
    print(f"Successful transmissions (both ways): {successful_transmissions}")
    print(f"Successful validations (data integrity): {successful_validations}")
    print(f"Transmission success rate: {100 * successful_transmissions / cycle_count:.1f}%")
    print(f"VALIDATION SUCCESS RATE: {100 * successful_validations / cycle_count:.1f}%")
    
    # Detailed integrity analysis
    if tensor_integrity_results:
        print(f"\nData Integrity Details:")
        perfect_cycles = sum(1 for r in tensor_integrity_results if r['matches'])
        print(f"  Perfect matches: {perfect_cycles}/{len(tensor_integrity_results)}")
        
        error_cycles = [r for r in tensor_integrity_results if not r['matches'] and r['num_diff'] > 0]
        if error_cycles:
            avg_errors = np.mean([r['num_diff'] for r in error_cycles])
            max_error_overall = max(r['max_error'] for r in error_cycles)
            print(f"  Average values with errors: {avg_errors:.1f} per cycle")
            print(f"  Maximum error magnitude: {max_error_overall}")
            
            print("\n  Per-cycle errors:")
            for r in tensor_integrity_results:
                if not r['matches']:
                    if r['num_diff'] > 0:
                        print(f"    Cycle {r['cycle']}: {r['num_diff']} errors, max error: {r['max_error']:.6f}")
                    else:
                        print(f"    Cycle {r['cycle']}: Transmission/reception failed")
    
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
