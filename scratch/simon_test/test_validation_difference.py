"""
Demonstrates the difference between the original and enhanced clock tests.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import (  # noqa: E402
    generate_test_tensor, tensor_to_frames, frames_to_tensor,
    verify_tensor_integrity, apply_test_transform
)


def simulate_transmission_with_errors(tensor, error_rate=0.001):
    """Simulate transmission errors by flipping some bits."""
    frames = tensor_to_frames(tensor)
    
    # Introduce errors in frame data
    for frame in frames:
        if np.random.random() < 0.5:  # 50% chance to corrupt this frame
            # Flip some pixels
            num_pixels = len(frame.data)
            num_errors = int(num_pixels * error_rate)
            if num_errors > 0:
                error_indices = np.random.choice(num_pixels, num_errors, replace=False)
                for idx in error_indices:
                    # Change to a different valid value
                    frame.data[idx] = (frame.data[idx] + np.random.randint(1, 8)) % 8
    
    # Decode back
    received = frames_to_tensor(frames)
    return received


def main():
    print("Demonstrating validation differences between original and enhanced tests")
    print("="*70)
    
    # Generate test tensor
    original_tensor = generate_test_tensor(seed=42, cycle_number=0)
    print(f"\nOriginal tensor: shape={original_tensor.shape}")
    print(f"  Stats: min={np.min(original_tensor):.3f}, max={np.max(original_tensor):.3f}")
    
    # Simulate perfect transmission
    print("\n1. PERFECT TRANSMISSION:")
    perfect_frames = tensor_to_frames(original_tensor)
    perfect_received = frames_to_tensor(perfect_frames)
    
    # Original test would only check:
    shape_match = perfect_received.shape == original_tensor.shape
    print(f"  Original test: shape matches? {shape_match} → PASS ✓")
    
    # Enhanced test checks actual values:
    matches, num_diff, max_error = verify_tensor_integrity(original_tensor, perfect_received)
    print(f"  Enhanced test: values match? {matches} (0 differences) → PASS ✓")
    
    # Simulate transmission with errors
    print("\n2. TRANSMISSION WITH ERRORS:")
    corrupted_received = simulate_transmission_with_errors(original_tensor, error_rate=0.01)
    
    # Original test would only check:
    shape_match = corrupted_received.shape == original_tensor.shape
    print(f"  Original test: shape matches? {shape_match} → PASS ✓ (but data is corrupted!)")
    
    # Enhanced test detects the corruption:
    matches, num_diff, max_error = verify_tensor_integrity(original_tensor, corrupted_received)
    print(f"  Enhanced test: values match? {matches} ({num_diff} differences) → FAIL ✗")
    print(f"    Maximum error: {max_error:.6f}")
    
    # Show some examples of corruption
    if num_diff > 0:
        diffs = np.where(original_tensor != corrupted_received)[0]
        print(f"\n  Example corrupted values:")
        for i in range(min(5, len(diffs))):
            idx = diffs[i]
            print(f"    Index {idx}: {original_tensor[idx]:.6f} → {corrupted_received[idx]:.6f}")
    
    # Test the full cycle (with transformation)
    print("\n3. FULL CYCLE TEST (Root → Worker → Root):")
    
    # Worker receives and transforms
    worker_received = simulate_transmission_with_errors(original_tensor, error_rate=0.005)
    worker_transform = apply_test_transform(worker_received, "multiply")
    
    # Root receives transformed data
    root_received = simulate_transmission_with_errors(worker_transform, error_rate=0.005)
    
    # What root expects (perfect case)
    expected = apply_test_transform(original_tensor, "multiply")
    
    # Original test would pass if shape matches
    shape_match = root_received.shape == expected.shape
    print(f"  Original test: shape matches? {shape_match} → PASS ✓ (ignoring data corruption)")
    
    # Enhanced test checks the actual values
    matches, num_diff, max_error = verify_tensor_integrity(expected, root_received)
    error_percentage = 100 * num_diff / len(expected)
    print(f"  Enhanced test: values match expected? {matches}")
    print(f"    → {num_diff}/{len(expected)} differences ({error_percentage:.2f}%) → FAIL ✗")
    print(f"    → Maximum error: {max_error:.6f}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("- Original test: Only checks if transmission completed (shape matches)")
    print("- Enhanced test: Verifies actual tensor values match expectations")
    print("\nThe '100% success' in original test means transmission completed,")
    print("NOT that the data was transmitted correctly!")


if __name__ == "__main__":
    main()
