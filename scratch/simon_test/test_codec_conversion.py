"""Test script to verify tensor to frame conversion works correctly."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import (  # noqa: E402
    generate_test_tensor, tensor_to_frames, frames_to_tensor,
    verify_tensor_integrity, apply_test_transform,
    TENSOR_SIZE, FRAMES_PER_TENSOR
)


def test_basic_conversion():
    """Test basic tensor to frames and back conversion."""
    print("Testing basic tensor conversion...")
    
    # Generate a test tensor
    tensor = generate_test_tensor(seed=42, cycle_number=0)
    print(f"Original tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"  Stats: min={np.min(tensor):.3f}, max={np.max(tensor):.3f}, mean={np.mean(tensor):.3f}")
    
    # Convert to frames
    frames = tensor_to_frames(tensor)
    print(f"\nConverted to {len(frames)} frames")
    
    # Check frame data validity
    for i, frame in enumerate(frames):
        invalid_pixels = np.sum((frame.data < 0) | (frame.data > 7))
        if invalid_pixels > 0:
            print(f"  WARNING: Frame {i} has {invalid_pixels} invalid pixels!")
        else:
            print(f"  Frame {i}: OK (all pixels in range 0-7)")
    
    # Convert back to tensor
    reconstructed = frames_to_tensor(frames)
    print(f"\nReconstructed tensor: shape={reconstructed.shape}, dtype={reconstructed.dtype}")
    print(f"  Stats: min={np.min(reconstructed):.3f}, max={np.max(reconstructed):.3f}, mean={np.mean(reconstructed):.3f}")
    
    # Verify integrity
    matches, num_diff, max_error = verify_tensor_integrity(tensor, reconstructed)
    
    if matches:
        print(f"\n✓ SUCCESS: Perfect reconstruction!")
    else:
        print(f"\n✗ FAILURE: {num_diff} values differ, max error: {max_error}")
        # Show some examples of differences
        if num_diff > 0:
            diffs = np.where(tensor != reconstructed)[0]
            for i in range(min(5, len(diffs))):
                idx = diffs[i]
                print(f"  Index {idx}: {tensor[idx]} → {reconstructed[idx]}")
    
    return matches


def test_transforms():
    """Test that transforms work correctly."""
    print("\n" + "="*60)
    print("Testing tensor transforms...")
    
    tensor = generate_test_tensor(seed=123, cycle_number=5)
    print(f"Original tensor stats: min={np.min(tensor):.3f}, max={np.max(tensor):.3f}, mean={np.mean(tensor):.3f}")
    
    transforms = ["multiply", "add", "negate"]
    
    for transform in transforms:
        transformed = apply_test_transform(tensor, transform)
        print(f"\nAfter '{transform}' transform:")
        print(f"  Stats: min={np.min(transformed):.3f}, max={np.max(transformed):.3f}, mean={np.mean(transformed):.3f}")
        
        # Test round-trip
        frames = tensor_to_frames(transformed)
        reconstructed = frames_to_tensor(frames)
        matches, _, _ = verify_tensor_integrity(transformed, reconstructed)
        
        if matches:
            print(f"  ✓ Round-trip successful")
        else:
            print(f"  ✗ Round-trip failed!")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("Testing edge cases...")
    
    # Test with extreme values
    print("\nTesting with extreme values...")
    extreme_tensor = np.full(TENSOR_SIZE, np.finfo(np.float16).max, dtype=np.float16)
    extreme_tensor[::2] = np.finfo(np.float16).min  # Alternate min/max
    
    frames = tensor_to_frames(extreme_tensor)
    reconstructed = frames_to_tensor(frames)
    matches, num_diff, max_error = verify_tensor_integrity(extreme_tensor, reconstructed)
    
    if matches:
        print("  ✓ Extreme values handled correctly")
    else:
        print(f"  ✗ Failed with {num_diff} differences, max error: {max_error}")
    
    # Test with zeros
    print("\nTesting with all zeros...")
    zero_tensor = np.zeros(TENSOR_SIZE, dtype=np.float16)
    frames = tensor_to_frames(zero_tensor)
    reconstructed = frames_to_tensor(frames)
    matches, _, _ = verify_tensor_integrity(zero_tensor, reconstructed)
    
    if matches:
        print("  ✓ Zero tensor handled correctly")
    else:
        print("  ✗ Zero tensor failed!")
    
    # Test with NaN/Inf (should fail gracefully)
    print("\nTesting with NaN values...")
    nan_tensor = np.zeros(TENSOR_SIZE, dtype=np.float16)
    nan_tensor[0] = np.nan
    
    try:
        frames = tensor_to_frames(nan_tensor)
        reconstructed = frames_to_tensor(frames)
        print("  ✗ NaN handling needs improvement (should have raised error)")
    except Exception as e:
        print(f"  ✓ NaN correctly rejected: {type(e).__name__}")


def main():
    print("Codec Conversion Test Suite")
    print("="*60)
    
    # Run tests
    basic_ok = test_basic_conversion()
    test_transforms()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if basic_ok:
        print("✓ Basic conversion test PASSED")
        print("\nThe tensor encoding/decoding system is working correctly!")
        print("You can now run the clock synchronization test:")
        print("  1. Terminal 1: python clock_test_root.py")
        print("  2. Terminal 2: python clock_test_worker.py")
        print("  3. Press SPACE in both terminals at the same time to start")
    else:
        print("✗ Basic conversion test FAILED")
        print("Please fix the encoding/decoding issues before running the clock test.")


if __name__ == "__main__":
    main()
