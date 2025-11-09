"""Quick test to verify tensor encoding/decoding works correctly."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tensor_verification_test import (
    generate_deterministic_tensor, tensor_to_frames_simple, 
    frames_to_tensor_simple, apply_transform, verify_tensors,
    TENSOR_SIZE
)


def test_roundtrip():
    print("Testing tensor round-trip encoding/decoding...")
    print("="*60)
    
    # Test 1: Basic round-trip
    print("\n1. Testing basic round-trip:")
    tensor1 = generate_deterministic_tensor(42, 0)
    print(f"   Original: shape={tensor1.shape}, mean={np.mean(tensor1):.6f}, min={np.min(tensor1):.3f}, max={np.max(tensor1):.3f}")
    
    frames = tensor_to_frames_simple(tensor1)
    print(f"   Encoded to {len(frames)} frames")
    
    decoded = frames_to_tensor_simple(frames)
    print(f"   Decoded:  shape={decoded.shape}, mean={np.mean(decoded):.6f}, min={np.min(decoded):.3f}, max={np.max(decoded):.3f}")
    
    matches, num_diff, max_error = verify_tensors(tensor1, decoded)
    if matches:
        print("   ✓ Perfect round-trip!")
    else:
        print(f"   ✗ Round-trip failed: {num_diff} differences, max error: {max_error}")
        # Show some examples
        if num_diff > 0:
            diff_indices = np.where(~np.isclose(tensor1, decoded, rtol=1e-3, atol=1e-3))[0]
            print("\n   Examples of differences:")
            for i in range(min(5, len(diff_indices))):
                idx = diff_indices[i]
                print(f"     [{idx}] Original: {tensor1[idx]:.6f}, Decoded: {decoded[idx]:.6f}, Diff: {abs(tensor1[idx] - decoded[idx]):.6f}")
    
    # Test 2: Transform and round-trip
    print("\n2. Testing transform:")
    transformed = apply_transform(tensor1)
    print(f"   Transformed: mean={np.mean(transformed):.6f} (expected ~2x original)")
    
    frames2 = tensor_to_frames_simple(transformed)
    decoded2 = frames_to_tensor_simple(frames2)
    
    matches2, num_diff2, max_error2 = verify_tensors(transformed, decoded2)
    if matches2:
        print("   ✓ Transform round-trip successful!")
    else:
        print(f"   ✗ Transform round-trip failed: {num_diff2} differences")
    
    # Test 3: Extreme values
    print("\n3. Testing extreme values:")
    extreme_tensor = np.zeros(TENSOR_SIZE, dtype=np.float16)
    extreme_tensor[0] = np.float16(1.0)
    extreme_tensor[1] = np.float16(-1.0)
    extreme_tensor[2] = np.float16(0.0)
    extreme_tensor[3] = np.float16(0.5)
    extreme_tensor[4] = np.float16(-0.5)
    
    frames3 = tensor_to_frames_simple(extreme_tensor)
    decoded3 = frames_to_tensor_simple(frames3)
    
    print(f"   Test values: {extreme_tensor[:5]}")
    print(f"   Decoded:     {decoded3[:5]}")
    
    matches3 = np.allclose(extreme_tensor[:5], decoded3[:5], rtol=1e-3, atol=1e-3)
    if matches3:
        print("   ✓ Extreme values handled correctly!")
    else:
        print("   ✗ Extreme values failed!")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Tensor size: {TENSOR_SIZE} float16 values")
    print(f"  Frames needed: {len(frames)}")
    print(f"  Basic round-trip: {'PASS' if matches else 'FAIL'}")
    print(f"  Transform test: {'PASS' if matches2 else 'FAIL'}")
    print(f"  Extreme values: {'PASS' if matches3 else 'FAIL'}")


if __name__ == "__main__":
    test_roundtrip()
