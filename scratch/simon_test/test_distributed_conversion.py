"""Test script to verify distributed training tensor conversions."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.model import MLP  # noqa: E402
from simon_test.distributed_utils import (  # noqa: E402
    tensors_to_frames, frames_to_tensors, get_parameter_shapes,
    FLOATS_PER_FRAME
)


def test_parameter_conversion():
    """Test converting model parameters to frames and back."""
    print("Testing parameter/gradient conversion for distributed training")
    print("="*60)
    
    # Create model
    model = MLP().bfloat16()
    param_shapes = get_parameter_shapes(model)
    
    # Get parameter info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params}")
    print(f"  Parameter tensors: {len(list(model.parameters()))}")
    print(f"  Floats per frame: {FLOATS_PER_FRAME}")
    
    # Test parameter conversion
    print("\n1. Testing parameter conversion:")
    original_params = [p.clone() for p in model.parameters()]
    
    # Convert to frames
    frames = tensors_to_frames(original_params)
    print(f"   Encoded to {len(frames)} frames")
    print(f"   Expected frames: {(total_params + FLOATS_PER_FRAME - 1) // FLOATS_PER_FRAME}")
    
    # Convert back
    decoded_params = frames_to_tensors(frames, param_shapes)
    print(f"   Decoded {len(decoded_params)} parameter tensors")
    
    # Verify shapes
    shapes_match = all(
        orig.shape == dec.shape 
        for orig, dec in zip(original_params, decoded_params)
    )
    print(f"   Shapes match: {shapes_match}")
    
    # Verify values
    all_close = True
    max_diff = 0.0
    
    for i, (orig, dec) in enumerate(zip(original_params, decoded_params)):
        if not torch.allclose(orig.float(), dec.float(), rtol=1e-3, atol=1e-3):
            all_close = False
            diff = torch.abs(orig.float() - dec.float()).max().item()
            max_diff = max(max_diff, diff)
            print(f"   Parameter {i} mismatch: max diff = {diff}")
    
    if all_close:
        print("   ✓ Perfect parameter round-trip!")
    else:
        print(f"   ✗ Parameter mismatch, max diff: {max_diff}")
    
    # Test gradient conversion
    print("\n2. Testing gradient conversion:")
    
    # Create fake gradients
    fake_gradients = []
    for p in model.parameters():
        # Create gradient with similar magnitude to real training
        grad = torch.randn_like(p) * 0.1
        fake_gradients.append(grad)
    
    # Convert to frames
    grad_frames = tensors_to_frames(fake_gradients)
    print(f"   Encoded gradients to {len(grad_frames)} frames")
    
    # Convert back
    decoded_grads = frames_to_tensors(grad_frames, param_shapes)
    print(f"   Decoded {len(decoded_grads)} gradient tensors")
    
    # Verify gradient values
    grad_close = True
    max_grad_diff = 0.0
    
    for i, (orig, dec) in enumerate(zip(fake_gradients, decoded_grads)):
        if not torch.allclose(orig.float(), dec.float(), rtol=1e-3, atol=1e-3):
            grad_close = False
            diff = torch.abs(orig.float() - dec.float()).max().item()
            max_grad_diff = max(max_grad_diff, diff)
    
    if grad_close:
        print("   ✓ Perfect gradient round-trip!")
    else:
        print(f"   ✗ Gradient mismatch, max diff: {max_grad_diff}")
    
    # Test with None gradients (can happen in training)
    print("\n3. Testing with None gradients:")
    mixed_gradients = fake_gradients.copy()
    mixed_gradients[1] = None  # Set one gradient to None
    
    # This should still work
    try:
        mixed_frames = tensors_to_frames(mixed_gradients)
        decoded_mixed = frames_to_tensors(mixed_frames, param_shapes)
        print("   ✓ Handled None gradients correctly")
    except Exception as e:
        print(f"   ✗ Failed with None gradients: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Model parameters: {total_params} float16 values")
    print(f"  Frames needed: {len(frames)} (at {FLOATS_PER_FRAME} values/frame)")
    print(f"  Round-trip success: {'YES' if all_close and grad_close else 'NO'}")
    
    if all_close and grad_close:
        print("\n✓ Distributed training tensor conversion working correctly!")
    else:
        print("\n✗ Issues detected - check encoding/decoding logic")


if __name__ == "__main__":
    test_parameter_conversion()
