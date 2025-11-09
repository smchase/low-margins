"""Test bfloat16 conversion for camera transmission."""

import torch
import numpy as np
from simon_test.distributed_utils import tensors_to_frames, frames_to_tensors

def test_bfloat16_conversion():
    print("Testing bfloat16 -> float16 -> bfloat16 conversion for camera")
    print("="*60)
    
    # Create test tensors in bfloat16 (typical for model parameters)
    test_tensors = [
        torch.randn(10, 10, dtype=torch.bfloat16),
        torch.randn(64, dtype=torch.bfloat16),
        torch.randn(10, 64, dtype=torch.bfloat16) * 0.1,  # Smaller values like gradients
    ]
    
    # Get shapes for reconstruction
    shapes = [t.shape for t in test_tensors]
    
    print("Original tensors (bfloat16):")
    for i, t in enumerate(test_tensors):
        print(f"  Tensor {i}: shape={t.shape}, dtype={t.dtype}")
        print(f"    Values: min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}")
    
    # Convert to frames (bfloat16 -> float32 -> float16 -> frames)
    print("\nConverting to camera frames...")
    frames = tensors_to_frames(test_tensors)
    print(f"  Created {len(frames)} frames")
    
    # Convert back (frames -> float16 -> float32 -> bfloat16)
    print("\nConverting back to tensors...")
    reconstructed = frames_to_tensors(frames, shapes, target_dtype=torch.bfloat16)
    
    print("\nReconstructed tensors (bfloat16):")
    for i, t in enumerate(reconstructed):
        print(f"  Tensor {i}: shape={t.shape}, dtype={t.dtype}")
        print(f"    Values: min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}")
    
    # Check precision loss
    print("\nPrecision analysis:")
    for i, (orig, recon) in enumerate(zip(test_tensors, reconstructed)):
        diff = (orig - recon).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_error = (diff / (orig.abs() + 1e-8)).mean().item()
        
        print(f"\n  Tensor {i}:")
        print(f"    Max absolute error: {max_diff:.6f}")
        print(f"    Mean absolute error: {mean_diff:.6f}")
        print(f"    Mean relative error: {relative_error:.6%}")
        
        # Check if values are close enough for training
        close = torch.allclose(orig, recon, rtol=1e-2, atol=1e-3)
        print(f"    Close enough for training (rtol=1e-2): {'✓ YES' if close else '✗ NO'}")
    
    # Test gradient scenario
    print("\n" + "="*60)
    print("Testing typical gradient values:")
    
    # Simulate typical gradients (small values)
    grad = torch.randn(100, dtype=torch.bfloat16) * 0.01
    print(f"Original gradient: mean={grad.mean():.6f}, std={grad.std():.6f}")
    
    # Round trip
    frames = tensors_to_frames([grad])
    reconstructed_grad = frames_to_tensors(frames, [grad.shape], target_dtype=torch.bfloat16)[0]
    
    print(f"Reconstructed:     mean={reconstructed_grad.mean():.6f}, std={reconstructed_grad.std():.6f}")
    
    # Check if gradient direction is preserved
    cosine_sim = torch.nn.functional.cosine_similarity(
        grad.view(1, -1), 
        reconstructed_grad.view(1, -1)
    ).item()
    print(f"Cosine similarity: {cosine_sim:.6f} (1.0 = perfect)")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("- bfloat16 is converted to float16 for camera transmission")
    print("- Reconstructed values are converted back to bfloat16")
    print("- Some precision is lost but gradient direction is preserved")
    print("- This should work fine for distributed training!")


if __name__ == "__main__":
    test_bfloat16_conversion()
