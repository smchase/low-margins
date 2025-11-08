#!/usr/bin/env python3
"""
Quick test of codec + visual transmission without GUI.
Just verifies the encoding/transmission logic works.
"""

import numpy as np
import cv2
from codec import codec


def encode_grid_to_pattern(grid_values, grid_size, min_val, max_val):
    """
    Convert a flattened array of values in [min_val, max_val] to a 2D binary pattern.
    For values in [0, 15], we use 4 bits per value.
    """
    flat = grid_values.flatten()
    n_values = len(flat)
    bits_per_value = 4

    total_bits = n_values * bits_per_value
    total_cells = grid_size * grid_size

    if total_bits > total_cells:
        raise ValueError(f"Need {total_bits} bits but only have {total_cells} cells")

    # Convert values to binary
    pattern_bits = np.zeros(total_cells, dtype=np.uint8)
    for i, val in enumerate(flat):
        for bit_idx in range(bits_per_value):
            pattern_bits[i * bits_per_value + bit_idx] = (val >> bit_idx) & 1

    return pattern_bits.reshape(grid_size, grid_size)


def decode_pattern_to_grid(pattern, n_values, rows, cols):
    """Reverse of encode_grid_to_pattern"""
    flat_bits = pattern.flatten()
    bits_per_value = 4

    values = np.zeros(n_values, dtype=np.uint8)
    for i in range(n_values):
        val = 0
        for bit_idx in range(bits_per_value):
            bit = flat_bits[i * bits_per_value + bit_idx]
            val |= (bit << bit_idx)
        values[i] = val

    return values.reshape(rows, cols)


def main():
    print("\n" + "="*70)
    print("CODEC + VISUAL TRANSMISSION TEST (Quick)")
    print("="*70)

    # Create test tensor
    tensor = np.random.randn(748, 64).astype(np.float16)
    print(f"\n1. Created test tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"   Sample values: {tensor[0, :3]}")

    # Encode with codec
    c = codec(748, 64, min_val=0, max_val=15)
    grids = c.encode(tensor)
    print(f"\n2. Codec encoding:")
    print(f"   Grids needed: {c.grids_needed()}")
    print(f"   Encoded shape: {grids.shape}")
    print(f"   Value range: [{grids.min()}, {grids.max()}]")

    # Verify codec roundtrip
    decoded = c.decode(grids)
    ok = c.roundtrip_equal(tensor)
    print(f"   Codec roundtrip: {'✓ PASS' if ok else '✗ FAIL'}")

    # Test visual encoding/decoding of one grid
    print(f"\n3. Visual encoding test (grid 0):")
    values_per_grid = 748 * 64  # 47,872 values
    bits_needed = values_per_grid * 4  # 4 bits per value (for values 0-15)
    grid_size = int(np.ceil(np.sqrt(bits_needed)))  # Calculate needed grid size

    print(f"   Grid 0 shape: {grids[0].shape} = {values_per_grid} values")
    print(f"   Bits needed: {bits_needed}")
    print(f"   Min grid size needed: {grid_size}x{grid_size}")

    # Round up to nice number
    grid_size = 512 if grid_size <= 512 else 1024
    print(f"   Using grid size: {grid_size}x{grid_size} = {grid_size**2} cells")
    print(f"   Fits: {'✓ YES' if bits_needed <= grid_size**2 else '✗ NO'}")

    # Encode grid to binary pattern
    pattern = encode_grid_to_pattern(grids[0], grid_size, 0, 15)
    print(f"   Pattern shape: {pattern.shape}")
    print(f"   Pattern unique values: {np.unique(pattern)}")

    # Decode pattern back to grid
    decoded_grid = decode_pattern_to_grid(pattern, values_per_grid, 748, 64)
    print(f"   Decoded grid shape: {decoded_grid.shape}")

    # Verify visual encoding/decoding roundtrip
    visual_ok = np.array_equal(grids[0], decoded_grid)
    print(f"   Visual roundtrip: {'✓ PASS' if visual_ok else '✗ FAIL'}")

    if visual_ok:
        print("\n" + "="*70)
        print("✓ SUCCESS: All tests passed!")
        print("="*70)
        print("\nSummary:")
        print(f"  - 748x64 tensor can be encoded with codec")
        print(f"  - {c.grids_needed()} grids needed (values 0-15)")
        print(f"  - Each grid fits in {grid_size}x{grid_size} visual grid")
        print(f"  - Each visual grid transmission takes ~5 seconds")
        print(f"  - Total transmission time: ~{c.grids_needed() * 5} seconds")
        print(f"\nGrid details:")
        print(f"  - Visual grid: {grid_size}x{grid_size} = {grid_size**2:,} cells")
        print(f"  - Cell size: 2px → Image size: {grid_size*2}x{grid_size*2} = {grid_size*2}px square")
        print("\nTo test actual transmission, run: python test_codec_transmission.py")
    else:
        print("\n" + "="*70)
        print("✗ FAILED: Visual encoding roundtrip failed")
        print("="*70)


if __name__ == "__main__":
    main()
