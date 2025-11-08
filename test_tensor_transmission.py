#!/usr/bin/env python3
"""
Test transmitting a 748x64 tensor using codec and virtual camera.

The codec encodes FP16 tensors into multiple grids of integer values.
We'll transmit these grids through the visual data link.
"""

import numpy as np
import threading
import time
import cv2
from codec import codec
from virtual_camera import VirtualCameraCapture
from visual_main import Transmitter, Receiver, FIXED_MESSAGE


def create_test_tensor(rows=748, cols=64):
    """Create a test FP16 tensor"""
    # Create a tensor with some pattern
    x = np.random.randn(rows, cols).astype(np.float16)
    return x


def run_transmitter(vcam, tensor, min_val=0, max_val=15):
    """
    Encode tensor with codec and transmit grids through visual data link.

    For a 748x64 tensor with range [0,15] (K=16), we need 4 grids.
    Each grid is 748x64 = 47,872 values.

    We'll need to determine the best grid size to fit this data.
    """
    print("[TX] Starting transmitter...")
    print(f"[TX] Tensor shape: {tensor.shape}")

    # Initialize codec
    rows, cols = tensor.shape
    c = codec(rows, cols, min_val=min_val, max_val=max_val)
    print(f"[TX] Codec K={c.K}, grids_needed={c.grids_needed()}")

    # Encode tensor into grids
    grids = c.encode(tensor)
    print(f"[TX] Encoded grids shape: {grids.shape}")

    # Calculate grid size needed to fit one grid (748x64 = 47,872 values)
    # We'll use a square grid for simplicity
    values_per_grid = rows * cols
    grid_side = int(np.ceil(np.sqrt(values_per_grid)))
    print(f"[TX] Need {values_per_grid} cells, using {grid_side}x{grid_side} grid")

    # For each encoded grid, transmit it
    for grid_idx in range(c.grids_needed()):
        print(f"\n[TX] Transmitting grid {grid_idx+1}/{c.grids_needed()}")

        # Flatten grid and pad to square
        flat = grids[grid_idx].flatten()
        padded = np.zeros(grid_side * grid_side, dtype=flat.dtype)
        padded[:len(flat)] = flat
        grid_2d = padded.reshape(grid_side, grid_side)

        # Normalize to [0, 1] for visual transmission
        # Grid values are in [min_val, max_val]
        grid_normalized = (grid_2d - min_val) / (max_val - min_val)

        # Create transmitter with appropriate grid size
        # Cell size = 2 pixels for faster transmission
        tx = Transmitter(grid_size=grid_side, cell_size=2)

        # Set the pattern directly (bypass message encoding)
        tx.pattern = (grid_normalized > 0.5).astype(np.uint8)

        # Transmit for a few seconds
        tx.calibration_mode = True
        for frame_idx in range(90):  # 3 seconds of calibration
            img = tx.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vcam.write_frame(img_bgr)
            time.sleep(0.03)

        # Switch to normal mode
        tx.calibration_mode = False
        print(f"[TX] Grid {grid_idx+1} calibration done, transmitting data...")

        for frame_idx in range(90):  # 3 seconds of data
            img = tx.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vcam.write_frame(img_bgr)
            time.sleep(0.03)

            if frame_idx % 30 == 0:
                print(f"[TX] Grid {grid_idx+1} - sent {frame_idx} frames")

    print("\n[TX] All grids transmitted")


def main():
    print("\n" + "="*70)
    print("TENSOR TRANSMISSION TEST - 748x64 FP16 Tensor")
    print("="*70)

    # Create test tensor
    tensor = create_test_tensor(748, 64)
    print(f"Created test tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"Sample values: {tensor[0, :5]}")

    # Create virtual camera
    vcam = VirtualCameraCapture(buffer_size=10)
    print("Virtual camera created")

    # For now, just test the codec encoding/decoding
    print("\n" + "="*70)
    print("Testing codec roundtrip...")
    print("="*70)

    c = codec(748, 64, min_val=0, max_val=15)
    print(f"Codec: K={c.K}, grids_needed={c.grids_needed()}")

    grids = c.encode(tensor)
    print(f"Encoded grids shape: {grids.shape}")
    print(f"Grid dtype: {grids.dtype}")
    print(f"Grid value range: [{grids.min()}, {grids.max()}]")

    decoded = c.decode(grids)
    print(f"Decoded tensor shape: {decoded.shape}, dtype={decoded.dtype}")

    # Check roundtrip
    ok = c.roundtrip_equal(tensor)
    print(f"Roundtrip successful: {ok}")

    if ok:
        print("✓ Codec working correctly!")
    else:
        print("✗ Codec roundtrip failed")
        return

    # Now test transmission (commented out for now - would require large grid support)
    print("\n" + "="*70)
    print("Visual transmission test (requires large grid support)")
    print("="*70)
    print(f"To transmit {grids.shape[0]} grids of {grids.shape[1]}x{grids.shape[2]} each")
    print(f"Would need grid size of ~{int(np.ceil(np.sqrt(748*64)))}x{int(np.ceil(np.sqrt(748*64)))}")
    print("This exceeds current visual data link capabilities (16x16 grid)")

    # Uncomment to test actual transmission (very slow with large grids):
    # thread = threading.Thread(target=run_transmitter, args=(vcam, tensor))
    # thread.daemon = True
    # thread.start()


if __name__ == "__main__":
    main()
