#!/usr/bin/env python3
"""
Test codec transmission with composite camera using 8-color encoding.

Uses real visual_main.py receiver logic with codec-encoded tensor transmission.

Transmission approach:
- Codec encodes a TEST_TENSOR_ROWS×TEST_TENSOR_COLS tensor into 4 grids of values [0-15]
- Each grid is spatially sliced into VISUAL_GRID_SIZE×VISUAL_GRID_SIZE chunks
- Each slice transmitted using 8-color encoding (2 color frames per slice)
- Visual grid: VISUAL_GRID_SIZE×VISUAL_GRID_SIZE cells (configured in config.py)
- 8 colors = 3 bits/cell: Frame 0 (lower 3 bits), Frame 1 (upper 1 bit)
- Total frames: 4 grids × (ceil(TEST_TENSOR_ROWS/VISUAL_GRID_SIZE) × ceil(TEST_TENSOR_COLS/VISUAL_GRID_SIZE)) × 2 color frames
  (With the current config this is 4 × 1 × 2 = 8 frames, ~4 seconds at 2 FPS)

Workflow:
1. Press 'c' to calibrate (lock onto grid)
2. Press 'p' to start auto-transmission at TRANSMISSION_FPS FPS
3. Frames auto-advance and auto-capture
4. After all frames received, decode tensor and verify
"""

import sys
import cv2
import numpy as np
from config import (
    VISUAL_GRID_SIZE,
    TRANSMISSION_FPS,
    TEST_TENSOR_ROWS,
    TEST_TENSOR_COLS,
)
from codec_composite_camera import CodecCompositeCamera, COLORS_8
from visual_main import Receiver
from codec import codec


def read_color_pattern(frame, locked_corners, grid_size, border_ratio=0.05):
    """
    Read color pattern from locked region.

    Args:
        frame: Input frame
        locked_corners: 4 corner points of locked grid
        grid_size: Grid size (e.g., VISUAL_GRID_SIZE)
        border_ratio: Border ratio for sampling (default: 0.05)

    Returns:
        grid_size×grid_size array of color indices [0-7], or None if failed
    """
    # Get perspective transform
    src_pts = locked_corners.astype(np.float32)
    size = 512  # Transform to 512×512 for easy sampling
    dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp to top-down view
    warped = cv2.warpPerspective(frame, M, (size, size))

    # Sample each cell
    cell_size = size / grid_size
    border = int(cell_size * border_ratio)
    color_indices = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for i in range(grid_size):
        for j in range(grid_size):
            # Get cell center region (avoid borders)
            y1 = int(i * cell_size + border)
            y2 = int((i + 1) * cell_size - border)
            x1 = int(j * cell_size + border)
            x2 = int((j + 1) * cell_size - border)

            # Sample cell color (mean of center region)
            cell_region = warped[y1:y2, x1:x2]
            mean_color = cell_region.mean(axis=(0, 1))  # BGR mean

            # Find closest color from palette
            min_dist = float('inf')
            best_idx = 0
            for idx, palette_color in enumerate(COLORS_8):
                # Distance in BGR space
                dist = np.sum((mean_color - palette_color) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx

            color_indices[i, j] = best_idx

    return color_indices


def main():
    print("\n" + "="*70)
    print(f"CODEC COMPOSITE CAMERA TEST - {TEST_TENSOR_ROWS}x{TEST_TENSOR_COLS} Tensor Transmission")
    print("="*70)

    # Create test tensor
    tensor = np.random.randn(TEST_TENSOR_ROWS, TEST_TENSOR_COLS).astype(np.float16)
    print(f"Test tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"Sample values: {tensor[0, :3]}")

    # Create codec composite camera (visual grid size from config)
    cap = CodecCompositeCamera(
        tensor,
        cell_size=30,
        grid_size=VISUAL_GRID_SIZE,
        fps=TRANSMISSION_FPS,
    )

    # Create receiver with matching grid size
    rx = Receiver(grid_size=VISUAL_GRID_SIZE)
    print(f"Receiver configured for {rx.grid_size}x{rx.grid_size} grid")

    num_grids = cap.encoded_grids.shape[0]
    num_row_slices = cap.num_row_slices
    num_col_slices = cap.num_col_slices
    slices_per_grid = cap.num_slices

    print("\n" + "="*70)
    print("CONTROLS:")
    print("  c   - Calibrate (lock onto grid)")
    print(f"  p   - Start auto-transmission at {TRANSMISSION_FPS} FPS")
    print("  f   - Toggle horizontal flip")
    print("  +/= - Increase border offset")
    print("  -   - Decrease border offset")
    print("  q   - Quit")
    print("="*70)
    print("\nReady. Point camera at screen, then press 'c' to calibrate.\n")
    print("After calibration, press 'p' to start. Frames will auto-capture.")
    print()

    # Storage for received color frames
    # received_color_frames[grid][row_slice][col_slice][color_frame_idx] = VISUAL_GRID_SIZE×VISUAL_GRID_SIZE color indices
    received_color_frames = [
        [
            [[] for _ in range(num_col_slices)]
            for _ in range(num_row_slices)
        ]
        for _ in range(num_grids)
    ]

    # Frame change detection
    last_frame_id = None
    last_pattern_hash = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Auto-capture when frame changes (if calibrated and transmitting)
        if rx.calibrated and rx.locked_corners is not None and cap.transmitting:
            # Get current frame ID
            current_frame_id = cap.get_current_frame_id()

            # Detect frame change
            if current_frame_id != last_frame_id and current_frame_id >= 0:
                last_frame_id = current_frame_id

                # Read and store color pattern
                color_pattern = read_color_pattern(
                    frame,
                    rx.locked_corners,
                    VISUAL_GRID_SIZE,
                    rx.border_ratio,
                )
                if color_pattern is not None:
                    grid_idx = cap.current_grid_idx
                    row_slice_idx = cap.current_row_slice
                    col_slice_idx = cap.current_col_slice
                    color_frame_idx = cap.current_color_frame
                    received_color_frames[grid_idx][row_slice_idx][col_slice_idx].append(color_pattern.copy())

                    # Count total received
                    total_received = sum(
                        sum(
                            sum(len(frames) for frames in col_slices)
                            for col_slices in row_slices
                        )
                        for row_slices in received_color_frames
                    )
                    total_needed = num_grids * slices_per_grid * 2
                    print(
                        "✓ Auto-captured "
                        f"grid {grid_idx+1}/{num_grids}, "
                        f"row {row_slice_idx+1}/{num_row_slices}, "
                        f"col {col_slice_idx+1}/{num_col_slices}, "
                        f"frame {color_frame_idx}/2 "
                        f"({total_received}/{total_needed} total)"
                    )

            # Check if done
            if current_frame_id == -1 and last_frame_id != -1:
                last_frame_id = -1
                # Done frame detected!
                print("\n" + "="*70)
                print("✓ DONE FRAME DETECTED - DECODING...")
                print("="*70)

                try:
                    # Get tensor dimensions
                    rows, cols = tensor.shape

                    # Reconstruct full grids from slices
                    decoded_grids = []
                    for grid_idx in range(num_grids):
                        reconstructed_rows = []
                        for row_slice_idx in range(num_row_slices):
                            row_blocks = []
                            for col_slice_idx in range(num_col_slices):
                                color_frames = received_color_frames[grid_idx][row_slice_idx][col_slice_idx]
                                if len(color_frames) != 2:
                                    print(
                                        f"✗ Grid {grid_idx+1}, "
                                        f"Row slice {row_slice_idx+1}, Col slice {col_slice_idx+1} "
                                        f"has {len(color_frames)} color frames (expected 2)"
                                    )
                                    continue

                                # Combine 2 color frames into values [0-15]
                                lower_3bits = color_frames[0].astype(np.uint8)  # Values 0-7
                                upper_bit = color_frames[1].astype(np.uint8)    # Values 0-1

                                slice_values = lower_3bits | (upper_bit << 3)  # grid_size×grid_size
                                row_blocks.append(slice_values)

                            if row_blocks:
                                row_band = np.hstack(row_blocks)
                                reconstructed_rows.append(row_band)

                        if not reconstructed_rows:
                            print(f"✗ Grid {grid_idx+1}: no complete rows reconstructed")
                            continue

                        # Stack rows to reconstruct full grid
                        full_grid = np.vstack(reconstructed_rows)

                        # Trim to actual tensor dimensions
                        full_grid = full_grid[:rows, :cols]

                        decoded_grids.append(full_grid)
                        print(f"Grid {grid_idx+1}: reconstructed {full_grid.shape}, range [{full_grid.min()}, {full_grid.max()}]")

                    if len(decoded_grids) != num_grids:
                        print(f"\n✗ Reconstructed {len(decoded_grids)}/{num_grids} grids - waiting for remaining frames")
                        continue

                    # Stack into (4, rows, cols) array
                    grids_array = np.stack(decoded_grids, axis=0)
                    print(f"\nStacked grids: {grids_array.shape}")

                    # Decode with codec
                    c = codec(rows, cols, min_val=0, max_val=15)
                    decoded_tensor = c.decode(grids_array)
                    print(f"Decoded tensor: {decoded_tensor.shape}, dtype={decoded_tensor.dtype}")

                    # Verify
                    original_u16 = tensor.view(np.uint16)
                    decoded_u16 = decoded_tensor.view(np.uint16)
                    match = np.array_equal(original_u16, decoded_u16)

                    print("\n" + "="*70)
                    if match:
                        print("✓✓✓ SUCCESS! Tensor transmitted losslessly!")
                        print("="*70)
                        print(f"Original sample: {tensor[0, :5]}")
                        print(f"Decoded sample:  {decoded_tensor[0, :5]}")
                        print(f"Match: {match}")
                    else:
                        print("✗✗✗ MISMATCH - Transmission had errors")
                        print("="*70)
                        diff = np.sum(original_u16 != decoded_u16)
                        print(f"Mismatched values: {diff} / {original_u16.size}")
                        print(f"Original sample: {tensor[0, :5]}")
                        print(f"Decoded sample:  {decoded_tensor[0, :5]}")

                except Exception as e:
                    print(f"\n✗ Decoding failed: {e}")
                    import traceback
                    traceback.print_exc()

        # Draw locked region
        if rx.calibrated and rx.locked_corners is not None:
            corners = rx.locked_corners.astype(np.int32)
            cv2.polylines(display, [corners], True, (0, 255, 0), 3)

            # Show status
            if cap.done:
                cv2.putText(display, "✓ DONE - Transmission complete",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif cap.transmitting:
                grid_num = cap.current_grid_idx + 1
                row_slice_num = cap.current_row_slice + 1
                col_slice_num = cap.current_col_slice + 1
                color_frame_num = cap.current_color_frame
                cv2.putText(
                    display,
                    (f"G{grid_num}/{num_grids} "
                     f"R{row_slice_num}/{num_row_slices} "
                     f"C{col_slice_num}/{num_col_slices} "
                     f"F{color_frame_num}/2 - AUTO TX"),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Show received color frames count
                total_received = sum(
                    sum(
                        sum(len(frames) for frames in col_slices)
                        for col_slices in row_slices
                    )
                    for row_slices in received_color_frames
                )
                total_needed = num_grids * slices_per_grid * 2
                cv2.putText(display, f"Received: {total_received}/{total_needed} frames",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "LOCKED - Press 'p' to start",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(display, "NOT CALIBRATED - Press 'c'",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Codec Transmission Test', display)

        # Show debug windows
        if rx.debug_gray is not None:
            cv2.imshow('Debug: Grayscale', rx.debug_gray)
        if rx.debug_warped is not None:
            cv2.imshow('Debug: Binary + Grid', rx.debug_warped)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            # Calibrate
            if rx.calibrated:
                rx.calibrated = False
                rx.locked_corners = None
                print("\n✗ Calibration reset")
            else:
                if rx.calibrate(frame):
                    print("✓ Calibration locked!")
                else:
                    print("✗ Calibration failed. Make sure grid is visible.")

        elif key == ord('p'):
            # Start transmission
            if not rx.calibrated:
                print("\n✗ Must calibrate first (press 'c')")
            else:
                cap.start_transmission()

        elif key == ord('f'):
            rx.flip_horizontal = not rx.flip_horizontal
            print(f"\n✓ Flip horizontal: {'ON' if rx.flip_horizontal else 'OFF'}")

        elif key == ord('+') or key == ord('='):
            rx.border_ratio += 0.01
            print(f"\n✓ Border: {rx.border_ratio:.3f}")

        elif key == ord('-') or key == ord('_'):
            rx.border_ratio = max(0.0, rx.border_ratio - 0.01)
            print(f"\n✓ Border: {rx.border_ratio:.3f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
