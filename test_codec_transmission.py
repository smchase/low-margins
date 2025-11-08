#!/usr/bin/env python3
"""
Test full tensor transmission: encode with codec -> transmit via virtual camera -> receive and decode.

Strategy: Send each of the 4 encoded grids as a separate visual frame.
Each grid (748x64) flattened is 47,872 values in range [0,15].
We'll use a 220x220 visual grid to fit 48,400 cells (slightly more than needed).
"""

import numpy as np
import threading
import time
import cv2
import sys
from codec import codec
from virtual_camera import VirtualCameraCapture


class LargeGridTransmitter:
    """Transmitter that can handle arbitrary grid sizes"""

    def __init__(self, grid_size=220, cell_size=2):
        """
        Args:
            grid_size: Number of cells per side (e.g., 220 for 220x220 grid)
            cell_size: Pixels per cell (smaller = faster but harder to detect)
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pattern = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.calibration_mode = False
        self.flash_counter = 0

    def set_pattern(self, pattern):
        """Set the grid pattern directly (values 0 or 1)"""
        if pattern.shape != (self.grid_size, self.grid_size):
            raise ValueError(f"Pattern must be {self.grid_size}x{self.grid_size}")
        self.pattern = pattern.astype(np.uint8)

    def render(self):
        """Render the grid"""
        total_size = self.grid_size * self.cell_size

        if self.calibration_mode:
            # Flash pattern for calibration
            self.flash_counter += 1
            flash_state = (self.flash_counter // 10) % 3

            if flash_state == 0:
                grid = np.ones((total_size, total_size), dtype=np.uint8) * 255
            elif flash_state == 1:
                grid = np.zeros((total_size, total_size), dtype=np.uint8)
            else:
                # Checkerboard
                grid = np.zeros((total_size, total_size), dtype=np.uint8)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        color = 255 if (i + j) % 2 else 0
                        y1 = i * self.cell_size
                        y2 = (i + 1) * self.cell_size
                        x1 = j * self.cell_size
                        x2 = (j + 1) * self.cell_size
                        grid[y1:y2, x1:x2] = color
        else:
            # Normal mode - show the actual pattern
            grid = np.zeros((total_size, total_size), dtype=np.uint8)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    color = 255 if self.pattern[i, j] else 0
                    y1 = i * self.cell_size
                    y2 = (i + 1) * self.cell_size
                    x1 = j * self.cell_size
                    x2 = (j + 1) * self.cell_size
                    grid[y1:y2, x1:x2] = color

        # Add white border (smaller border for large grids)
        border = max(20, total_size // 20)
        final_size = total_size + 2 * border
        image = np.ones((final_size, final_size), dtype=np.uint8) * 255
        image[border:-border, border:-border] = grid

        # Add black frame
        frame_width = 2
        image[border:border+frame_width, border:-border] = 0
        image[-border-frame_width:-border, border:-border] = 0
        image[border:-border, border:border+frame_width] = 0
        image[border:-border, -border-frame_width:-border] = 0

        return image


def encode_grid_to_pattern(grid_values, grid_size, min_val, max_val):
    """
    Convert a flattened array of values in [min_val, max_val] to a 2D binary pattern.

    For values in [0, 15], we'll use 4 bits per value.
    Grid size 220x220 = 48,400 cells
    Each value needs 4 bits, so we can fit 48,400/4 = 12,100 values
    But we need 47,872 values, so this works!
    """
    flat = grid_values.flatten()
    n_values = len(flat)
    bits_per_value = 4  # For range [0, 15]

    # Total bits needed
    total_bits = n_values * bits_per_value
    total_cells = grid_size * grid_size

    if total_bits > total_cells:
        raise ValueError(f"Need {total_bits} bits but only have {total_cells} cells")

    # Convert values to binary
    pattern_bits = np.zeros(total_cells, dtype=np.uint8)
    for i, val in enumerate(flat):
        # Convert value to 4 bits
        for bit_idx in range(bits_per_value):
            pattern_bits[i * bits_per_value + bit_idx] = (val >> bit_idx) & 1

    return pattern_bits.reshape(grid_size, grid_size)


def decode_pattern_to_grid(pattern, n_values, min_val, max_val):
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

    return values


def run_transmitter(vcam, tensor, grid_size=512, cell_size=2):
    """Transmit tensor through virtual camera"""
    print("[TX] Starting transmitter...")

    # Encode tensor
    c = codec(tensor.shape[0], tensor.shape[1], min_val=0, max_val=15)
    grids = c.encode(tensor)
    print(f"[TX] Encoded into {grids.shape[0]} grids of {grids.shape[1]}x{grids.shape[2]}")

    tx = LargeGridTransmitter(grid_size=grid_size, cell_size=cell_size)

    # Transmit each grid
    for grid_idx in range(grids.shape[0]):
        print(f"\n[TX] === Transmitting grid {grid_idx+1}/{grids.shape[0]} ===")

        # Convert grid to binary pattern
        pattern = encode_grid_to_pattern(grids[grid_idx], grid_size, 0, 15)
        tx.set_pattern(pattern)

        # Calibration phase (flashing)
        tx.calibration_mode = True
        print(f"[TX] Grid {grid_idx+1}: calibration (flashing)...")
        for _ in range(60):  # 2 seconds
            img = tx.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vcam.write_frame(img_bgr)
            time.sleep(0.03)

        # Data transmission phase
        tx.calibration_mode = False
        print(f"[TX] Grid {grid_idx+1}: transmitting data...")
        for _ in range(90):  # 3 seconds
            img = tx.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vcam.write_frame(img_bgr)
            time.sleep(0.03)

        print(f"[TX] Grid {grid_idx+1}: complete")

    print("\n[TX] All grids transmitted successfully!")


def main():
    print("\n" + "="*70)
    print("CODEC + VISUAL DATA LINK TEST")
    print("Transmitting 748x64 FP16 tensor through virtual camera")
    print("="*70)

    # Create test tensor
    tensor = np.random.randn(748, 64).astype(np.float16)
    print(f"Test tensor: shape={tensor.shape}, dtype={tensor.dtype}")

    # Create virtual camera
    vcam = VirtualCameraCapture(buffer_size=10)

    # Start transmitter thread
    tx_thread = threading.Thread(target=run_transmitter, args=(vcam, tensor, 512, 2))
    tx_thread.daemon = True
    tx_thread.start()

    print("\n[RX] Receiver would go here...")
    print("[RX] Press Ctrl+C to stop")

    # Let it run for a while
    try:
        while tx_thread.is_alive():
            # Read from virtual camera (receiver would go here)
            ret, frame = vcam.read()
            if ret:
                cv2.imshow("Virtual Camera Feed", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n\nStopped by user")

    vcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
