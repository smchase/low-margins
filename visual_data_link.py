#!/usr/bin/env python3
"""
Visual Data Link - Simple version
Fixed message transmission between two MacBooks facing each other
"""

import cv2
import numpy as np
import sys
import time
import argparse


# FIXED MESSAGE TO TRANSMIT
FIXED_MESSAGE = "HELLO WORLD"


class Transmitter:
    """Displays a fixed message as a grid pattern"""

    def __init__(self, grid_size=16, cell_size=30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pattern = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.calibration_mode = False
        self.flash_counter = 0

    def set_message(self, message):
        """Encode message into grid pattern"""
        # Convert message to bytes
        message_bytes = message.encode('utf-8')
        max_bytes = (self.grid_size * self.grid_size) // 8

        # Pad with zeros
        if len(message_bytes) < max_bytes:
            message_bytes = message_bytes + b'\x00' * (max_bytes - len(message_bytes))
        else:
            message_bytes = message_bytes[:max_bytes]

        # Convert to bit pattern
        pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                if bit_index < len(message_bytes) * 8:
                    byte_index = bit_index // 8
                    bit_offset = bit_index % 8
                    pattern[i, j] = (message_bytes[byte_index] >> bit_offset) & 1

        self.pattern = pattern
        print(f"Encoded message: '{message}'")
        print(f"Hex (first 8 bytes): {message_bytes[:8].hex()}")

    def render(self):
        """Render the grid"""
        total_size = self.grid_size * self.cell_size

        # If calibrating, flash
        if self.calibration_mode:
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

        # Add white border
        border = 60
        final_size = total_size + 2 * border
        image = np.ones((final_size, final_size), dtype=np.uint8) * 255
        image[border:-border, border:-border] = grid

        # Add black frame
        frame_width = 5
        image[border:border+frame_width, border:-border] = 0
        image[-border-frame_width:-border, border:-border] = 0
        image[border:-border, border:border+frame_width] = 0
        image[border:-border, -border-frame_width:-border] = 0

        # Add status text
        if self.calibration_mode:
            cv2.putText(image, "CALIBRATION MODE - FLASHING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 128, 2)
        else:
            cv2.putText(image, f"TX: {FIXED_MESSAGE}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)

        return image


class Receiver:
    """Receives and decodes grid patterns"""

    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.calibrated = False
        self.locked_corners = None
        self.warp_matrix = None
        self.warp_size = 600

    def calibrate(self, frame):
        """Detect and lock onto the grid"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find quadrilaterals
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    candidates.append({'contour': approx, 'area': area})

        if not candidates:
            return False

        # Use largest
        candidates.sort(key=lambda x: x['area'], reverse=True)
        best = candidates[0]

        # Store corners
        pts = best['contour'].reshape(4, 2).astype(np.float32)
        self.locked_corners = self._order_points(pts)

        # Compute warp matrix
        dst = np.array([
            [0, 0],
            [self.warp_size - 1, 0],
            [self.warp_size - 1, self.warp_size - 1],
            [0, self.warp_size - 1]
        ], dtype=np.float32)
        self.warp_matrix = cv2.getPerspectiveTransform(self.locked_corners, dst)

        self.calibrated = True
        print(f"\n✓ CALIBRATED!")
        return True

    def read(self, frame):
        """Read grid from locked position"""
        if not self.calibrated:
            return None

        # Apply perspective transform
        warped = cv2.warpPerspective(frame, self.warp_matrix, (self.warp_size, self.warp_size))

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )

        # Read grid (accounting for border)
        border_ratio = 0.13
        grid_start = int(self.warp_size * border_ratio)
        grid_end = int(self.warp_size * (1 - border_ratio))
        grid_size_px = grid_end - grid_start

        pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        cell_size = grid_size_px / self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y = int(grid_start + i * cell_size + cell_size / 2)
                x = int(grid_start + j * cell_size + cell_size / 2)

                sample_size = max(3, int(cell_size * 0.4))
                y1 = max(0, y - sample_size)
                y2 = min(self.warp_size, y + sample_size)
                x1 = max(0, x - sample_size)
                x2 = min(self.warp_size, x + sample_size)

                region = binary[y1:y2, x1:x2]
                white_ratio = np.sum(region == 255) / region.size
                pattern[i, j] = 1 if white_ratio > 0.5 else 0

        return pattern

    def decode(self, pattern):
        """Decode pattern to message - with horizontal flip for facing cameras"""
        if pattern is None:
            return None

        # FLIP HORIZONTALLY - cameras are facing each other!
        pattern = np.fliplr(pattern)

        # Convert to bytes
        total_bits = self.grid_size * self.grid_size
        num_bytes = (total_bits + 7) // 8
        data = bytearray(num_bytes)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                byte_index = bit_index // 8
                bit_offset = bit_index % 8
                if pattern[i, j]:
                    data[byte_index] |= (1 << bit_offset)

        message_bytes = bytes(data)

        # Decode as UTF-8
        try:
            message = message_bytes.decode('utf-8', errors='strict').rstrip('\x00')
            return message, message_bytes
        except UnicodeDecodeError:
            return None, message_bytes

    def _order_points(self, pts):
        """Order points: TL, TR, BR, BL"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


def main():
    parser = argparse.ArgumentParser(description='Visual Data Link')
    parser.add_argument('--mode', required=True, choices=['tx', 'rx'],
                       help='Transmitter or receiver')
    parser.add_argument('--grid-size', type=int, default=16,
                       help='Grid size (default: 16)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"Visual Data Link - {'TRANSMITTER' if args.mode == 'tx' else 'RECEIVER'}")
    print("="*70)
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Fixed message: '{FIXED_MESSAGE}'")
    print()

    if args.mode == 'tx':
        print("Transmitter Controls:")
        print("  c - Toggle calibration mode (flashing)")
        print("  q - Quit")
        print()

        tx = Transmitter(grid_size=args.grid_size)
        tx.set_message(FIXED_MESSAGE)

        print("Ready. Press 'c' to enter calibration mode.\n")

        while True:
            img = tx.render()
            cv2.imshow('TX', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                tx.calibration_mode = not tx.calibration_mode
                print(f"Calibration: {'ON (flashing)' if tx.calibration_mode else 'OFF (transmitting)'}")

    else:  # rx mode
        print("Receiver Controls:")
        print("  c - Calibrate (lock onto grid)")
        print("  q - Quit")
        print()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        rx = Receiver(grid_size=args.grid_size)

        print("Ready. Point camera at TX, make sure TX is flashing, then press 'c'.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            # Draw locked region
            if rx.calibrated and rx.locked_corners is not None:
                corners = rx.locked_corners.astype(np.int32)
                cv2.polylines(display, [corners], True, (0, 255, 0), 3)

                # Read and decode
                pattern = rx.read(frame)
                result = rx.decode(pattern)

                if result:
                    message, message_bytes = result
                    if message:
                        # SUCCESS!
                        cv2.putText(display, "DECODED:", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(display, f"'{message}'", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                        # Show if it matches
                        if message == FIXED_MESSAGE:
                            cv2.putText(display, "MATCH!", (10, 110),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.putText(display, "MISMATCH", (10, 110),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # Show hex
                        hex_str = message_bytes[:8].hex()
                        cv2.putText(display, f"Hex: {hex_str}", (10, display.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        print(f"\r✓ Decoded: '{message}' | Hex: {hex_str}", end='', flush=True)
                    else:
                        cv2.putText(display, "INVALID UTF-8", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        hex_str = message_bytes[:8].hex()
                        cv2.putText(display, f"Hex: {hex_str}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        print(f"\r✗ Invalid UTF-8 | Hex: {hex_str}", end='', flush=True)

                status = "LOCKED"
                color = (0, 255, 0)
            else:
                status = "NOT CALIBRATED - Press 'c'"
                color = (0, 0, 255)

            cv2.putText(display, status, (10, display.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('RX', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if rx.calibrated:
                    rx.calibrated = False
                    rx.locked_corners = None
                    print("\n✗ Calibration reset")
                else:
                    if rx.calibrate(frame):
                        print("✓ Locked! TX can now stop flashing.")
                    else:
                        print("✗ Failed. Make sure TX is flashing.")

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
