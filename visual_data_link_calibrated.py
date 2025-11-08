#!/usr/bin/env python3
"""
Visual Data Link - Calibration Based
Uses a calibration phase to lock onto exact position, then reads from fixed coordinates
"""

import cv2
import numpy as np
import sys
import time
import string
import random
import json
from collections import deque


class GridTransmitter:
    """Grid transmitter with calibration mode"""

    def __init__(self, grid_size=24, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pattern = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.window_name = "TX Grid"
        self.calibration_mode = False
        self.calibration_frame = 0

    def set_calibration_mode(self, enabled):
        """Enable/disable calibration mode"""
        self.calibration_mode = enabled
        self.calibration_frame = 0

    def set_pattern_from_bytes(self, data):
        """Set pattern from bytes"""
        pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                if bit_index < len(data) * 8:
                    byte_index = bit_index // 8
                    bit_offset = bit_index % 8
                    pattern[i, j] = (data[byte_index] >> bit_offset) & 1
        self.pattern = pattern

    def get_pattern_as_bytes(self):
        """Convert pattern to bytes"""
        total_bits = self.grid_size * self.grid_size
        num_bytes = (total_bits + 7) // 8
        data = bytearray(num_bytes)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                byte_index = bit_index // 8
                bit_offset = bit_index % 8
                if self.pattern[i, j]:
                    data[byte_index] |= (1 << bit_offset)
        return bytes(data)

    def render(self):
        """Render the grid"""
        total_size = self.grid_size * self.cell_size

        # If in calibration mode, flash between distinctive patterns
        if self.calibration_mode:
            self.calibration_frame += 1
            # Flash every 10 frames
            flash_state = (self.calibration_frame // 10) % 3

            if flash_state == 0:
                # All white
                grid = np.ones((total_size, total_size), dtype=np.uint8) * 255
            elif flash_state == 1:
                # All black
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
            # Normal mode - display the actual pattern
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
        border = 50
        final_size = total_size + 2 * border
        image = np.ones((final_size, final_size), dtype=np.uint8) * 255
        image[border:-border, border:-border] = grid

        # Add black frame inside white border for clear detection
        frame_width = 5
        image[border:border+frame_width, border:-border] = 0
        image[-border-frame_width:-border, border:-border] = 0
        image[border:-border, border:border+frame_width] = 0
        image[border:-border, -border-frame_width:-border] = 0

        # If calibration mode, add text
        if self.calibration_mode:
            cv2.putText(image, "CALIBRATION MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
            cv2.putText(image, "Press 'c' in RX to lock", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)

        return image

    def display(self):
        """Display the grid"""
        image = self.render()
        cv2.imshow(self.window_name, image)


class GridReceiver:
    """Grid receiver with calibration and position locking"""

    def __init__(self, grid_size=24, debug=False):
        self.grid_size = grid_size
        self.debug = debug
        self.debug_images = {}

        # Calibration state
        self.calibrated = False
        self.locked_corners = None  # Store the exact corner coordinates
        self.warp_matrix = None
        self.warp_size = 800

        # Statistics
        self.successful_reads = 0
        self.failed_reads = 0
        self.last_read_time = time.time()
        self.read_times = deque(maxlen=30)

    def save_calibration(self, filename="calibration.json"):
        """Save calibration to file"""
        if self.locked_corners is not None:
            data = {
                'corners': self.locked_corners.tolist(),
                'grid_size': self.grid_size,
                'warp_size': self.warp_size
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
            print(f"Calibration saved to {filename}")
            return True
        return False

    def load_calibration(self, filename="calibration.json"):
        """Load calibration from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if data['grid_size'] != self.grid_size:
                print(f"Warning: Calibration grid size {data['grid_size']} doesn't match current {self.grid_size}")
                return False

            self.locked_corners = np.array(data['corners'], dtype=np.float32)
            self.warp_size = data['warp_size']
            self._compute_warp_matrix()
            self.calibrated = True
            print(f"Calibration loaded from {filename}")
            return True
        except Exception as e:
            print(f"Could not load calibration: {e}")
            return False

    def calibrate(self, frame):
        """Try to detect and lock onto the grid position"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )

        self.debug_images['calibration_binary'] = binary.copy()

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
                    candidates.append({
                        'contour': approx,
                        'area': area
                    })

        if not candidates:
            return False

        # Use largest candidate
        candidates.sort(key=lambda x: x['area'], reverse=True)
        best = candidates[0]

        # Store the corner positions
        pts = best['contour'].reshape(4, 2).astype(np.float32)
        self.locked_corners = self._order_points(pts)
        self._compute_warp_matrix()

        self.calibrated = True
        print(f"\n✓ CALIBRATED! Locked onto position:")
        print(f"  Corners: {self.locked_corners.astype(int).tolist()}")

        return True

    def _compute_warp_matrix(self):
        """Compute the perspective transform matrix from locked corners"""
        dst = np.array([
            [0, 0],
            [self.warp_size - 1, 0],
            [self.warp_size - 1, self.warp_size - 1],
            [0, self.warp_size - 1]
        ], dtype=np.float32)

        self.warp_matrix = cv2.getPerspectiveTransform(self.locked_corners, dst)

    def read_from_locked_position(self, frame):
        """Read grid from the locked position (fast, no detection needed)"""
        if not self.calibrated or self.warp_matrix is None:
            return None

        try:
            # Apply the stored perspective transform
            warped = cv2.warpPerspective(frame, self.warp_matrix, (self.warp_size, self.warp_size))

            # Convert to grayscale and threshold
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 5
            )

            self.debug_images['locked_warped'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            # Read the grid (accounting for border)
            border_ratio = 0.12
            grid_start = int(self.warp_size * border_ratio)
            grid_end = int(self.warp_size * (1 - border_ratio))
            grid_size_px = grid_end - grid_start

            pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            cell_size = grid_size_px / self.grid_size

            debug_grid = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

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

                    # Debug visualization
                    color = (0, 255, 0) if pattern[i, j] == 1 else (255, 0, 0)
                    cv2.rectangle(debug_grid, (x1, y1), (x2, y2), color, 1)

            self.debug_images['locked_grid'] = debug_grid

            # Update stats
            self.successful_reads += 1
            current_time = time.time()
            self.read_times.append(current_time - self.last_read_time)
            self.last_read_time = current_time

            return pattern

        except Exception as e:
            if self.debug:
                print(f"Error reading locked position: {e}")
            self.failed_reads += 1
            return None

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

    def pattern_to_bytes(self, pattern):
        """Convert pattern to bytes"""
        if pattern is None:
            return None

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

        return bytes(data)

    def get_stats(self):
        """Get statistics"""
        total = self.successful_reads + self.failed_reads
        success_rate = (self.successful_reads / total * 100) if total > 0 else 0

        if len(self.read_times) > 1:
            avg_interval = np.mean(list(self.read_times))
            frames_per_sec = 1.0 / avg_interval if avg_interval > 0 else 0
            bytes_per_sec = (frames_per_sec * self.grid_size * self.grid_size) / 8
        else:
            frames_per_sec = 0
            bytes_per_sec = 0

        return {
            'success_rate': success_rate,
            'successful_reads': self.successful_reads,
            'failed_reads': self.failed_reads,
            'frames_per_sec': frames_per_sec,
            'bytes_per_sec': bytes_per_sec
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visual Data Link - Calibration Based')
    parser.add_argument('--mode', type=str, required=True, choices=['tx', 'rx'],
                       help='Run as transmitter (tx) or receiver (rx)')
    parser.add_argument('--grid-size', type=int, default=24)
    parser.add_argument('--cell-size', type=int, default=20)
    parser.add_argument('--message', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load-calibration', action='store_true',
                       help='Load calibration from file (RX only)')

    args = parser.parse_args()

    is_tx = args.mode == 'tx'
    is_rx = args.mode == 'rx'

    print("\n" + "="*70)
    print(f"Visual Data Link - {'TRANSMITTER' if is_tx else 'RECEIVER'}")
    print("="*70)
    print(f"\nGrid: {args.grid_size}x{args.grid_size} ({args.grid_size**2 // 8} bytes)")

    if is_tx:
        print("\nTransmitter Controls:")
        print("  c       - Toggle CALIBRATION MODE (flashing)")
        print("  SPACE/r - New random message")
        print("  q       - Quit")
        print("\nCalibration:")
        print("  1. Press 'c' to start flashing")
        print("  2. Wait for RX to lock on")
        print("  3. Press 'c' again to stop flashing and transmit data")
    else:
        print("\nReceiver Controls:")
        print("  c       - LOCK onto calibration pattern")
        print("  d       - Toggle debug windows")
        print("  s       - Save calibration to file")
        print("  l       - Load calibration from file")
        print("  q       - Quit")
        print("\nCalibration:")
        print("  1. Make sure TX is flashing (TX pressed 'c')")
        print("  2. Point camera at TX screen")
        print("  3. Press 'c' to lock on")

    print("="*70 + "\n")

    # Initialize camera (only for RX)
    if is_rx:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = None

    # Initialize transmitter and/or receiver based on mode
    if is_tx:
        transmitter = GridTransmitter(grid_size=args.grid_size, cell_size=args.cell_size)
        receiver = None
    else:
        transmitter = None
        receiver = GridReceiver(grid_size=args.grid_size, debug=args.debug)
        # Try to load calibration
        if args.load_calibration:
            receiver.load_calibration()

    # Generate message (TX only)
    max_bytes = args.grid_size**2 // 8
    if is_tx:
        if args.message:
            message = args.message[:max_bytes]
            message_bytes = message.encode('utf-8')
            if len(message_bytes) < max_bytes:
                message_bytes = message_bytes + b'\x00' * (max_bytes - len(message_bytes))
        else:
            chars = string.ascii_letters + string.digits + ' !?.'
            message = ''.join(random.choice(chars) for _ in range(max_bytes))
            message_bytes = message.encode('utf-8')

        print(f"Message: '{message_bytes.decode('utf-8', errors='replace').rstrip(chr(0))}'")
        print()

        transmitter.set_pattern_from_bytes(message_bytes)
        transmitted_message = message_bytes
    else:
        transmitted_message = None

    last_stats_time = time.time()

    while True:
        # TX MODE - Just display the transmitter
        if is_tx:
            transmitter.display()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                # Toggle calibration mode
                transmitter.set_calibration_mode(not transmitter.calibration_mode)
                mode_text = "ON (FLASHING)" if transmitter.calibration_mode else "OFF (TRANSMITTING)"
                print(f"\nCalibration mode: {mode_text}")
            elif key == ord(' ') or key == ord('r'):
                # Generate new message
                chars = string.ascii_letters + string.digits + ' !?.'
                message = ''.join(random.choice(chars) for _ in range(max_bytes))
                transmitted_message = message.encode('utf-8')
                transmitter.set_pattern_from_bytes(transmitted_message)
                print(f"\nNew message: '{message}'")

        # RX MODE - Capture and read
        else:
            ret, frame = cap.read()
            if not ret:
                break

            # Read grid if calibrated
            if receiver.calibrated:
                detected_pattern = receiver.read_from_locked_position(frame)
            else:
                detected_pattern = None

            # Visualize
            display_frame = frame.copy()

            # Draw locked region if calibrated
            if receiver.calibrated and receiver.locked_corners is not None:
                corners = receiver.locked_corners.astype(np.int32)
                cv2.polylines(display_frame, [corners], True, (0, 255, 0), 3)
                for corner in corners:
                    cv2.circle(display_frame, tuple(corner), 8, (0, 255, 255), -1)

            # Status
            info_y = 30
            if receiver.calibrated:
                cv2.putText(display_frame, "LOCKED", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "NOT CALIBRATED - Press 'c'", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Stats
            stats = receiver.get_stats()
            info_y += 40
            cv2.putText(display_frame,
                       f"Success: {stats['success_rate']:.1f}%",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if stats['bytes_per_sec'] > 0:
                info_y += 35
                cv2.putText(display_frame,
                           f"Rate: {stats['frames_per_sec']:.1f} fps, {stats['bytes_per_sec']:.0f} B/s",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show received message
            if detected_pattern is not None:
                rx_bytes = receiver.pattern_to_bytes(detected_pattern)
                rx_message = rx_bytes.decode('utf-8', errors='replace').rstrip('\x00')

                cv2.putText(display_frame, "RECEIVING", (10, display_frame.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                max_len = 50
                rx_disp = rx_message[:max_len] + ('...' if len(rx_message) > max_len else '')
                cv2.putText(display_frame, f"RX: {rx_disp}", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Print to terminal
                print(f"\r✓ Received: '{rx_message[:70]}'", end='', flush=True)

            cv2.imshow('Camera Feed', display_frame)

            # Debug windows
            if receiver.debug:
                for name, img in receiver.debug_images.items():
                    cv2.imshow(f'Debug: {name}', img)

            # Stats
            if time.time() - last_stats_time > 5.0 and stats['bytes_per_sec'] > 0:
                print(f"\nStats: {stats['frames_per_sec']:.1f} fps | {stats['bytes_per_sec']:.0f} B/s")
                last_stats_time = time.time()

            # Input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                if receiver.calibrated:
                    # Reset calibration
                    receiver.calibrated = False
                    receiver.locked_corners = None
                    print("\n✗ Calibration reset. Press 'c' to recalibrate.")
                else:
                    # Try to calibrate
                    if receiver.calibrate(frame):
                        print("✓ Success! TX can now stop flashing (press 'c' on TX)")
                    else:
                        print("\n✗ Calibration failed. Make sure TX is flashing (press 'c' on TX)")
            elif key == ord('d'):
                receiver.debug = not receiver.debug
                if not receiver.debug:
                    for name in list(receiver.debug_images.keys()):
                        cv2.destroyWindow(f'Debug: {name}')
                print(f"\nDebug: {'ON' if receiver.debug else 'OFF'}")
            elif key == ord('s'):
                receiver.save_calibration()
            elif key == ord('l'):
                receiver.load_calibration()

    # Cleanup
    print("\n\n")
    if is_rx:
        stats = receiver.get_stats()
        print(f"Final stats:")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Total reads: {stats['successful_reads'] + stats['failed_reads']}")
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
