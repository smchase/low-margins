#!/usr/bin/env python3
"""
Visual Data Link V2 - Simplified, robust detection
Uses white border detection instead of color markers
"""

import cv2
import numpy as np
import sys
import time
import string
import random
from collections import deque


class GridTransmitter:
    """Displays grid patterns with a distinctive white border for easy detection"""

    def __init__(self, grid_size=24, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pattern = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.window_name = "TX Grid"

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
        """Convert current pattern to bytes"""
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
        """Render with thick white border and black inner border for clear detection"""
        total_size = self.grid_size * self.cell_size

        # Create the grid
        grid = np.zeros((total_size, total_size), dtype=np.uint8)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = 255 if self.pattern[i, j] else 0
                y1 = i * self.cell_size
                y2 = (i + 1) * self.cell_size
                x1 = j * self.cell_size
                x2 = (j + 1) * self.cell_size
                grid[y1:y2, x1:x2] = color

        # Add thick white outer border
        white_border = 60
        # Add thin black inner border
        black_border = 10
        total_border = white_border + black_border

        final_size = total_size + 2 * total_border
        image = np.zeros((final_size, final_size), dtype=np.uint8)

        # Fill with white
        image[:] = 255

        # Add black border inside white
        image[white_border:white_border+black_border, :] = 0
        image[-white_border-black_border:-white_border, :] = 0
        image[:, white_border:white_border+black_border] = 0
        image[:, -white_border-black_border:-white_border] = 0

        # Place grid in center
        image[total_border:-total_border, total_border:-total_border] = grid

        # Add orientation markers in corners (small black squares in white border)
        marker_size = 20
        margin = 15
        # Top-left: extra large for orientation
        image[margin:margin+marker_size+10, margin:margin+marker_size+10] = 0
        # Top-right
        image[margin:margin+marker_size, -margin-marker_size:-margin] = 0
        # Bottom-right
        image[-margin-marker_size:-margin, -margin-marker_size:-margin] = 0
        # Bottom-left
        image[-margin-marker_size:-margin, margin:margin+marker_size] = 0

        return image

    def display(self):
        """Display the grid"""
        image = self.render()
        cv2.imshow(self.window_name, image)


class GridReceiver:
    """Simplified robust grid detector"""

    def __init__(self, grid_size=24, debug=False):
        self.grid_size = grid_size
        self.debug = debug
        self.debug_images = {}
        self.successful_reads = 0
        self.failed_reads = 0
        self.last_read_time = time.time()
        self.read_times = deque(maxlen=30)

    def detect_and_read(self, frame):
        """Detect grid by finding the white border"""
        self.debug_images = {}

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.debug_images['1_gray'] = gray.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.debug_images['2_blurred'] = blurred.copy()

        # Adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )
        self.debug_images['3_binary'] = binary.copy()

        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw all contours for debugging
        debug_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
        self.debug_images['4_contours'] = debug_contours

        # Find potential grid candidates
        candidates = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Must be large enough
            if area < 10000:
                continue

            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Must be a quadrilateral
            if len(approx) != 4:
                continue

            # Check aspect ratio (should be roughly square)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0

            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            # Check if it's convex
            if not cv2.isContourConvex(approx):
                continue

            # This is a good candidate
            candidates.append({
                'contour': approx,
                'area': area,
                'aspect_ratio': aspect_ratio
            })

        # Sort by area (largest first)
        candidates.sort(key=lambda x: x['area'], reverse=True)

        # Draw candidates
        debug_candidates = frame.copy()
        for i, candidate in enumerate(candidates[:5]):
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.drawContours(debug_candidates, [candidate['contour']], -1, color, 3)
            # Label it
            M = cv2.moments(candidate['contour'])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(debug_candidates, f"#{i+1}: {int(candidate['area'])}",
                           (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        self.debug_images['5_candidates'] = debug_candidates

        # Try to read from best candidates
        for candidate in candidates[:3]:  # Try top 3 candidates
            pattern = self._read_grid_from_contour(frame, candidate['contour'])
            if pattern is not None:
                self.successful_reads += 1
                current_time = time.time()
                self.read_times.append(current_time - self.last_read_time)
                self.last_read_time = current_time
                return pattern, candidate['contour']

        # No successful read
        self.failed_reads += 1
        return None, None

    def _read_grid_from_contour(self, frame, contour):
        """Read grid pattern from detected quadrilateral"""
        try:
            # Get the four corners
            pts = contour.reshape(4, 2).astype(np.float32)

            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = self._order_points(pts)

            # Determine warp size
            warp_size = max(800, self.grid_size * 20)

            dst = np.array([
                [0, 0],
                [warp_size - 1, 0],
                [warp_size - 1, warp_size - 1],
                [0, warp_size - 1]
            ], dtype=np.float32)

            # Get perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (warp_size, warp_size))

            # Convert to grayscale
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # Apply adaptive threshold
            binary_warped = cv2.adaptiveThreshold(
                gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 5
            )

            self.debug_images['6_warped'] = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

            # The actual data grid is in the center, accounting for borders
            # Border is roughly 15% on each side
            border_ratio = 0.15
            grid_start = int(warp_size * border_ratio)
            grid_end = int(warp_size * (1 - border_ratio))
            grid_size_px = grid_end - grid_start

            # Read the grid
            pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            cell_size = grid_size_px / self.grid_size

            # Create debug visualization
            debug_grid = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Sample center of cell
                    y = int(grid_start + i * cell_size + cell_size / 2)
                    x = int(grid_start + j * cell_size + cell_size / 2)

                    # Sample region
                    sample_size = max(3, int(cell_size * 0.4))
                    y1 = max(0, y - sample_size)
                    y2 = min(warp_size, y + sample_size)
                    x1 = max(0, x - sample_size)
                    x2 = min(warp_size, x + sample_size)

                    region = binary_warped[y1:y2, x1:x2]

                    # Count white pixels
                    white_ratio = np.sum(region == 255) / region.size
                    pattern[i, j] = 1 if white_ratio > 0.5 else 0

                    # Draw on debug image
                    color = (0, 255, 0) if pattern[i, j] == 1 else (255, 0, 0)
                    cv2.rectangle(debug_grid, (x1, y1), (x2, y2), color, 1)

            self.debug_images['7_grid_read'] = debug_grid

            return pattern

        except Exception as e:
            if self.debug:
                print(f"Error reading grid: {e}")
            return None

    def _order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum: top-left has smallest sum, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Diff: top-right has smallest difference, bottom-left has largest
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
        """Get detection statistics"""
        total = self.successful_reads + self.failed_reads
        success_rate = (self.successful_reads / total * 100) if total > 0 else 0

        if len(self.read_times) > 1:
            avg_interval = np.mean(list(self.read_times))
            frames_per_sec = 1.0 / avg_interval if avg_interval > 0 else 0
            bits_per_frame = self.grid_size * self.grid_size
            bytes_per_sec = (frames_per_sec * bits_per_frame) / 8
        else:
            frames_per_sec = 0
            bytes_per_sec = 0

        return {
            'success_rate': success_rate,
            'successful_reads': self.successful_reads,
            'failed_reads': self.failed_reads,
            'frames_per_sec': frames_per_sec,
            'bytes_per_sec': bytes_per_sec,
            'bits_per_frame': self.grid_size * self.grid_size
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visual Data Link V2 - Simplified Detection')
    parser.add_argument('--grid-size', type=int, default=24,
                       help='Grid size (8-64, default: 24)')
    parser.add_argument('--cell-size', type=int, default=20,
                       help='Cell size in pixels (default: 20)')
    parser.add_argument('--message', type=str, default=None,
                       help='Custom message to transmit')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug windows')

    args = parser.parse_args()

    if args.grid_size < 8 or args.grid_size > 64:
        print("Error: Grid size must be between 8 and 64")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"Visual Data Link V2 - {args.grid_size}x{args.grid_size} Grid")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Grid: {args.grid_size}x{args.grid_size} = {args.grid_size**2} bits")
    print(f"  Max message: {args.grid_size**2 // 8} bytes")
    print(f"\nControls:")
    print(f"  SPACE/r - New random message")
    print(f"  d       - Toggle debug windows")
    print(f"  s       - Show statistics")
    print(f"  q       - Quit")
    print("="*70 + "\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {actual_width}x{actual_height}")

    # Initialize transmitter and receiver
    transmitter = GridTransmitter(grid_size=args.grid_size, cell_size=args.cell_size)
    receiver = GridReceiver(grid_size=args.grid_size, debug=args.debug)

    # Generate message
    max_bytes = args.grid_size**2 // 8
    if args.message:
        message = args.message[:max_bytes]
        message_bytes = message.encode('utf-8')
        if len(message_bytes) < max_bytes:
            message_bytes = message_bytes + b'\x00' * (max_bytes - len(message_bytes))
    else:
        chars = string.ascii_letters + string.digits + ' !?.'
        message = ''.join(random.choice(chars) for _ in range(max_bytes))
        message_bytes = message.encode('utf-8')

    print(f"\nTransmitting: '{message_bytes.decode('utf-8', errors='replace').rstrip(chr(0))}'")
    print("\nStarting...\n")

    transmitter.set_pattern_from_bytes(message_bytes)
    transmitted_message = message_bytes

    last_stats_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Display transmitter
        transmitter.display()

        # Detect and read
        detected_pattern, contour = receiver.detect_and_read(frame)

        # Visualize on frame
        display_frame = frame.copy()

        # Draw detection
        if contour is not None:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 3)
            for point in contour:
                cv2.circle(display_frame, tuple(point[0]), 8, (0, 255, 255), -1)

        # Get stats
        stats = receiver.get_stats()

        # Display info
        info_y = 30
        status_color = (0, 255, 0) if contour is not None else (0, 0, 255)
        status_text = "LOCKED" if contour is not None else "SEARCHING..."
        cv2.putText(display_frame, status_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        info_y += 40
        cv2.putText(display_frame,
                   f"Success: {stats['success_rate']:.1f}% ({stats['successful_reads']}/{stats['successful_reads']+stats['failed_reads']})",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if stats['bytes_per_sec'] > 0:
            info_y += 35
            cv2.putText(display_frame,
                       f"Rate: {stats['frames_per_sec']:.1f} fps, {stats['bytes_per_sec']:.0f} B/s",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show messages
        if detected_pattern is not None:
            rx_bytes = receiver.pattern_to_bytes(detected_pattern)
            rx_message = rx_bytes.decode('utf-8', errors='replace').rstrip('\x00')
            tx_message = transmitted_message.decode('utf-8', errors='replace').rstrip('\x00')

            match = rx_bytes == transmitted_message
            match_color = (0, 255, 255) if match else (0, 165, 255)
            match_text = "MATCH!" if match else "MISMATCH"

            cv2.putText(display_frame, match_text, (10, display_frame.shape[0] - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, match_color, 2)

            # Truncate for display
            max_len = 40
            tx_disp = tx_message[:max_len] + ('...' if len(tx_message) > max_len else '')
            rx_disp = rx_message[:max_len] + ('...' if len(rx_message) > max_len else '')

            cv2.putText(display_frame, f"TX: {tx_disp}", (10, display_frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"RX: {rx_disp}", (10, display_frame.shape[0] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Camera Feed', display_frame)

        # Show debug windows
        if receiver.debug:
            for name, img in receiver.debug_images.items():
                cv2.imshow(f'Debug: {name}', img)

        # Print periodic stats
        if time.time() - last_stats_time > 5.0 and stats['bytes_per_sec'] > 0:
            print(f"Stats: {stats['frames_per_sec']:.1f} fps | "
                  f"{stats['bytes_per_sec']:.0f} B/s | "
                  f"{stats['success_rate']:.1f}% success")
            last_stats_time = time.time()

        # Handle input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') or key == ord('r'):
            chars = string.ascii_letters + string.digits + ' !?.'
            message = ''.join(random.choice(chars) for _ in range(max_bytes))
            transmitted_message = message.encode('utf-8')
            transmitter.set_pattern_from_bytes(transmitted_message)
            print(f"New message: '{message}'")
        elif key == ord('d'):
            receiver.debug = not receiver.debug
            if not receiver.debug:
                # Close all debug windows
                for name in list(receiver.debug_images.keys()):
                    cv2.destroyWindow(f'Debug: {name}')
            print(f"Debug mode: {'ON' if receiver.debug else 'OFF'}")
        elif key == ord('s'):
            print("\n" + "="*50)
            print("Statistics:")
            print(f"  Grid: {args.grid_size}x{args.grid_size}")
            print(f"  Success: {stats['success_rate']:.1f}%")
            print(f"  Rate: {stats['frames_per_sec']:.1f} fps, {stats['bytes_per_sec']:.0f} B/s")
            print(f"  TX: '{transmitted_message.decode('utf-8', errors='replace').rstrip(chr(0))}'")
            print("="*50 + "\n")

    # Cleanup
    print("\nFinal stats:")
    stats = receiver.get_stats()
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Total reads: {stats['successful_reads'] + stats['failed_reads']}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
