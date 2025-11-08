#!/usr/bin/env python3
"""
Visual Data Link HD - High-resolution bidirectional data transmission
Supports variable grid sizes from 8x8 to 64x64 with robust detection
"""

import cv2
import numpy as np
import sys
import time
from collections import deque


class GridTransmitter:
    """Displays high-resolution grid patterns that can be read by a camera"""

    def __init__(self, grid_size=24, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pattern = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.window_name = "TX Grid (Show this to camera)"
        self.frame_counter = 0

    def set_pattern(self, pattern):
        """Set the pattern (0 for black, 1 for white)"""
        if pattern.shape != (self.grid_size, self.grid_size):
            raise ValueError(f"Pattern must be {self.grid_size}x{self.grid_size}")
        self.pattern = pattern

    def set_pattern_from_bytes(self, data):
        """Set pattern from bytes (packs bits into the grid)"""
        pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        total_bits = self.grid_size * self.grid_size

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
        """Render the grid pattern with corner markers and orientation"""
        total_size = self.grid_size * self.cell_size

        # Create RGB image for colored markers
        image = np.zeros((total_size, total_size, 3), dtype=np.uint8)

        # Draw the main grid pattern
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = 255 if self.pattern[i, j] else 0
                y1 = i * self.cell_size
                y2 = (i + 1) * self.cell_size
                x1 = j * self.cell_size
                x2 = (j + 1) * self.cell_size
                image[y1:y2, x1:x2] = [color, color, color]

        # Add thick white border
        border_size = 50
        bordered = np.zeros((total_size + 2*border_size, total_size + 2*border_size, 3), dtype=np.uint8)
        bordered[border_size:-border_size, border_size:-border_size] = image
        bordered[:border_size, :] = [255, 255, 255]
        bordered[-border_size:, :] = [255, 255, 255]
        bordered[:, :border_size] = [255, 255, 255]
        bordered[:, -border_size:] = [255, 255, 255]

        # Add red corner markers
        marker_radius = 25
        marker_offset = border_size // 2
        corners = [
            (marker_offset, marker_offset),  # Top-left
            (bordered.shape[1] - marker_offset, marker_offset),  # Top-right
            (bordered.shape[1] - marker_offset, bordered.shape[0] - marker_offset),  # Bottom-right
            (marker_offset, bordered.shape[0] - marker_offset),  # Bottom-left
        ]

        for i, corner in enumerate(corners):
            # All corners red, but top-left is larger for orientation
            radius = marker_radius + 5 if i == 0 else marker_radius
            cv2.circle(bordered, corner, radius, (0, 0, 255), -1)

        self.frame_counter += 1
        return bordered

    def display(self):
        """Display the grid in a window"""
        image = self.render()
        cv2.imshow(self.window_name, image)


class GridReceiver:
    """Detects and reads high-resolution grid patterns from camera feed"""

    def __init__(self, grid_size=24, debug=False):
        self.grid_size = grid_size
        self.last_detected = None
        self.detection_confidence = 0.0
        self.debug = debug
        self.debug_image = None
        self.frame_count = 0
        self.successful_reads = 0
        self.failed_reads = 0
        self.last_read_time = time.time()
        self.read_times = deque(maxlen=30)  # Track last 30 read times

    def detect_and_read(self, frame):
        """Detect grid in frame and read the pattern with robust detection"""
        self.frame_count += 1

        # Preprocessing for better detection
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Sharpen the image to improve edge detection
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        # Detect red corner markers
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)

        # Red color range (red wraps around in HSV)
        # More permissive range for better detection
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up mask
        kernel = np.ones((7, 7), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find circular markers
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 150:  # Minimum area for a marker
                # Check if it's roughly circular
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                if circularity > 0.5:  # Reasonably circular
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        markers.append((cx, cy, area))  # Include area for orientation

        # Need exactly 4 markers
        if len(markers) == 4:
            # Find the largest marker (top-left orientation marker)
            markers.sort(key=lambda x: x[2], reverse=True)
            largest_marker = markers[0][:2]  # Just x, y
            other_markers = [m[:2] for m in markers[1:]]

            # Order all markers
            all_markers = [largest_marker] + other_markers
            markers_array = np.array(all_markers, dtype=np.float32)
            rect = self._order_points_with_orientation(markers_array, largest_marker)

            # Validate that this forms a reasonable quadrilateral
            if self._validate_quadrilateral(rect):
                # Read the grid
                pattern = self._read_grid_from_corners(sharpened, rect)
                if pattern is not None:
                    self.last_detected = pattern
                    self.detection_confidence = 1.0
                    self.successful_reads += 1

                    # Track read timing
                    current_time = time.time()
                    self.read_times.append(current_time - self.last_read_time)
                    self.last_read_time = current_time

                    # Create contour from corners for visualization
                    contour = rect.reshape(-1, 1, 2).astype(np.int32)
                    return pattern, contour

        # If no detection, reduce confidence
        self.detection_confidence *= 0.9
        self.failed_reads += 1
        return None, None

    def _order_points_with_orientation(self, pts, largest_marker):
        """Order points ensuring largest marker is top-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left is the largest marker
        rect[0] = largest_marker

        # Remove largest from points
        other_pts = [p for p in pts if not np.array_equal(p, largest_marker)]
        other_pts = np.array(other_pts, dtype=np.float32)

        # Calculate vectors from top-left to other points
        vectors = other_pts - largest_marker

        # Calculate angles
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Sort by angle: top-right (~0°), bottom-right (~90°), bottom-left (~180°)
        sorted_indices = np.argsort(angles)

        rect[1] = other_pts[sorted_indices[0]]  # Top-right (smallest angle)
        rect[2] = other_pts[sorted_indices[1]]  # Bottom-right
        rect[3] = other_pts[sorted_indices[2]]  # Bottom-left

        return rect

    def _validate_quadrilateral(self, rect):
        """Check if the quadrilateral is reasonable"""
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        side1 = distance(rect[0], rect[1])
        side2 = distance(rect[1], rect[2])
        side3 = distance(rect[2], rect[3])
        side4 = distance(rect[3], rect[0])

        sides = [side1, side2, side3, side4]
        max_side = max(sides)
        min_side = min(sides)

        if min_side < 50:  # Too small
            return False

        if max_side / min_side > 4:  # Too skewed
            return False

        # Check that opposite sides are similar
        if max(side1, side3) / min(side1, side3) > 2:
            return False
        if max(side2, side4) / min(side2, side4) > 2:
            return False

        return True

    def _read_grid_from_corners(self, frame, corners):
        """Read the grid pattern from detected corner markers"""
        try:
            # Determine size for warped image - scale with grid size
            warp_size = max(800, self.grid_size * 15)  # At least 15px per cell
            width = warp_size
            height = warp_size

            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            # Get perspective transform and warp
            M = cv2.getPerspectiveTransform(corners, dst)
            warped = cv2.warpPerspective(frame, M, (width, height))

            # Convert to grayscale
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding for better cell detection
            binary = cv2.adaptiveThreshold(
                gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Store debug image
            if self.debug:
                self.debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            # The markers are in the border, the grid is in the center
            border_ratio = 0.12  # Border is roughly 12% of total size
            grid_start = int(width * border_ratio)
            grid_end = int(width * (1 - border_ratio))
            grid_size_px = grid_end - grid_start

            # Read the grid
            pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            cell_size = grid_size_px / self.grid_size

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Sample the center of each cell
                    y = int(grid_start + i * cell_size + cell_size / 2)
                    x = int(grid_start + j * cell_size + cell_size / 2)

                    # Sample a region in the center of the cell
                    sample_size = max(3, int(cell_size * 0.4))

                    y1 = max(0, y - sample_size)
                    y2 = min(height, y + sample_size)
                    x1 = max(0, x - sample_size)
                    x2 = min(width, x + sample_size)

                    region = binary[y1:y2, x1:x2]

                    # Count white pixels (more robust than average)
                    white_pixel_ratio = np.sum(region == 255) / region.size
                    pattern[i, j] = 1 if white_pixel_ratio > 0.5 else 0

                    # Draw debug visualization
                    if self.debug and self.debug_image is not None:
                        color = (0, 255, 0) if pattern[i, j] == 1 else (255, 0, 0)
                        cv2.rectangle(self.debug_image, (x1, y1), (x2, y2), color, 1)

            return pattern

        except Exception as e:
            if self.debug:
                print(f"Error reading grid: {e}")
            return None

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

        # Calculate average read rate
        if len(self.read_times) > 1:
            avg_interval = np.mean(list(self.read_times))
            frames_per_sec = 1.0 / avg_interval if avg_interval > 0 else 0
            bits_per_frame = self.grid_size * self.grid_size
            bits_per_sec = frames_per_sec * bits_per_frame
            bytes_per_sec = bits_per_sec / 8
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
    import string
    import random

    parser = argparse.ArgumentParser(description='Visual Data Link HD')
    parser.add_argument('--grid-size', type=int, default=24,
                       help='Grid size (8-64, default: 24)')
    parser.add_argument('--cell-size', type=int, default=20,
                       help='Cell size in pixels (default: 20)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--message', type=str, default=None,
                       help='Custom message to transmit (default: random)')

    args = parser.parse_args()

    if args.grid_size < 8 or args.grid_size > 64:
        print("Error: Grid size must be between 8 and 64")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"Visual Data Link HD - {args.grid_size}x{args.grid_size} Grid")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Grid size: {args.grid_size}x{args.grid_size} = {args.grid_size**2} bits/frame")
    print(f"  Max message: {args.grid_size**2 // 8} bytes ({args.grid_size**2 // 8} characters)")
    print(f"  Cell size: {args.cell_size}px")
    print(f"  Display size: {args.grid_size * args.cell_size}px + borders")
    print(f"\nControls:")
    print(f"  SPACE/r - Generate new random message")
    print(f"  d       - Toggle debug mode")
    print(f"  s       - Show statistics")
    print(f"  q       - Quit")
    print("="*70 + "\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Set camera to highest resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"TIP: Make sure the grid is well-lit and in focus!\n")

    # Initialize transmitter and receiver
    transmitter = GridTransmitter(grid_size=args.grid_size, cell_size=args.cell_size)
    receiver = GridReceiver(grid_size=args.grid_size, debug=args.debug)

    # Calculate how many bytes fit in the grid
    total_bits = args.grid_size * args.grid_size
    max_bytes = total_bits // 8

    # Generate or use custom message
    if args.message:
        # Use custom message, truncate or pad as needed
        message = args.message[:max_bytes]
        message_bytes = message.encode('utf-8')
        # Pad with zeros if too short
        if len(message_bytes) < max_bytes:
            message_bytes = message_bytes + b'\x00' * (max_bytes - len(message_bytes))
    else:
        # Generate random printable string
        chars = string.ascii_letters + string.digits + ' !?.'
        message = ''.join(random.choice(chars) for _ in range(max_bytes))
        message_bytes = message.encode('utf-8')

    print(f"Transmitting message ({len(message_bytes)} bytes):")
    print(f"  '{message_bytes.decode('utf-8', errors='replace').rstrip(chr(0))}'")
    print()

    # Encode message into pattern
    transmitter.set_pattern_from_bytes(message_bytes)
    transmitted_message = message_bytes

    print("Starting...\n")
    last_stats_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Display the transmitter grid
        transmitter.display()

        # Try to detect and read grid from camera
        detected_pattern, contour = receiver.detect_and_read(frame)

        # Draw detection on camera feed
        display_frame = frame.copy()

        if contour is not None:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 3)
            # Draw corner points
            for point in contour:
                cv2.circle(display_frame, tuple(point[0]), 5, (0, 255, 255), -1)

        # Get statistics
        stats = receiver.get_stats()

        # Display info on frame
        info_y = 30
        line_height = 35

        status_color = (0, 255, 0) if contour is not None else (0, 0, 255)
        status_text = "LOCKED" if contour is not None else "SEARCHING..."
        cv2.putText(display_frame, status_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        info_y += line_height
        cv2.putText(display_frame, f"Grid: {args.grid_size}x{args.grid_size} ({stats['bits_per_frame']} bits)",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_y += line_height
        cv2.putText(display_frame, f"Success: {stats['success_rate']:.1f}% ({stats['successful_reads']}/{stats['successful_reads']+stats['failed_reads']})",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_y += line_height
        if stats['bytes_per_sec'] > 0:
            cv2.putText(display_frame, f"Rate: {stats['frames_per_sec']:.1f} fps, {stats['bytes_per_sec']:.0f} B/s",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show transmitted and received messages
        if detected_pattern is not None:
            rx_bytes = receiver.pattern_to_bytes(detected_pattern)
            rx_message = rx_bytes.decode('utf-8', errors='replace').rstrip('\x00')
            tx_message = transmitted_message.decode('utf-8', errors='replace').rstrip('\x00')

            # Check if they match
            if rx_bytes == transmitted_message:
                match_color = (0, 255, 255)
                match_text = "MATCH!"
            else:
                match_color = (0, 165, 255)
                match_text = "MISMATCH"

            cv2.putText(display_frame, match_text, (10, display_frame.shape[0] - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, match_color, 2)

            # Display messages (truncated if too long)
            max_display_len = 40
            tx_display = tx_message[:max_display_len] + ('...' if len(tx_message) > max_display_len else '')
            rx_display = rx_message[:max_display_len] + ('...' if len(rx_message) > max_display_len else '')

            cv2.putText(display_frame, f"TX: {tx_display}", (10, display_frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"RX: {rx_display}", (10, display_frame.shape[0] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show debug mode status
        if receiver.debug:
            cv2.putText(display_frame, "DEBUG MODE", (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow('Camera Feed', display_frame)

        # Show debug window if in debug mode
        if receiver.debug and receiver.debug_image is not None:
            cv2.imshow('Debug - Binary Grid', receiver.debug_image)

        # Print stats periodically
        if time.time() - last_stats_time > 5.0 and stats['bytes_per_sec'] > 0:
            print(f"Stats: {stats['frames_per_sec']:.1f} fps | "
                  f"{stats['bytes_per_sec']:.0f} B/s | "
                  f"{stats['success_rate']:.1f}% success")
            last_stats_time = time.time()

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') or key == ord('r'):
            # Generate new random message
            chars = string.ascii_letters + string.digits + ' !?.'
            message = ''.join(random.choice(chars) for _ in range(max_bytes))
            transmitted_message = message.encode('utf-8')
            transmitter.set_pattern_from_bytes(transmitted_message)
            print(f"New message: '{message}'")
        elif key == ord('d'):
            # Toggle debug mode
            receiver.debug = not receiver.debug
            if not receiver.debug:
                cv2.destroyWindow('Debug - Binary Grid')
            print(f"Debug mode: {'ON' if receiver.debug else 'OFF'}")
        elif key == ord('s'):
            # Show detailed stats
            print("\n" + "="*50)
            print("Statistics:")
            print(f"  Grid: {args.grid_size}x{args.grid_size} ({stats['bits_per_frame']} bits/frame)")
            print(f"  Max message length: {max_bytes} bytes")
            print(f"  Success rate: {stats['success_rate']:.1f}%")
            print(f"  Successful reads: {stats['successful_reads']}")
            print(f"  Failed reads: {stats['failed_reads']}")
            print(f"  Frame rate: {stats['frames_per_sec']:.1f} fps")
            print(f"  Data rate: {stats['bytes_per_sec']:.0f} bytes/sec")
            print(f"  Bit rate: {stats['bytes_per_sec']*8:.0f} bits/sec")
            print(f"  Current TX: '{transmitted_message.decode('utf-8', errors='replace').rstrip(chr(0))}'")
            print("="*50 + "\n")

    # Cleanup
    print("\n" + "="*50)
    print("Final Statistics:")
    stats = receiver.get_stats()
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Total reads: {stats['successful_reads'] + stats['failed_reads']}")
    print(f"  Average data rate: {stats['bytes_per_sec']:.0f} bytes/sec")
    print("="*50 + "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
