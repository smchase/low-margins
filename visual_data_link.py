#!/usr/bin/env python3
"""
Visual Data Link - Bidirectional data transmission using 4x4 black/white grids
Two computers running this can point cameras at each other to exchange data
"""

import cv2
import numpy as np
import sys
import time


class GridTransmitter:
    """Displays a 4x4 grid pattern that can be read by a camera"""

    def __init__(self, grid_size=4, cell_size=100):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pattern = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.window_name = "Transmitting Grid (Show this to camera)"

    def set_pattern(self, pattern):
        """Set the 4x4 pattern (0 for black, 1 for white)"""
        if pattern.shape != (self.grid_size, self.grid_size):
            raise ValueError(f"Pattern must be {self.grid_size}x{self.grid_size}")
        self.pattern = pattern

    def set_pattern_from_int(self, value):
        """Set pattern from a 16-bit integer (0-65535)"""
        pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                pattern[i, j] = (value >> bit_index) & 1
        self.pattern = pattern

    def get_pattern_as_int(self):
        """Convert current pattern to integer"""
        value = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                if self.pattern[i, j]:
                    value |= (1 << bit_index)
        return value

    def render(self):
        """Render the grid pattern as an image"""
        total_size = self.grid_size * self.cell_size
        image = np.zeros((total_size, total_size), dtype=np.uint8)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = 255 if self.pattern[i, j] else 0
                y1 = i * self.cell_size
                y2 = (i + 1) * self.cell_size
                x1 = j * self.cell_size
                x2 = (j + 1) * self.cell_size
                image[y1:y2, x1:x2] = color

        # Add a border around the entire grid
        border_size = 20
        bordered = np.zeros((total_size + 2*border_size, total_size + 2*border_size), dtype=np.uint8)
        bordered[border_size:-border_size, border_size:-border_size] = image
        bordered[:border_size, :] = 128  # Gray border
        bordered[-border_size:, :] = 128
        bordered[:, :border_size] = 128
        bordered[:, -border_size:] = 128

        return bordered

    def display(self):
        """Display the grid in a window"""
        image = self.render()
        cv2.imshow(self.window_name, image)


class GridReceiver:
    """Detects and reads 4x4 grid patterns from camera feed"""

    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.last_detected = None
        self.detection_confidence = 0.0

    def detect_and_read(self, frame):
        """Detect grid in frame and read the pattern"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for rectangular contours that could be our grid
        for contour in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Look for quadrilaterals with reasonable area
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > 10000:  # Minimum size threshold
                    # Try to read the grid
                    pattern = self._read_grid_from_contour(gray, approx)
                    if pattern is not None:
                        self.last_detected = pattern
                        self.detection_confidence = 1.0
                        return pattern, approx

        # If no detection, reduce confidence
        self.detection_confidence *= 0.9
        return None, None

    def _read_grid_from_contour(self, gray, contour):
        """Read the grid pattern from a detected quadrilateral"""
        try:
            # Get the four corners
            pts = contour.reshape(4, 2).astype(np.float32)

            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = self._order_points(pts)

            # Determine size for warped image
            width = 400
            height = 400

            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            # Get perspective transform and warp
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray, M, (width, height))

            # Read the 4x4 grid
            pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            cell_size = width // self.grid_size

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Sample the center of each cell
                    y = i * cell_size + cell_size // 2
                    x = j * cell_size + cell_size // 2

                    # Get average brightness in a small region around center
                    region = warped[y-10:y+10, x-10:x+10]
                    avg_brightness = np.mean(region)

                    # Threshold to determine if it's white (1) or black (0)
                    pattern[i, j] = 1 if avg_brightness > 127 else 0

            return pattern

        except Exception as e:
            return None

    def _order_points(self, pts):
        """Order points in clockwise order starting from top-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest difference, bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def pattern_to_int(self, pattern):
        """Convert pattern to integer"""
        if pattern is None:
            return None
        value = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                bit_index = i * self.grid_size + j
                if pattern[i, j]:
                    value |= (1 << bit_index)
        return value


def main():
    print("\n" + "="*60)
    print("Visual Data Link - 4x4 Grid Communication")
    print("="*60)
    print("\nInstructions:")
    print("  - Point two computers running this at each other")
    print("  - Press number keys (0-9) to change transmitted pattern")
    print("  - Press SPACE to cycle through test patterns")
    print("  - Press 'q' to quit")
    print("="*60 + "\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Initialize transmitter and receiver
    transmitter = GridTransmitter(grid_size=4, cell_size=100)
    receiver = GridReceiver(grid_size=4)

    # Set initial pattern (checkerboard)
    initial_pattern = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=np.uint8)
    transmitter.set_pattern(initial_pattern)

    # Test patterns
    test_patterns = [
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8),  # Checkerboard
        np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.uint8),  # Stripes
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8),  # Center square
        np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.uint8),  # Corners
    ]
    pattern_index = 0

    print("Starting... Press SPACE to cycle patterns\n")

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
            cv2.putText(display_frame, "GRID DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display info
        tx_value = transmitter.get_pattern_as_int()
        rx_value = receiver.pattern_to_int(detected_pattern) if detected_pattern is not None else None

        info_y = 70
        cv2.putText(display_frame, f"TX: {tx_value:05d} (0x{tx_value:04X})", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if rx_value is not None:
            cv2.putText(display_frame, f"RX: {rx_value:05d} (0x{rx_value:04X})", (10, info_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show if TX and RX match (when pointing at a mirror or another screen)
            if rx_value == tx_value:
                cv2.putText(display_frame, "MATCH!", (10, info_y + 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "RX: NO SIGNAL", (10, info_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Camera Feed (Point at another screen)', display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # Cycle through test patterns
            pattern_index = (pattern_index + 1) % len(test_patterns)
            transmitter.set_pattern(test_patterns[pattern_index])
            print(f"Switched to pattern {pattern_index + 1}/{len(test_patterns)}")
        elif ord('0') <= key <= ord('9'):
            # Set pattern from digit (0-9)
            digit = key - ord('0')
            value = digit * 6553  # Scale to use full 16-bit range
            transmitter.set_pattern_from_int(value)
            print(f"Set pattern to {value} (digit {digit})")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
