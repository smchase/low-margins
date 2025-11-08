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

    def __init__(self, grid_size=4, cell_size=120):
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
        """Render the grid pattern as an image with corner markers"""
        total_size = self.grid_size * self.cell_size

        # Create RGB image for colored markers
        image = np.zeros((total_size, total_size, 3), dtype=np.uint8)

        # Draw the grid pattern
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = 255 if self.pattern[i, j] else 0
                y1 = i * self.cell_size
                y2 = (i + 1) * self.cell_size
                x1 = j * self.cell_size
                x2 = (j + 1) * self.cell_size
                image[y1:y2, x1:x2] = [color, color, color]

        # Add thick white border around the entire grid
        border_size = 40
        bordered = np.zeros((total_size + 2*border_size, total_size + 2*border_size, 3), dtype=np.uint8)
        bordered[border_size:-border_size, border_size:-border_size] = image
        bordered[:border_size, :] = [255, 255, 255]  # White border
        bordered[-border_size:, :] = [255, 255, 255]
        bordered[:, :border_size] = [255, 255, 255]
        bordered[:, -border_size:] = [255, 255, 255]

        # Add colored corner markers (red circles) for easy detection
        marker_radius = 20
        marker_offset = border_size // 2
        corners = [
            (marker_offset, marker_offset),  # Top-left
            (bordered.shape[1] - marker_offset, marker_offset),  # Top-right
            (bordered.shape[1] - marker_offset, bordered.shape[0] - marker_offset),  # Bottom-right
            (marker_offset, bordered.shape[0] - marker_offset),  # Bottom-left
        ]

        for corner in corners:
            cv2.circle(bordered, corner, marker_radius, (0, 0, 255), -1)  # Red filled circles

        return bordered

    def display(self):
        """Display the grid in a window"""
        image = self.render()
        cv2.imshow(self.window_name, image)


class GridReceiver:
    """Detects and reads 4x4 grid patterns from camera feed"""

    def __init__(self, grid_size=4, debug=False):
        self.grid_size = grid_size
        self.last_detected = None
        self.detection_confidence = 0.0
        self.debug = debug
        self.debug_image = None

    def detect_and_read(self, frame):
        """Detect grid in frame and read the pattern"""
        # Preprocessing
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Detect red corner markers
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Red color range (red wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find red circles (corner markers)
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find circular markers
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Minimum area for a marker
                # Check if it's roughly circular
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                if circularity > 0.6:  # Reasonably circular
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        markers.append((cx, cy))

        # Need exactly 4 markers
        if len(markers) == 4:
            # Order the markers to form a quadrilateral
            markers_array = np.array(markers, dtype=np.float32)
            rect = self._order_points(markers_array)

            # Validate that this forms a reasonable quadrilateral
            if self._validate_quadrilateral(rect):
                # Read the grid
                pattern = self._read_grid_from_corners(frame, rect)
                if pattern is not None:
                    self.last_detected = pattern
                    self.detection_confidence = 1.0

                    # Create contour from corners for visualization
                    contour = rect.reshape(-1, 1, 2).astype(np.int32)
                    return pattern, contour

        # If no detection, reduce confidence
        self.detection_confidence *= 0.9
        return None, None

    def _validate_quadrilateral(self, rect):
        """Check if the quadrilateral is reasonable (not too skewed)"""
        # Check that sides are similar in length (within 3x ratio)
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

        if max_side / min_side > 3:  # Too skewed
            return False

        return True

    def _read_grid_from_corners(self, frame, corners):
        """Read the grid pattern from detected corner markers"""
        try:
            # Determine size for warped image
            width = 600
            height = 600

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

            # Store debug image
            if self.debug:
                self.debug_image = warped.copy()

            # The markers are in the border, the grid is in the center
            # Calculate the grid region (excluding the border with markers)
            border_ratio = 0.15  # Border is roughly 15% of total size
            grid_start = int(width * border_ratio)
            grid_end = int(width * (1 - border_ratio))
            grid_size_px = grid_end - grid_start

            # Read the 4x4 grid
            pattern = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            cell_size = grid_size_px / self.grid_size

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Sample the center of each cell
                    y = int(grid_start + i * cell_size + cell_size / 2)
                    x = int(grid_start + j * cell_size + cell_size / 2)

                    # Get average brightness in a region around center
                    sample_size = int(cell_size * 0.3)  # Sample 30% of cell size
                    sample_size = max(10, sample_size)

                    y1 = max(0, y - sample_size)
                    y2 = min(height, y + sample_size)
                    x1 = max(0, x - sample_size)
                    x2 = min(width, x + sample_size)

                    region = gray_warped[y1:y2, x1:x2]
                    avg_brightness = np.mean(region)

                    # Threshold to determine if it's white (1) or black (0)
                    pattern[i, j] = 1 if avg_brightness > 127 else 0

                    # Draw debug visualization
                    if self.debug and self.debug_image is not None:
                        color = (0, 255, 0) if pattern[i, j] == 1 else (255, 0, 0)
                        cv2.rectangle(self.debug_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(self.debug_image, str(pattern[i, j]),
                                  (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return pattern

        except Exception as e:
            if self.debug:
                print(f"Error reading grid: {e}")
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
    print("  - Press 'd' to toggle debug mode")
    print("  - Press 'q' to quit")
    print("="*60 + "\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Set camera to higher resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize transmitter and receiver
    transmitter = GridTransmitter(grid_size=4, cell_size=120)
    receiver = GridReceiver(grid_size=4, debug=False)

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
    print("TIP: Make sure the transmitted grid is well-lit and in focus!\n")

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

        # Show debug mode status
        if receiver.debug:
            cv2.putText(display_frame, "DEBUG MODE", (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow('Camera Feed (Point at another screen)', display_frame)

        # Show debug window if in debug mode
        if receiver.debug and receiver.debug_image is not None:
            cv2.imshow('Debug - Warped Grid', receiver.debug_image)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # Cycle through test patterns
            pattern_index = (pattern_index + 1) % len(test_patterns)
            transmitter.set_pattern(test_patterns[pattern_index])
            print(f"Switched to pattern {pattern_index + 1}/{len(test_patterns)}")
        elif key == ord('d'):
            # Toggle debug mode
            receiver.debug = not receiver.debug
            if not receiver.debug:
                cv2.destroyWindow('Debug - Warped Grid')
            print(f"Debug mode: {'ON' if receiver.debug else 'OFF'}")
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
