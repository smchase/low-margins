"""
Receiver: Captures camera feed, detects grid, decodes message
"""

import cv2
import numpy as np
import threading
import time
from color_utils import ColorDetector, draw_grid
from decoder import MessageDecoder
from config import (
    GRID_SIZE, DISPLAY_WIDTH, DISPLAY_HEIGHT, CELL_WIDTH, CELL_HEIGHT,
    TARGET_FPS, FRAME_TIME_MS, COLOR_PALETTE, COLOR_NAMES
)


class GridDetector:
    """Detects and extracts the 128x128 color grid from camera frame"""

    def __init__(self, cell_width, cell_height):
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.detector = ColorDetector()

    def extract_grid(self, frame):
        """
        Extract the 128x128 grid from the camera frame.

        This is a simple approach: assumes the grid fills the frame.
        For robustness, you could add corner detection markers.

        Args:
            frame: OpenCV image (BGR)

        Returns:
            grid: numpy array (128, 128) with values 0-15
        """
        h, w = frame.shape[:2]

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        # Sample color from center of each cell
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Calculate cell bounds in frame space
                x1 = int((col / GRID_SIZE) * w)
                x2 = int(((col + 1) / GRID_SIZE) * w)
                y1 = int((row / GRID_SIZE) * h)
                y2 = int(((row + 1) / GRID_SIZE) * h)

                # Ensure bounds
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                if x2 - x1 > 0 and y2 - y1 > 0:
                    # Extract cell region
                    cell_region = frame[y1:y2, x1:x2]

                    # Detect dominant color
                    colors = self.detector.detect_colors_adaptive(cell_region)
                    color_idx = np.bincount(colors.flatten()).argmax()
                    grid[row, col] = color_idx

        return grid

    def extract_grid_fast(self, frame):
        """
        Faster grid extraction using direct color detection on sampled region.
        """
        h, w = frame.shape[:2]
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = int((col + 0.5) * w / GRID_SIZE)
                y = int((row + 0.5) * h / GRID_SIZE)

                x = np.clip(x, 0, w - 1)
                y = np.clip(y, 0, h - 1)

                pixel = frame[y, x]
                grid[row, col] = self.detector.detect_color(pixel)

        return grid


class ReceiverApp:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.running = True
        self.decoder = MessageDecoder()
        self.grid_detector = GridDetector(CELL_WIDTH, CELL_HEIGHT)
        self.last_frame = None
        self.last_grid = None
        self.lock = threading.Lock()
        self.fps_counter = 0
        self.fps_time = time.time()

    def camera_thread(self):
        """Thread that captures camera frames and decodes grid"""
        print(f"Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            self.running = False
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        frame_time = FRAME_TIME_MS / 1000.0

        print("Camera thread started")

        while self.running:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Detect grid
            grid = self.grid_detector.extract_grid_fast(frame)

            # Decode
            decoded_text = self.decoder.process_frame(grid)

            # Store for display thread
            with self.lock:
                self.last_frame = frame
                self.last_grid = grid

            self.fps_counter += 1

            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        cap.release()

    def display_thread(self):
        """Thread that displays split-screen: camera feed + decoded message"""
        print("Display thread started")

        window_name = "NO-MARGIN-VIS Receiver"
        display_available = False

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, DISPLAY_WIDTH * 2, DISPLAY_HEIGHT)
            display_available = True
            print("Display window created successfully")
        except cv2.error as e:
            print(f"WARNING: Display not available - running in headless mode")
            print(f"  (This is normal on headless/remote systems)")
            print(f"  Grid decoding is still working")

        while self.running:
            display_start = time.time()

            with self.lock:
                frame = self.last_frame
                grid = self.last_grid

            if frame is None or grid is None:
                time.sleep(0.01)
                continue

            # Only display if display is available
            if display_available:
                # Left side: camera feed
                h, w = frame.shape[:2]
                # Scale to fit left half
                target_h = DISPLAY_HEIGHT
                target_w = int(target_h * w / h)
                camera_display = cv2.resize(frame, (target_w, target_h))

                # Pad to exact width
                pad_left = (DISPLAY_WIDTH - target_w) // 2
                pad_right = DISPLAY_WIDTH - target_w - pad_left
                camera_display = cv2.copyMakeBorder(
                    camera_display, 0, 0, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

                # Right side: reconstructed grid
                grid_display = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                draw_grid(grid_display, grid, CELL_WIDTH, CELL_HEIGHT)

                # Add decoded message as text overlay
                message = self.decoder.get_full_message()
                y_offset = 40
                x_offset = 10

                # Display message with line wrapping
                lines = []
                max_chars_per_line = 40
                for i in range(0, len(message), max_chars_per_line):
                    lines.append(message[i:i+max_chars_per_line])

                for i, line in enumerate(lines[-10:]):  # Show last 10 lines
                    y = y_offset + i * 25
                    if y < DISPLAY_HEIGHT - 20:
                        cv2.putText(
                            grid_display, line,
                            (x_offset, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1
                        )

                # Add stats
                stats = self.decoder.get_stats()
                fps = self.fps_counter / (time.time() - self.fps_time + 0.001)
                stats_text = f"FPS: {fps:.1f} | Frames: {stats['frames']} | Msg: {stats['message_length']}ch"
                cv2.putText(
                    grid_display, stats_text,
                    (x_offset, DISPLAY_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1
                )

                # Combine left and right
                combined = np.hstack([camera_display, grid_display])

                try:
                    cv2.imshow(window_name, combined)
                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                    elif key == ord('c'):
                        # Clear decoded message
                        self.decoder.decoded_text = ""
                except cv2.error:
                    # Display was lost
                    display_available = False

            # Maintain frame rate
            elapsed = time.time() - display_start
            sleep_time = max(0, FRAME_TIME_MS / 1000.0 - elapsed)
            time.sleep(sleep_time)

        if display_available:
            cv2.destroyAllWindows()

    def start(self):
        """Start receiver"""
        print(f"Starting receiver")

        # Start camera thread
        camera_t = threading.Thread(target=self.camera_thread, daemon=True)
        camera_t.start()

        # Start display thread
        display_t = threading.Thread(target=self.display_thread, daemon=True)
        display_t.start()

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False

        camera_t.join(timeout=2)
        display_t.join(timeout=2)


if __name__ == "__main__":
    app = ReceiverApp(camera_id=0)

    print("=" * 60)
    print("NO-MARGIN-VIS RECEIVER")
    print("=" * 60)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"FPS: {TARGET_FPS}")
    print(f"\nControls:")
    print(f"  q: Quit")
    print(f"  c: Clear decoded message")
    print("=" * 60)

    app.start()
