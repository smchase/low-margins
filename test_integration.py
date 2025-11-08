"""
Integration test: Simulate transmitter grid and feed to receiver detection
"""

import cv2
import numpy as np
from config import GRID_SIZE, COLOR_PALETTE, CELL_WIDTH, CELL_HEIGHT, DISPLAY_WIDTH, DISPLAY_HEIGHT
from encoder import MessageEncoder
from receiver import GridDetector
import struct


def create_transmitter_frame_with_markers(grid, add_border=True, add_corners=True):
    """
    Create a realistic transmitter frame with grid, green border, and corner markers.

    Args:
        grid: Grid from encoder (GRID_SIZE x GRID_SIZE)
        add_border: If True, add green border around grid
        add_corners: If True, add black corner markers

    Returns:
        BGR frame suitable for camera input
    """
    # Create white background
    frame = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 255

    # Calculate grid position (centered)
    grid_w = GRID_SIZE * CELL_WIDTH
    grid_h = GRID_SIZE * CELL_HEIGHT
    grid_x = (DISPLAY_WIDTH - grid_w) // 2
    grid_y = (DISPLAY_HEIGHT - grid_h) // 2

    # Draw grid cells manually with offset
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Grid values are 0-15 (4-bit), matching our 16-color palette
            color_idx = int(grid[row, col])
            color = COLOR_PALETTE[color_idx]

            x1 = grid_x + col * CELL_WIDTH
            y1 = grid_y + row * CELL_HEIGHT
            x2 = x1 + CELL_WIDTH
            y2 = y1 + CELL_HEIGHT

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

    # Add green border around grid
    if add_border:
        border_thickness = 6
        green_color = (0, 255, 0)  # BGR
        cv2.rectangle(
            frame,
            (grid_x - border_thickness, grid_y - border_thickness),
            (grid_x + grid_w + border_thickness, grid_y + grid_h + border_thickness),
            green_color,
            border_thickness
        )

    # Add black corner markers
    if add_corners:
        marker_size = 30
        marker_color = (0, 0, 0)  # BGR (black)
        x1, y1 = grid_x - 15, grid_y - 15
        x2, y2 = grid_x + grid_w + 15, grid_y + grid_h + 15

        # Top-left
        cv2.rectangle(frame, (x1, y1), (x1 + marker_size, y1 + marker_size), marker_color, -1)
        # Top-right
        cv2.rectangle(frame, (x2 - marker_size, y1), (x2, y1 + marker_size), marker_color, -1)
        # Bottom-left
        cv2.rectangle(frame, (x1, y2 - marker_size), (x1 + marker_size, y2), marker_color, -1)
        # Bottom-right
        cv2.rectangle(frame, (x2 - marker_size, y2 - marker_size), (x2, y2), marker_color, -1)

    return frame


def test_detection_and_decoding(test_message="Hello"):
    """
    Simulate transmitter encoding a message and receiver detecting/decoding it.
    """
    print("=" * 60)
    print("Integration Test: Transmitter → Receiver Detection & Decoding")
    print("=" * 60)

    # Create encoder and queue test message
    encoder = MessageEncoder()
    encoder.queue_message(test_message)

    # Create detector
    detector = GridDetector(CELL_WIDTH, CELL_HEIGHT)

    print(f"\nTest message: '{test_message}'")
    print(f"Message length: {len(test_message)} bytes")

    # Generate and test first few frames
    detected_count = 0
    decoded_frames = []

    for frame_num in range(10):
        print(f"\n--- Frame {frame_num} ---")

        # Generate frame from encoder
        grid = encoder.create_frame()
        print(f"  Encoder grid shape: {grid.shape}")

        # Create simulated camera frame with border and markers
        camera_frame = create_transmitter_frame_with_markers(
            grid,
            add_border=True,
            add_corners=True
        )

        # Test detection
        grid_found, x1, y1, x2, y2 = detector.detect_green_border(camera_frame)
        print(f"  Grid detected: {grid_found}")

        if grid_found:
            detected_count += 1
            # For this test, use the known grid position from the frame layout
            grid_w = GRID_SIZE * CELL_WIDTH
            grid_h = GRID_SIZE * CELL_HEIGHT
            exact_x1 = (DISPLAY_WIDTH - grid_w) // 2
            exact_y1 = (DISPLAY_HEIGHT - grid_h) // 2
            exact_x2 = exact_x1 + grid_w
            exact_y2 = exact_y1 + grid_h

            # Extract and decode grid
            detected_grid = detector.extract_grid_fast(camera_frame, exact_x1, exact_y1, exact_x2, exact_y2)

            # Simple decode: extract message from grid
            from color_utils import grid_to_bytes
            frame_bytes = grid_to_bytes(detected_grid)

            try:
                frame_counter = struct.unpack('>I', frame_bytes[0:4])[0]
                message_id = struct.unpack('>H', frame_bytes[4:6])[0]
                message_length = struct.unpack('>H', frame_bytes[6:8])[0]
                payload = frame_bytes[8:2048]
                decoded_text = payload[:message_length].decode('utf-8', errors='ignore')

                print(f"  Frame counter: {frame_counter}")
                print(f"  Message ID: {message_id:04x}")
                print(f"  Message length: {message_length}")
                print(f"  Decoded text: '{decoded_text}'")

                decoded_frames.append({
                    'frame': frame_num,
                    'text': decoded_text,
                    'length': message_length
                })
            except Exception as e:
                print(f"  Decode error: {e}")
        else:
            print(f"  ❌ Border detection failed")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Frames processed: 10")
    print(f"Frames with detected border: {detected_count}/10")
    print(f"Detection success rate: {detected_count*10}%")
    print(f"Frames decoded: {len(decoded_frames)}")

    if decoded_frames:
        print("\nDecoded messages:")
        for info in decoded_frames:
            print(f"  Frame {info['frame']}: '{info['text']}' ({info['length']} bytes)")

        # Check if we got the full message
        full_text = ''.join([info['text'] for info in decoded_frames])
        print(f"\nFull reconstructed text: '{full_text}'")
        if test_message in full_text:
            print("✓ Message successfully decoded!")
        else:
            print("✗ Message incomplete or corrupted")
    else:
        print("✗ No frames were successfully decoded")


def test_visual_output():
    """
    Create and display simulated frames visually.
    """
    print("\n" + "=" * 60)
    print("Visual Test: Displaying simulated transmitter frames")
    print("=" * 60)

    encoder = MessageEncoder()
    encoder.queue_message("TEST")
    detector = GridDetector(CELL_WIDTH, CELL_HEIGHT)

    for frame_num in range(5):
        # Generate frame
        grid = encoder.create_frame()

        # Create camera frame
        camera_frame = create_transmitter_frame_with_markers(grid, add_border=True, add_corners=True)

        # Test detection and draw overlay
        grid_found, x1, y1, x2, y2 = detector.detect_green_border(camera_frame)

        # Create display copy
        display_frame = camera_frame.copy()

        if grid_found:
            # Draw detected bounds
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(display_frame, "DETECTED", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "NOT DETECTED", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display
        cv2.imshow("Simulated Transmitter → Receiver", display_frame)

        # Wait for key press
        key = cv2.waitKey(500)  # 500ms per frame
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run detection/decoding test
    test_detection_and_decoding("Hello World")

    # Optionally show visual output
    print("\n\nPress any key to view visual test, or Ctrl+C to skip...")
    try:
        import time
        time.sleep(2)
        test_visual_output()
    except KeyboardInterrupt:
        print("Skipped visual test")
