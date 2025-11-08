#!/usr/bin/env python3
"""
Scrolling barcode encoder - transmit data as colored bars scrolling down the screen
Each bar encodes 4 bits (16 colors), bars scroll vertically for continuous transmission
"""
import numpy as np
import cv2
from typing import List, Tuple
import time


# 16 highly distinct colors (same palette as before)
COLOR_PALETTE = [
    (230, 25, 75),    # Bright Red - 0000
    (255, 255, 255),  # White - 0001
    (245, 130, 49),   # Orange - 0010
    (154, 99, 36),    # Brown - 0011
    (255, 225, 25),   # Yellow - 0100
    (128, 128, 0),    # Olive - 0101
    (60, 180, 75),    # Lime Green - 0110
    (34, 139, 34),    # Dark Green - 0111
    (0, 255, 127),    # Spring Green - 1000
    (255, 69, 0),     # Red Orange - 1001
    (135, 206, 235),  # Sky Blue - 1010
    (255, 215, 0),    # Gold - 1011
    (0, 0, 117),      # Navy - 1100
    (145, 30, 180),   # Purple - 1101
    (240, 50, 230),   # Magenta - 1110
    (250, 190, 212),  # Pink - 1111
]


# Sync pattern - special sequence to mark start of data frame
# Using alternating pattern: Red, White, Red, White (0, 1, 0, 1)
SYNC_PATTERN = [0, 1, 0, 1, 0, 1, 0, 1]  # 8 bars


def bits_to_color(bits: int) -> Tuple[int, int, int]:
    """Convert 4-bit value to RGB color"""
    if not 0 <= bits <= 15:
        raise ValueError(f"Bits must be between 0 and 15, got {bits}")
    return COLOR_PALETTE[bits]


def encode_data_to_bars(data: bytes) -> List[int]:
    """
    Encode binary data into a sequence of 4-bit values (bars)
    
    Args:
        data: Binary data to encode
        
    Returns:
        List of integers 0-15, each representing one bar color
    """
    bars = []
    
    # Add sync pattern at start
    bars.extend(SYNC_PATTERN)
    
    # Convert each byte to two bars (4 bits each)
    for byte in data:
        # High nibble (first 4 bits)
        high_nibble = (byte >> 4) & 0x0F
        bars.append(high_nibble)
        
        # Low nibble (last 4 bits)
        low_nibble = byte & 0x0F
        bars.append(low_nibble)
    
    # Add sync pattern at end
    bars.extend(SYNC_PATTERN)
    
    return bars


def create_barcode_frame(bars: List[int], bar_width: int = 10, 
                         screen_height: int = 1080, screen_width: int = 1920,
                         scroll_offset: int = 0) -> np.ndarray:
    """
    Create a single frame of the scrolling barcode
    
    Args:
        bars: List of bar values (0-15)
        bar_width: Width of each bar in pixels
        screen_height: Screen height
        screen_width: Screen width  
        scroll_offset: Vertical scroll offset in pixels
        
    Returns:
        RGB frame as numpy array
    """
    frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Calculate how many bars fit horizontally
    num_bars = screen_width // bar_width
    
    # Fill the screen with bars, repeating the data if needed
    for i in range(num_bars):
        bar_idx = i % len(bars)
        bar_value = bars[bar_idx]
        color = bits_to_color(bar_value)
        
        # Calculate bar position
        x_start = i * bar_width
        x_end = min(x_start + bar_width, screen_width)
        
        # Draw vertical bar (full height for now, will shift for scrolling)
        frame[:, x_start:x_end] = color
    
    # Apply scrolling by shifting the image vertically
    if scroll_offset > 0:
        # Shift down
        frame = np.roll(frame, scroll_offset, axis=0)
        # Fill the top with the same pattern (wrap around)
        # This creates continuous scrolling effect
    
    return frame


def create_horizontal_barcode_frame(bars: List[int], bar_height: int = 4,
                                   screen_height: int = 1080, screen_width: int = 1920,
                                   scroll_offset: int = 0) -> np.ndarray:
    """
    Create a frame with horizontal bars scrolling down (better for throughput)
    
    Args:
        bars: List of bar values (0-15)
        bar_height: Height of each bar in pixels
        screen_height: Screen height
        screen_width: Screen width
        scroll_offset: Vertical scroll offset in pixels
        
    Returns:
        RGB frame as numpy array
    """
    # Total height needed for all bars
    total_height = len(bars) * bar_height
    
    # Create extended canvas to allow smooth scrolling
    canvas_height = total_height + screen_height
    canvas = np.zeros((canvas_height, screen_width, 3), dtype=np.uint8)
    
    # Draw all bars
    for i, bar_value in enumerate(bars):
        color = bits_to_color(bar_value)
        y_start = i * bar_height
        y_end = y_start + bar_height
        canvas[y_start:y_end, :] = color
    
    # Extract the visible portion based on scroll offset
    # Wrap around when we reach the end
    offset = scroll_offset % total_height
    
    # Extract the visible screen portion
    frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    if offset + screen_height <= canvas_height:
        frame = canvas[offset:offset+screen_height, :]
    else:
        # Need to wrap around
        first_part = canvas[offset:canvas_height, :]
        remaining = screen_height - (canvas_height - offset)
        second_part = canvas[0:remaining, :]
        frame = np.vstack([first_part, second_part])
    
    return frame


def stream_data(data: bytes, bar_height: int = 4, fps: int = 30, 
               screen_width: int = 1920, screen_height: int = 1080,
               loop: bool = True):
    """
    Stream data as scrolling barcode in a window
    
    Args:
        data: Binary data to transmit
        bar_height: Height of each bar in pixels
        fps: Frames per second
        screen_width: Screen width
        screen_height: Screen height
        loop: Whether to loop the data continuously
    """
    # Encode data to bars
    bars = encode_data_to_bars(data)
    
    print(f"Streaming {len(data)} bytes ({len(bars)} bars)")
    print(f"Bar height: {bar_height}px")
    print(f"FPS: {fps}")
    print(f"Throughput: {len(data) * fps} bytes/sec = {len(data) * fps / 1024:.2f} KB/sec")
    print(f"Screen: {screen_width}x{screen_height}")
    print("\nPress 'q' to quit, 'f' for fullscreen, SPACE to pause")
    
    window_name = "Barcode Data Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    scroll_offset = 0
    frame_delay = int(1000 / fps)  # milliseconds per frame
    scroll_speed = bar_height  # Scroll by one bar height per frame
    
    paused = False
    fullscreen = False
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        if not paused:
            # Create frame
            frame = create_horizontal_barcode_frame(
                bars, bar_height=bar_height,
                screen_height=screen_height, screen_width=screen_width,
                scroll_offset=scroll_offset
            )
            
            # Add info overlay
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            info_text = f"FPS: {actual_fps:.1f} | Offset: {scroll_offset} | Bars: {len(bars)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            # Update scroll offset
            scroll_offset += scroll_speed
            frame_count += 1
        else:
            # Paused - just show the current frame
            cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('f'):  # f for fullscreen
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_NORMAL)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Encode a string
        text = " ".join(sys.argv[1:])
        data = text.encode('utf-8')
        print(f"Encoding: '{text}'")
    else:
        # Default test data
        data = b"Hello from the Matrix! This is continuous data streaming through colored barcodes. " * 5
        print("Using default test data")
    
    # Stream the data
    stream_data(
        data,
        bar_height=4,      # 4 pixels per bar
        fps=30,            # 30 frames per second
        screen_width=1920,
        screen_height=1080,
        loop=True
    )

