#!/usr/bin/env python3
"""
Scrolling barcode decoder - receive data from camera capturing colored bars
Reads horizontal scan lines to decode the scrolling barcode
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from collections import deque
import time


# 16 color palette (must match encoder)
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


# Sync pattern (must match encoder)
SYNC_PATTERN = [0, 1, 0, 1, 0, 1, 0, 1]


def color_to_bits_lab(color: Tuple[int, int, int]) -> int:
    """
    Convert BGR color to closest 4-bit value using LAB color space
    
    Args:
        color: BGR tuple from OpenCV
        
    Returns:
        Integer 0-15
    """
    # Convert BGR to LAB
    bgr_pixel = np.uint8([[color]])
    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0]
    
    min_dist = float('inf')
    best_match = 0
    
    for i, palette_color in enumerate(COLOR_PALETTE):
        # Convert palette color (RGB) to BGR then LAB
        palette_bgr = np.uint8([[[palette_color[2], palette_color[1], palette_color[0]]]])
        palette_lab = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2LAB)[0][0]
        
        # Calculate perceptual distance (weight L less)
        delta_L = (float(lab_pixel[0]) - float(palette_lab[0])) * 0.5
        delta_A = float(lab_pixel[1]) - float(palette_lab[1])
        delta_B = float(lab_pixel[2]) - float(palette_lab[2])
        
        dist = np.sqrt(delta_L**2 + delta_A**2 + delta_B**2)
        
        if dist < min_dist:
            min_dist = dist
            best_match = i
    
    return best_match


def sample_scan_line(frame: np.ndarray, y: int, num_samples: int = 100) -> List[int]:
    """
    Sample colors along a horizontal scan line
    
    Args:
        frame: Input frame (BGR)
        y: Y coordinate of scan line
        num_samples: Number of samples to take across the width
        
    Returns:
        List of decoded 4-bit values
    """
    h, w = frame.shape[:2]
    samples = []
    
    # Sample evenly across the width
    for i in range(num_samples):
        x = int((i + 0.5) * w / num_samples)
        if 0 <= x < w and 0 <= y < h:
            # Sample a small region for robustness (median of 5x5 area)
            y_start = max(0, y - 2)
            y_end = min(h, y + 3)
            x_start = max(0, x - 2)
            x_end = min(w, x + 3)
            
            region = frame[y_start:y_end, x_start:x_end]
            median_color = np.median(region.reshape(-1, 3), axis=0).astype(np.uint8)
            
            # Decode color to bits
            value = color_to_bits_lab(tuple(median_color))
            samples.append(value)
    
    return samples


def find_sync_pattern(samples: List[int]) -> Optional[int]:
    """
    Find the sync pattern in a list of samples
    
    Args:
        samples: List of decoded values
        
    Returns:
        Index where sync pattern starts, or None if not found
    """
    pattern_len = len(SYNC_PATTERN)
    
    for i in range(len(samples) - pattern_len + 1):
        # Check if pattern matches
        match = True
        for j in range(pattern_len):
            if samples[i + j] != SYNC_PATTERN[j]:
                match = False
                break
        
        if match:
            return i
    
    return None


def decode_bars_to_bytes(bars: List[int]) -> bytes:
    """
    Decode a sequence of bars (4-bit values) to bytes
    
    Args:
        bars: List of 4-bit values (0-15)
        
    Returns:
        Decoded bytes
    """
    result = bytearray()
    
    # Process pairs of bars to reconstruct bytes
    for i in range(0, len(bars) - 1, 2):
        high_nibble = bars[i] & 0x0F
        low_nibble = bars[i + 1] & 0x0F
        byte = (high_nibble << 4) | low_nibble
        result.append(byte)
    
    return bytes(result)


def decode_from_camera(camera_id: int = 0, num_scan_lines: int = 5, 
                      num_samples: int = 100, show_preview: bool = True):
    """
    Decode data from camera capturing the barcode stream
    
    Args:
        camera_id: Camera device ID
        num_scan_lines: Number of horizontal lines to scan
        num_samples: Number of samples per scan line
        show_preview: Show live preview window
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print("Camera opened")
    print(f"Scanning {num_scan_lines} horizontal lines with {num_samples} samples each")
    print("Press 'q' to quit, SPACE to capture frame")
    
    # Buffer to accumulate decoded data
    decoded_buffer = deque(maxlen=1000)  # Keep last 1000 bars
    bytes_received = 0
    start_time = time.time()
    
    window_name = "Barcode Decoder"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Sample multiple scan lines
        scan_y_positions = [h // (num_scan_lines + 1) * (i + 1) for i in range(num_scan_lines)]
        
        all_samples = []
        for scan_y in scan_y_positions:
            samples = sample_scan_line(frame, scan_y, num_samples)
            all_samples.append(samples)
            
            # Draw scan line on preview
            if show_preview:
                cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 0), 2)
        
        # Use the middle scan line as primary
        primary_samples = all_samples[len(all_samples) // 2]
        
        # Look for sync pattern
        sync_pos = find_sync_pattern(primary_samples)
        
        if sync_pos is not None:
            # Found sync! Extract data after sync pattern
            data_start = sync_pos + len(SYNC_PATTERN)
            
            if data_start < len(primary_samples):
                # Find next sync pattern (end of data)
                remaining = primary_samples[data_start:]
                next_sync = find_sync_pattern(remaining)
                
                if next_sync is not None and next_sync > 0:
                    # Extract data between sync patterns
                    data_bars = remaining[:next_sync]
                    decoded_buffer.extend(data_bars)
                    
                    # Try to decode accumulated data
                    if len(decoded_buffer) >= 2:  # Need at least 2 bars for 1 byte
                        bars_list = list(decoded_buffer)
                        decoded_bytes = decode_bars_to_bytes(bars_list)
                        
                        if decoded_bytes:
                            bytes_received += len(decoded_bytes)
                            elapsed = time.time() - start_time
                            throughput = bytes_received / elapsed if elapsed > 0 else 0
                            
                            # Try to decode as text
                            try:
                                text = decoded_bytes.decode('utf-8', errors='ignore')
                                print(f"\rReceived: {text[:50]}... | {bytes_received} bytes | {throughput:.1f} B/s", end='')
                            except:
                                print(f"\rReceived: {len(decoded_bytes)} bytes | {bytes_received} total | {throughput:.1f} B/s", end='')
        
        # Show preview with visualization
        if show_preview:
            # Create visualization of detected colors
            vis_height = 100
            vis = np.zeros((vis_height, len(primary_samples) * 10, 3), dtype=np.uint8)
            
            for i, value in enumerate(primary_samples):
                color = COLOR_PALETTE[value]
                # BGR for OpenCV
                bgr_color = (color[2], color[1], color[0])
                cv2.rectangle(vis, (i * 10, 0), ((i + 1) * 10, vis_height), bgr_color, -1)
            
            # Add visualization to bottom of frame
            frame_resized = cv2.resize(frame, (vis.shape[1], frame.shape[0]))
            display = np.vstack([frame_resized, vis])
            
            # Add info text
            info_text = f"Bars: {len(decoded_buffer)} | Bytes: {bytes_received}"
            if sync_pos is not None:
                info_text += " | SYNC FOUND!"
            cv2.putText(display, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            # Capture and save current state
            print(f"\nCaptured {len(decoded_buffer)} bars")
            if decoded_buffer:
                bars_list = list(decoded_buffer)
                final_bytes = decode_bars_to_bytes(bars_list)
                print(f"Decoded to {len(final_bytes)} bytes")
                print(f"Data (first 200 chars): {final_bytes[:200]}")
                
                # Save to file
                with open("barcode_decoded.bin", "wb") as f:
                    f.write(final_bytes)
                print("Saved to barcode_decoded.bin")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final decode
    if decoded_buffer:
        bars_list = list(decoded_buffer)
        final_bytes = decode_bars_to_bytes(bars_list)
        print(f"\n\nFinal decode: {len(final_bytes)} bytes")
        try:
            text = final_bytes.decode('utf-8', errors='ignore')
            print(f"Text: {text}")
        except:
            print(f"Hex: {final_bytes.hex()}")


if __name__ == "__main__":
    import sys
    
    camera_id = 0
    if len(sys.argv) > 1:
        camera_id = int(sys.argv[1])
    
    decode_from_camera(
        camera_id=camera_id,
        num_scan_lines=5,
        num_samples=100,
        show_preview=True
    )

