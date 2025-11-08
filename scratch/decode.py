"""
Decode binary data from a 64x64 color grid image using OpenCV
Detects the grid pattern and extracts colors to reconstruct the original data
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional


# 16 color palette (must match encode.py)
COLOR_PALETTE = [
    (230, 25, 75),    # Bright Red - 0000 #E6194B
    (255, 255, 255),  # White - 0001
    (245, 130, 49),   # Orange - 0010 #F58231
    (154, 99, 36),    # Brown - 0011 #9A6324
    (255, 225, 25),   # Yellow - 0100 #FFE119
    (128, 128, 0),    # Olive - 0101 #808000
    (60, 180, 75),    # Lime Green - 0110 #3CB44B
    (34, 139, 34),    # Dark Green - 0111 #228B22
    (0, 255, 127),    # Spring Green - 1000 #00FF7F
    (255, 69, 0),     # Red Orange - 1001 #FF4500
    (135, 206, 235),  # Sky Blue - 1010 #87CEEB
    (255, 215, 0),    # Gold - 1011 #FFD700
    (0, 0, 117),      # Navy - 1100 #000075
    (145, 30, 180),   # Purple - 1101 #911EB4
    (240, 50, 230),   # Magenta - 1110 #F032E6
    (250, 190, 212),  # Pink - 1111 #FABED4
]


def color_to_bits(color: Tuple[int, int, int]) -> int:
    """
    Convert an RGB color to the closest matching 4-bit value using LAB color space
    
    Args:
        color: RGB tuple (B, G, R) for OpenCV
        
    Returns:
        Integer from 0-15 representing the 4-bit value
    """
    # Convert BGR to LAB for better perceptual color matching
    # OpenCV uses BGR, so we need a 1x1 image to convert
    bgr_pixel = np.uint8([[color]])
    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0]
    
    min_dist = float('inf')
    best_match = 0
    
    for i, palette_color in enumerate(COLOR_PALETTE):
        # Convert palette color (RGB) to BGR then to LAB
        palette_bgr = np.uint8([[[palette_color[2], palette_color[1], palette_color[0]]]])
        palette_lab = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2LAB)[0][0]
        
        # Calculate Delta E (CIE76) - perceptually uniform color difference
        # Weight L channel less since screens vary in brightness more
        delta_L = (float(lab_pixel[0]) - float(palette_lab[0])) * 0.5
        delta_A = float(lab_pixel[1]) - float(palette_lab[1])
        delta_B = float(lab_pixel[2]) - float(palette_lab[2])
        
        dist = np.sqrt(delta_L**2 + delta_A**2 + delta_B**2)
        
        if dist < min_dist:
            min_dist = dist
            best_match = i
    
    return best_match


def detect_grid_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the corners of the grid in the image
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Array of 4 corner points or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find black grid lines
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest rectangular contour (should be the grid)
    if not contours:
        return None
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Approximate the contour to a polygon
    for contour in contours[:5]:  # Check top 5 largest contours
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we found a quadrilateral
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    return None


def extract_grid_cells(image: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """
    Extract individual cells from the grid image
    
    Args:
        image: Input image (BGR format)
        grid_size: Expected grid size (64x64)
        
    Returns:
        Array of shape (grid_size, grid_size, 3) with average color of each cell
    """
    # Detect grid corners
    corners = detect_grid_corners(image)
    
    if corners is None:
        # If corner detection fails, assume the image is already the grid
        # and try to extract cells directly
        h, w = image.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        cells = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        for row in range(grid_size):
            for col in range(grid_size):
                # Extract cell region - take center 50% to avoid grid lines
                y_center = int((row + 0.5) * cell_h)
                x_center = int((col + 0.5) * cell_w)
                
                # Sample a region around the center (center 50% of cell)
                sample_h = max(cell_h // 2, 2)
                sample_w = max(cell_w // 2, 2)
                
                y_start = max(0, y_center - sample_h // 2)
                y_end = min(h, y_center + sample_h // 2)
                x_start = max(0, x_center - sample_w // 2)
                x_end = min(w, x_center + sample_w // 2)
                
                if y_end > y_start and x_end > x_start:
                    cell_region = image[y_start:y_end, x_start:x_end]
                    
                    # Use median instead of mean for more robustness
                    median_color = np.median(cell_region.reshape(-1, 3), axis=0)
                    cells[row, col] = median_color.astype(np.uint8)
                else:
                    cells[row, col] = [255, 255, 255]  # Default to white
        
        return cells
    
    # Perspective transform to get a top-down view
    # Order corners: top-left, top-right, bottom-right, bottom-left
    src_points = corners.astype(np.float32)
    
    # Calculate the bounding box
    x_coords = src_points[:, 0]
    y_coords = src_points[:, 1]
    
    width = int(max(x_coords) - min(x_coords))
    height = int(max(y_coords) - min(y_coords))
    
    # Destination points for perspective transform
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Warp the image
    warped = cv2.warpPerspective(image, M, (width, height))
    
    # Extract cells from warped image
    cell_h = height // grid_size
    cell_w = width // grid_size
    
    cells = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Extract cell region - take center 50% to avoid grid lines
            y_center = int((row + 0.5) * cell_h)
            x_center = int((col + 0.5) * cell_w)
            
            # Sample a region around the center (center 50% of cell)
            sample_h = max(cell_h // 2, 2)
            sample_w = max(cell_w // 2, 2)
            
            y_start = max(0, y_center - sample_h // 2)
            y_end = min(height, y_center + sample_h // 2)
            x_start = max(0, x_center - sample_w // 2)
            x_end = min(width, x_center + sample_w // 2)
            
            if y_end > y_start and x_end > x_start:
                cell_region = warped[y_start:y_end, x_start:x_end]
                
                # Use median instead of mean for more robustness
                median_color = np.median(cell_region.reshape(-1, 3), axis=0)
                cells[row, col] = median_color.astype(np.uint8)
            else:
                cells[row, col] = [255, 255, 255]  # Default to white
    
    return cells


def decode_grid_to_bits(cells: np.ndarray) -> List[int]:
    """
    Convert grid cells to bits
    
    Args:
        cells: Array of shape (grid_size, grid_size, 3) with cell colors
        
    Returns:
        List of bits (0s and 1s)
    """
    bits = []
    grid_size = cells.shape[0]
    
    # Process cells row by row, left to right
    for row in range(grid_size):
        for col in range(grid_size):
            # Get cell color (BGR format from OpenCV)
            cell_color = tuple(cells[row, col])
            
            # Convert to 4-bit value
            color_value = color_to_bits(cell_color)
            
            # Convert 4-bit value to bits (most significant first)
            bits.append((color_value >> 3) & 1)
            bits.append((color_value >> 2) & 1)
            bits.append((color_value >> 1) & 1)
            bits.append(color_value & 1)
    
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Convert list of bits to bytes
    
    Args:
        bits: List of bits (0s and 1s)
        
    Returns:
        bytes object
    """
    # Pad to multiple of 8
    while len(bits) % 8 != 0:
        bits.append(0)
    
    # Pack bits into bytes
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    
    return bytes(result)


def decode_image(image_path: str, grid_size: int = 64) -> bytes:
    """
    Decode data from an image file
    
    Args:
        image_path: Path to the encoded image
        grid_size: Expected grid size (64x64)
        
    Returns:
        Decoded bytes
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Extract grid cells
    cells = extract_grid_cells(image, grid_size)
    
    # Convert cells to bits
    bits = decode_grid_to_bits(cells)
    
    # Convert bits to bytes
    data = bits_to_bytes(bits)
    
    return data


def visualize_grid_detection(image: np.ndarray, cells: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """
    Create a visualization of the detected grid cells
    
    Args:
        image: Original image
        cells: Detected cells array
        grid_size: Grid size
        
    Returns:
        Visualization image
    """
    # Create a visualization showing the detected colors
    vis_size = 512
    cell_size = vis_size // grid_size
    vis = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Get detected color (convert from BGR to RGB for display)
            color = cells[row, col]
            # Draw cell
            y_start = row * cell_size
            y_end = (row + 1) * cell_size
            x_start = col * cell_size
            x_end = (col + 1) * cell_size
            vis[y_start:y_end, x_start:x_end] = color
    
    return vis


def decode_from_camera(grid_size: int = 64, camera_id: int = 0, show_live_detection: bool = True) -> Optional[bytes]:
    """
    Decode data from camera feed with live preview
    
    Args:
        grid_size: Expected grid size (64x64)
        camera_id: Camera device ID
        show_live_detection: Show live grid detection overlay
        
    Returns:
        Decoded bytes or None if cancelled
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None
    
    print("Camera opened. Press SPACE to capture and decode, 'q' to quit")
    print("Live grid detection preview enabled")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # Show live detection overlay
        if show_live_detection:
            try:
                # Quick extract cells for preview (lower resolution for speed)
                preview_grid_size = min(grid_size, 32)  # Use 32x32 for fast preview
                cells = extract_grid_cells(frame, preview_grid_size)
                
                # Create small visualization in corner
                vis_size = 200
                cell_size = vis_size // preview_grid_size
                vis = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
                
                for row in range(preview_grid_size):
                    for col in range(preview_grid_size):
                        color = cells[row, col]
                        y_start = row * cell_size
                        y_end = (row + 1) * cell_size
                        x_start = col * cell_size
                        x_end = (col + 1) * cell_size
                        vis[y_start:y_end, x_start:x_end] = color
                
                # Overlay in top-right corner
                h, w = display_frame.shape[:2]
                x_offset = w - vis_size - 10
                y_offset = 10
                
                # Add black border
                display_frame[y_offset-2:y_offset+vis_size+2, x_offset-2:x_offset+vis_size+2] = [0, 0, 0]
                display_frame[y_offset:y_offset+vis_size, x_offset:x_offset+vis_size] = vis
                
                # Add text
                cv2.putText(display_frame, "Live Preview", (x_offset, y_offset-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except:
                pass  # Silently fail if detection doesn't work in preview
        
        # Show the frame
        cv2.imshow('Camera Feed - Press SPACE to decode, Q to quit', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to capture
            print("Capturing and decoding...")
            try:
                # Extract grid cells at full resolution
                cells = extract_grid_cells(frame, grid_size)
                
                # Show visualization
                vis = visualize_grid_detection(frame, cells, grid_size)
                cv2.imshow('Detected Grid', vis)
                cv2.waitKey(2000)  # Show for 2 seconds
                
                # Convert cells to bits
                bits = decode_grid_to_bits(cells)
                
                # Convert bits to bytes
                data = bits_to_bytes(bits)
                
                print(f"Decoded {len(data)} bytes")
                return data
            except Exception as e:
                print(f"Error during decoding: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    cap.release()
    cv2.destroyAllWindows()
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Decode from image file
        image_path = sys.argv[1]
        print(f"Decoding image: {image_path}")
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Extract grid cells
            cells = extract_grid_cells(image, grid_size=64)
            
            # Show visualization
            vis = visualize_grid_detection(image, cells, grid_size=64)
            cv2.imshow('Detected Grid', vis)
            print("Press any key to continue after viewing the detected grid...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Convert cells to bits
            bits = decode_grid_to_bits(cells)
            
            # Convert bits to bytes
            data = bits_to_bytes(bits)
            
            print(f"Decoded {len(data)} bytes")
            print(f"First 100 bytes (hex): {data[:100].hex()}")
            
            # Save to file
            output_path = "decoded_data.bin"
            with open(output_path, 'wb') as f:
                f.write(data)
            print(f"Saved decoded data to {output_path}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Decode from camera
        print("No image file provided. Opening camera...")
        data = decode_from_camera(grid_size=64)
        if data:
            print(f"Decoded {len(data)} bytes")
            print(f"First 100 bytes (hex): {data[:100].hex()}")
            
            # Save to file
            output_path = "decoded_data.bin"
            with open(output_path, 'wb') as f:
                f.write(data)
            print(f"Saved decoded data to {output_path}")

