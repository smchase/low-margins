"""Camera utilities for 5-color codec system."""

import numpy as np
import cv2
from config import COLORS_5, COLOR_NAMES


def rgb_to_color_index(rgb_array):
    """
    Map RGB values to the nearest color index in the 5-color palette.
    
    Args:
        rgb_array: numpy array of shape (height, width, 3) in BGR format
        
    Returns:
        numpy array of shape (height, width) with color indices [0-4]
    """
    if len(rgb_array.shape) == 1:
        # Single pixel
        distances = np.linalg.norm(COLORS_5 - rgb_array, axis=1)
        return np.argmin(distances)
    elif len(rgb_array.shape) == 3:
        # Image array
        h, w = rgb_array.shape[:2]
        flat_pixels = rgb_array.reshape(-1, 3)
        
        # Compute distances to all colors for all pixels
        distances = np.zeros((flat_pixels.shape[0], len(COLORS_5)))
        for i, color in enumerate(COLORS_5):
            distances[:, i] = np.linalg.norm(flat_pixels - color, axis=1)
        
        # Find nearest color for each pixel
        indices = np.argmin(distances, axis=1)
        return indices.reshape(h, w)
    else:
        raise ValueError(f"Expected 1D or 3D array, got shape {rgb_array.shape}")


def color_index_to_rgb(indices):
    """
    Convert color indices to RGB values.
    
    Args:
        indices: numpy array of color indices [0-4]
        
    Returns:
        numpy array of RGB values in BGR format
    """
    if isinstance(indices, (int, np.integer)):
        return COLORS_5[indices]
    else:
        return COLORS_5[indices]


def detect_grid_colors(image, grid_size, margin=0.2):
    """
    Detect colors in a grid pattern from a camera image.
    
    Args:
        image: numpy array (height, width, 3) in BGR format
        grid_size: number of grid cells (assumes square grid)
        margin: fraction of cell to ignore at edges (default: 0.2)
        
    Returns:
        numpy array of shape (grid_size, grid_size) with color indices [0-4]
    """
    h, w = image.shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate cell bounds with margin
            y1 = int((row + margin) * cell_h)
            y2 = int((row + 1 - margin) * cell_h)
            x1 = int((col + margin) * cell_w)
            x2 = int((col + 1 - margin) * cell_w)
            
            # Extract cell center region
            cell_region = image[y1:y2, x1:x2]
            
            # Average color in the region
            mean_color = cell_region.mean(axis=(0, 1))
            
            # Map to nearest color
            grid[row, col] = rgb_to_color_index(mean_color)
    
    return grid


def draw_color_grid(image, grid, cell_size=30, alpha=0.7):
    """
    Draw a colored grid overlay on an image.
    
    Args:
        image: numpy array (height, width, 3) in BGR format
        grid: numpy array of shape (rows, cols) with color indices [0-4]
        cell_size: pixels per cell (default: 30)
        alpha: overlay transparency (0=transparent, 1=opaque)
        
    Returns:
        numpy array with grid overlay applied
    """
    rows, cols = grid.shape
    overlay = image.copy()
    
    for row in range(rows):
        for col in range(cols):
            color_idx = grid[row, col]
            color = COLORS_5[color_idx].tolist()
            
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend with original image
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result


def visualize_color_palette():
    """Display the 5-color palette for reference."""
    palette_img = np.zeros((100, 500, 3), dtype=np.uint8)
    
    for i, (color, name) in enumerate(zip(COLORS_5, COLOR_NAMES)):
        x1 = i * 100
        x2 = (i + 1) * 100
        palette_img[:, x1:x2] = color
        
        # Add text label
        text_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
        cv2.putText(palette_img, f"{i}:{name}", (x1 + 10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    return palette_img


if __name__ == "__main__":
    # Test color detection
    print("5-Color Palette:")
    for i, (color, name) in enumerate(zip(COLORS_5, COLOR_NAMES)):
        print(f"  {i}: {name} - BGR{tuple(color)}")
    
    # Show palette
    palette = visualize_color_palette()
    cv2.imshow("5-Color Palette", palette)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

