"""
Encode binary data into a 128x128 grid with 16 colors
Each color represents 4 bits, colors are diverse for easy detection
"""
import numpy as np
from PIL import Image, ImageDraw


# 16 highly distinct colors for easy differentiation (white is allowed, avoiding black)
# Colors sorted so similar colors are adjacent for easier visual grouping
# Each color maps to a 4-bit sequence (0-15)
COLOR_PALETTE = [
    (230, 25, 75),    # Bright Red - 0000 #E6194B
    (255, 255, 255),  # White - 0001 (replaces coral)
    (245, 130, 49),   # Orange - 0010 #F58231
    (154, 99, 36),    # Brown - 0011 #9A6324 (earth tone, goes with oranges)
    (255, 225, 25),   # Yellow - 0100 #FFE119
    (128, 128, 0),    # Olive - 0101 #808000 (yellow-green)
    (60, 180, 75),    # Lime Green - 0110 #3CB44B
    (34, 139, 34),    # Dark Green - 0111 #228B22
    (0, 255, 127),    # Spring Green - 1000 #00FF7F (bright green-cyan, distinct from other greens)
    (255, 69, 0),     # Red Orange - 1001 #FF4500 (distinct from orange, more red)
    (135, 206, 235),  # Sky Blue - 1010 #87CEEB (light blue)
    (255, 215, 0),    # Gold - 1011 #FFD700 (bright gold, distinct from yellow)
    (0, 0, 117),      # Navy - 1100 #000075 (dark blue)
    (145, 30, 180),   # Purple - 1101 #911EB4
    (240, 50, 230),   # Magenta - 1110 #F032E6 (purple-pink)
    (250, 190, 212),  # Pink - 1111 #FABED4 (light pink)
]


def bits_to_color(bits: int) -> tuple:
    """
    Convert 4 bits to a color from the palette
    
    Args:
        bits: Integer from 0-15 representing 4 bits
        
    Returns:
        RGB tuple
    """
    if not 0 <= bits <= 15:
        raise ValueError(f"Bits must be between 0 and 15, got {bits}")
    return COLOR_PALETTE[bits]


def encode_bits_to_grid(data: bytes, grid_size: int = 128) -> np.ndarray:
    """
    Encode binary data into a grid where each cell represents 4 bits
    
    Args:
        data: Binary data to encode
        grid_size: Size of the grid (grid_size x grid_size)
        
    Returns:
        numpy array of shape (grid_size, grid_size, 3) with RGB colors
    """
    # Calculate total bits
    total_bits = len(data) * 8
    
    # Calculate cells needed (each cell stores 4 bits)
    cells_needed = (total_bits + 3) // 4  # Ceiling division
    
    # Total cells available
    total_cells = grid_size * grid_size
    
    if cells_needed > total_cells:
        raise ValueError(
            f"Data too large: need {cells_needed} cells, "
            f"but grid only has {total_cells} cells"
        )
    
    # Create empty grid
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    # Convert data to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    
    # Pad bits to multiple of 4
    while len(bits) % 4 != 0:
        bits.append(0)
    
    # Fill grid with colors
    cell_idx = 0
    for i in range(0, len(bits), 4):
        # Get 4 bits
        four_bits = bits[i:i+4]
        # Convert to integer (0-15)
        value = (four_bits[0] << 3) | (four_bits[1] << 2) | (four_bits[2] << 1) | four_bits[3]
        
        # Get row and column
        row = cell_idx // grid_size
        col = cell_idx % grid_size
        
        if row < grid_size:
            # Set color in grid
            grid[row, col] = bits_to_color(value)
        
        cell_idx += 1
    
    return grid


def create_pairwise_color_test_pattern(grid_size: int = 64) -> bytes:
    """
    Create a bitstring that shows all pairwise combinations of colors.
    Each pair is shown as two adjacent columns (one for each color).
    
    With 16 colors, there are 16 * 15 / 2 = 120 unordered pairs.
    Each pair takes 2 columns, so we need 240 columns (but grid is 64x64).
    So we'll show pairs across multiple rows, wrapping as needed.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        
    Returns:
        bytes: Bitstring that encodes the pattern
    """
    num_colors = 16
    # Generate all unordered pairs (i, j) where i < j
    pairs = []
    for i in range(num_colors):
        for j in range(i + 1, num_colors):
            pairs.append((i, j))
    
    # Create list to store bits
    bits = []
    
    # Fill grid row by row, left to right
    pair_idx = 0
    for row in range(grid_size):
        for col in range(0, grid_size, 2):  # Step by 2 since each pair takes 2 columns
            if pair_idx < len(pairs):
                color1, color2 = pairs[pair_idx]
                # First column gets color1, second column gets color2
                for offset in range(2):
                    if col + offset < grid_size:
                        color_value = color1 if offset == 0 else color2
                        
                        # Convert color value (0-15) to 4 bits
                        b3 = (color_value >> 3) & 1
                        b2 = (color_value >> 2) & 1
                        b1 = (color_value >> 1) & 1
                        b0 = color_value & 1
                        
                        # Add bits in order (most significant first)
                        bits.extend([b3, b2, b1, b0])
                pair_idx += 1
            else:
                # Fill remaining cells with color 0
                for offset in range(2):
                    if col + offset < grid_size:
                        bits.extend([0, 0, 0, 0])
    
    # Convert bits to bytes
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


def create_color_test_pattern(grid_size: int = 64, columns_per_color: int = 4) -> bytes:
    """
    Create a bitstring that produces a grid with columns_per_color columns for each color.
    
    For a 64x64 grid with 4 columns per color:
    - Color 0: columns 0-3
    - Color 1: columns 4-7
    - ...
    - Color 15: columns 60-63
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        columns_per_color: Number of columns for each color (default 4)
        
    Returns:
        bytes: Bitstring that encodes the pattern
    """
    num_colors = 16
    total_columns = num_colors * columns_per_color
    
    if total_columns != grid_size:
        raise ValueError(
            f"Grid size {grid_size} must equal num_colors * columns_per_color = {total_columns}"
        )
    
    # Create list to store bits
    bits = []
    
    # Fill grid row by row, left to right
    for row in range(grid_size):
        for col in range(grid_size):
            # Determine which color this column belongs to
            color_value = col // columns_per_color
            
            # Convert color value (0-15) to 4 bits
            # Bit pattern: b3 b2 b1 b0 (most significant to least)
            b3 = (color_value >> 3) & 1
            b2 = (color_value >> 2) & 1
            b1 = (color_value >> 1) & 1
            b0 = color_value & 1
            
            # Add bits in order (most significant first)
            bits.extend([b3, b2, b1, b0])
    
    # Convert bits to bytes
    # Pad to multiple of 8 if needed
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


def encode_to_image(data: bytes, grid_size: int = 128, add_grid: bool = True, 
                    grid_line_width: int = 1, cell_size: int = 4) -> Image.Image:
    """
    Encode binary data into an image with optional grid overlay
    
    Args:
        data: Binary data to encode
        grid_size: Number of cells per side (default 128)
        add_grid: Whether to add black grid lines
        grid_line_width: Width of grid lines in pixels
        cell_size: Size of each cell in pixels (default 4, must be >= 2 to show grid)
        
    Returns:
        PIL Image
    """
    # Encode to grid
    grid = encode_bits_to_grid(data, grid_size)
    
    # Create image with proper sizing for grid
    # Each cell will be cell_size pixels, plus grid lines
    # Total width = (grid_size * cell_size) + ((grid_size + 1) * grid_line_width)
    height, width = grid.shape[:2]
    
    # Calculate image dimensions accounting for grid lines
    if add_grid:
        img_width = (width * cell_size) + ((width + 1) * grid_line_width)
        img_height = (height * cell_size) + ((height + 1) * grid_line_width)
    else:
        img_width = width * cell_size
        img_height = height * cell_size
    
    # Create blank image with white background
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    
    # Draw cells
    draw = ImageDraw.Draw(img)
    for row in range(height):
        for col in range(width):
            # Calculate pixel position accounting for grid lines
            if add_grid:
                # First grid line, then cell, then grid line, etc.
                x_start = grid_line_width + col * (cell_size + grid_line_width)
                y_start = grid_line_width + row * (cell_size + grid_line_width)
            else:
                x_start = col * cell_size
                y_start = row * cell_size
            
            x_end = x_start + cell_size
            y_end = y_start + cell_size
            
            # Get color for this cell
            color = tuple(grid[row, col])
            
            # Draw rectangle for this cell
            draw.rectangle([x_start, y_start, x_end - 1, y_end - 1], fill=color)
    
    # Draw black grid lines on top
    if add_grid:
        # Draw vertical lines
        for i in range(grid_size + 1):
            x = i * (cell_size + grid_line_width)
            if x < img_width:
                draw.rectangle([x, 0, min(x + grid_line_width, img_width) - 1, img_height - 1], fill=(0, 0, 0))
        
        # Draw horizontal lines
        for i in range(grid_size + 1):
            y = i * (cell_size + grid_line_width)
            if y < img_height:
                draw.rectangle([0, y, img_width - 1, min(y + grid_line_width, img_height) - 1], fill=(0, 0, 0))
    
    return img


if __name__ == "__main__":
    import sys
    
    grid_size = 64
    
    # Determine mode based on arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "pairwise":
            # Create pairwise test pattern
            print(f"Creating pairwise color test pattern: {grid_size}x{grid_size} grid")
            print(f"Showing all {16 * 15 // 2} pairwise combinations of 16 colors")
            test_data = create_pairwise_color_test_pattern(grid_size=grid_size)
            output_file = "encoded_data_pairwise.png"
            
            # Encode to image with visible grid
            img = encode_to_image(test_data, grid_size=grid_size, add_grid=True, cell_size=4, grid_line_width=1)
            
            # Save image
            img.save(output_file)
            print(f"Generated bitstring of length {len(test_data)} bytes ({len(test_data)*8} bits)")
            print(f"Image saved to {output_file}")
            print(f"Image size: {img.size}")
            print(f"\nPattern: Each pair of adjacent columns shows a color pair (color1, color2)")
            print(f"Pairs are shown left to right, top to bottom")
            
            # Show the image
            img.show()
        elif sys.argv[1] == "columns":
            # Create test pattern with 4 columns per color
            columns_per_color = 4
            
            print(f"Creating test pattern: {grid_size}x{grid_size} grid with {columns_per_color} columns per color")
            test_data = create_color_test_pattern(grid_size=grid_size, columns_per_color=columns_per_color)
            output_file = "encoded_data.png"
            
            # Encode to image with visible grid
            img = encode_to_image(test_data, grid_size=grid_size, add_grid=True, cell_size=4, grid_line_width=1)
            
            # Save image
            img.save(output_file)
            print(f"Generated bitstring of length {len(test_data)} bytes ({len(test_data)*8} bits)")
            print(f"Expected: {grid_size * grid_size * 4 / 8} bytes for {grid_size * grid_size} cells")
            print(f"Image saved to {output_file}")
            print(f"Image size: {img.size}")
            print(f"Total cells: {grid_size}x{grid_size} = {grid_size*grid_size}")
            print(f"Data capacity: {grid_size*grid_size*4/8} bytes ({grid_size*grid_size*4} bits)")
            print(f"\nColor pattern (similar colors grouped together):")
            color_names = [
                "Bright Red", "White", "Orange", "Brown",
                "Yellow", "Olive", "Lime Green", "Dark Green",
                "Spring Green", "Red Orange", "Sky Blue", "Gold",
                "Navy", "Purple", "Magenta", "Pink"
            ]
            for i in range(16):
                start_col = i * columns_per_color
                end_col = start_col + columns_per_color - 1
                rgb = COLOR_PALETTE[i]
                print(f"  - Color {i} ({color_names[i]}): columns {start_col}-{end_col} RGB{rgb}")
            
            # Show the image
            img.show()
        else:
            # Encode string provided as argument
            # Join all arguments in case string has spaces (user should quote it, but we handle both)
            if len(sys.argv) > 2:
                text = " ".join(sys.argv[1:])
            else:
                text = sys.argv[1]
            print(f"Encoding string: '{text}'")
            
            # Convert string to bytes (UTF-8 encoding)
            text_bytes = text.encode('utf-8')
            print(f"String length: {len(text)} characters")
            print(f"Byte length: {len(text_bytes)} bytes")
            
            # Calculate required grid size
            bits_needed = len(text_bytes) * 8
            cells_needed = (bits_needed + 3) // 4  # Ceiling division
            min_grid_size = int(np.ceil(np.sqrt(cells_needed)))
            
            # Use the specified grid size, or auto-calculate if needed
            if min_grid_size > grid_size:
                print(f"Warning: Data requires {cells_needed} cells, but {grid_size}x{grid_size} grid only has {grid_size*grid_size} cells")
                print(f"Minimum grid size needed: {min_grid_size}x{min_grid_size}")
                print(f"Using {min_grid_size}x{min_grid_size} grid instead")
                grid_size = min_grid_size
            
            # Encode to image with visible grid
            img = encode_to_image(text_bytes, grid_size=grid_size, add_grid=True, cell_size=4, grid_line_width=1)
            
            # Generate output filename
            safe_text = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in text[:30]).strip('_')
            output_file = f"encoded_{safe_text}.png" if safe_text else "encoded_string.png"
            # Limit filename length
            if len(output_file) > 100:
                output_file = f"encoded_{hash(text) % 10000}.png"
            
            # Save image
            img.save(output_file)
            print(f"Image saved to {output_file}")
            print(f"Image size: {img.size}")
            print(f"Grid size: {grid_size}x{grid_size}")
            print(f"Data capacity: {grid_size*grid_size*4/8} bytes ({grid_size*grid_size*4} bits)")
            print(f"Data used: {len(text_bytes)} bytes ({bits_needed} bits)")
            
            # Show the image
            img.show()
    else:
        # Default: Create test pattern with 4 columns per color
        columns_per_color = 4
        
        print(f"Creating test pattern: {grid_size}x{grid_size} grid with {columns_per_color} columns per color")
        print(f"\nUsage:")
        print(f"  python {sys.argv[0]} <string>          - Encode a string")
        print(f"  python {sys.argv[0]} pairwise          - Create pairwise test pattern")
        print(f"  python {sys.argv[0]} columns           - Create column test pattern")
        print()
        
        test_data = create_color_test_pattern(grid_size=grid_size, columns_per_color=columns_per_color)
        output_file = "encoded_data.png"
        
        # Encode to image with visible grid
        img = encode_to_image(test_data, grid_size=grid_size, add_grid=True, cell_size=4, grid_line_width=1)
        
        # Save image
        img.save(output_file)
        print(f"Generated bitstring of length {len(test_data)} bytes ({len(test_data)*8} bits)")
        print(f"Expected: {grid_size * grid_size * 4 / 8} bytes for {grid_size * grid_size} cells")
        print(f"Image saved to {output_file}")
        print(f"Image size: {img.size}")
        print(f"Total cells: {grid_size}x{grid_size} = {grid_size*grid_size}")
        print(f"Data capacity: {grid_size*grid_size*4/8} bytes ({grid_size*grid_size*4} bits)")
        print(f"\nColor pattern (similar colors grouped together):")
        color_names = [
            "Bright Red", "White", "Orange", "Brown",
            "Yellow", "Olive", "Lime Green", "Dark Green",
            "Spring Green", "Red Orange", "Sky Blue", "Gold",
            "Navy", "Purple", "Magenta", "Pink"
        ]
        for i in range(16):
            start_col = i * columns_per_color
            end_col = start_col + columns_per_color - 1
            rgb = COLOR_PALETTE[i]
            print(f"  - Color {i} ({color_names[i]}): columns {start_col}-{end_col} RGB{rgb}")
        
        # Show the image
        img.show()

