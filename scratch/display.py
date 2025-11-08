#!/usr/bin/env python3
"""
Display encoded image fullscreen for camera capture
Press ESC or 'q' to quit
"""
import cv2
import sys
import os


def display_fullscreen(image_path: str):
    """
    Display an image in fullscreen mode for camera capture
    
    Args:
        image_path: Path to the image file
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    window_name = "Data Transmission - Press ESC or Q to quit"
    
    # Create fullscreen window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print(f"Displaying {image_path} in fullscreen")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    print("Press ESC or 'q' to quit")
    print("\nTip: Point the camera at this screen and press SPACE in decode.py to capture")
    
    # Display the image
    cv2.imshow(window_name, img)
    
    # Wait for key press
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python display.py <image_file>")
        print("\nExample:")
        print("  python display.py encoded_data.png")
        print("\nThis will display the image fullscreen for camera capture.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    display_fullscreen(image_path)

