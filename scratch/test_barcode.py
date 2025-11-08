#!/usr/bin/env python3
"""
Test the barcode encoding/decoding without needing two laptops
Simulates the full pipeline
"""
import numpy as np
from barcode_encode import encode_data_to_bars, create_horizontal_barcode_frame, bits_to_color
from barcode_decode import sample_scan_line, find_sync_pattern, decode_bars_to_bytes


def test_encode_decode():
    """Test that we can encode and decode data correctly"""
    # Test data
    test_string = "Hello, Matrix! 🌊 This is a test of the barcode system."
    test_data = test_string.encode('utf-8')
    
    print(f"Original: '{test_string}'")
    print(f"Bytes: {len(test_data)}")
    
    # Encode to bars
    bars = encode_data_to_bars(test_data)
    print(f"Encoded to {len(bars)} bars")
    
    # Create a frame
    frame = create_horizontal_barcode_frame(
        bars, 
        bar_height=20,  # Large bars for easy testing
        screen_height=1080,
        screen_width=1920,
        scroll_offset=0
    )
    print(f"Frame shape: {frame.shape}")
    
    # Sample the frame (simulate camera capture)
    scan_y = frame.shape[0] // 2  # Middle of screen
    samples = sample_scan_line(frame, scan_y, num_samples=len(bars))
    print(f"Sampled {len(samples)} values")
    
    # Find sync pattern
    sync_pos = find_sync_pattern(samples)
    if sync_pos is not None:
        print(f"✓ Sync pattern found at position {sync_pos}")
        
        # Extract data
        data_start = sync_pos + 8  # Length of sync pattern
        data_samples = samples[data_start:]
        
        # Find end sync
        end_sync = find_sync_pattern(data_samples)
        if end_sync is not None:
            print(f"✓ End sync found at position {end_sync}")
            data_bars = data_samples[:end_sync]
            
            # Decode
            decoded_bytes = decode_bars_to_bytes(data_bars)
            decoded_string = decoded_bytes.decode('utf-8', errors='ignore')
            
            print(f"\nDecoded: '{decoded_string}'")
            print(f"Decoded bytes: {len(decoded_bytes)}")
            
            # Check if it matches
            if decoded_string == test_string:
                print("\n✓✓✓ SUCCESS! Encoding and decoding works perfectly!")
                return True
            else:
                print(f"\n✗ MISMATCH!")
                print(f"Expected: '{test_string}'")
                print(f"Got:      '{decoded_string}'")
                return False
        else:
            print("✗ End sync pattern not found")
            return False
    else:
        print("✗ Sync pattern not found")
        return False


def test_color_accuracy():
    """Test color encoding/decoding accuracy"""
    print("\n" + "="*60)
    print("Testing color accuracy")
    print("="*60)
    
    from barcode_decode import color_to_bits_lab
    
    errors = 0
    for i in range(16):
        # Get the color
        rgb_color = bits_to_color(i)
        # Convert to BGR for OpenCV
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        
        # Decode it back
        decoded = color_to_bits_lab(bgr_color)
        
        if decoded == i:
            print(f"✓ Color {i:2d} RGB{rgb_color} -> {decoded:2d}")
        else:
            print(f"✗ Color {i:2d} RGB{rgb_color} -> {decoded:2d} (WRONG!)")
            errors += 1
    
    if errors == 0:
        print("\n✓✓✓ All colors decoded correctly!")
        return True
    else:
        print(f"\n✗ {errors} color(s) failed")
        return False


def test_throughput_calculation():
    """Calculate theoretical throughput"""
    print("\n" + "="*60)
    print("Throughput calculations")
    print("="*60)
    
    test_data = b"X" * 1000  # 1KB test
    bars = encode_data_to_bars(test_data)
    
    print(f"1KB data = {len(bars)} bars")
    print(f"At 30 FPS with different bar heights:")
    
    for bar_height in [2, 4, 6, 8, 10]:
        screen_height = 1080
        bars_per_frame = screen_height // bar_height
        bytes_per_frame = bars_per_frame // 2  # 2 bars = 1 byte
        fps = 30
        throughput = bytes_per_frame * fps
        
        print(f"  {bar_height}px bars: {bars_per_frame} bars/frame = {bytes_per_frame} bytes/frame")
        print(f"           -> {throughput} bytes/sec = {throughput/1024:.2f} KB/sec")


if __name__ == "__main__":
    print("="*60)
    print("BARCODE ENCODING/DECODING TEST SUITE")
    print("="*60)
    
    # Run tests
    test1 = test_color_accuracy()
    test2 = test_encode_decode()
    test_throughput_calculation()
    
    print("\n" + "="*60)
    if test1 and test2:
        print("✓✓✓ ALL TESTS PASSED!")
        print("\nYou're ready to transmit data through the Matrix! 🌊")
        print("\nRun:")
        print("  python barcode_encode.py 'Your message here'")
        print("  python barcode_decode.py")
    else:
        print("✗ Some tests failed")
    print("="*60)

