"""
Test suite for NO-MARGIN-VIS
Run without hardware to verify logic
"""

import numpy as np
from encoder import MessageEncoder
from decoder import MessageDecoder
from color_utils import bytes_to_grid, grid_to_bytes, ColorDetector
from config import GRID_SIZE


def test_encoding_decoding():
    """Test that messages can be encoded and decoded correctly"""
    print("\n=== Testing Encoding/Decoding ===")

    encoder = MessageEncoder()
    decoder = MessageDecoder()

    test_message = "Hello World! This is a test message."
    encoder.queue_message(test_message)

    print(f"Original message: {repr(test_message)}")

    # Generate and process several frames
    for i in range(50):
        grid = encoder.create_frame()
        decoded_text = decoder.process_frame(grid)

        if i % 10 == 0:
            print(f"  Frame {i}: decoded length = {len(decoder.get_full_message())}")

    full_decoded = decoder.get_full_message()
    print(f"Decoded message: {repr(full_decoded)}")

    # Check if message was at least partially decoded
    if test_message[:5] in full_decoded:
        print("✓ Encoding/decoding test PASSED")
        return True
    else:
        print("✗ Encoding/decoding test FAILED")
        return False


def test_grid_byte_conversion():
    """Test converting between grids and bytes"""
    print("\n=== Testing Grid/Byte Conversion ===")

    # Create a test grid
    grid = np.random.randint(0, 16, (GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    # Convert to bytes
    data = grid_to_bytes(grid)
    print(f"Grid size: {grid.shape}")
    print(f"Bytes size: {len(data)}")

    # Convert back
    grid_reconstructed = bytes_to_grid(data)

    # Check if they match
    if np.array_equal(grid, grid_reconstructed):
        print("✓ Grid/byte conversion test PASSED")
        return True
    else:
        print("✗ Grid/byte conversion test FAILED")
        print(f"  Differences: {np.sum(grid != grid_reconstructed)}")
        return False


def test_color_detection():
    """Test color detection"""
    print("\n=== Testing Color Detection ===")

    detector = ColorDetector()

    test_cases = [
        ((0, 0, 0), 0, "Black"),
        ((0, 0, 255), 1, "Red"),
        ((0, 255, 0), 2, "Green"),
        ((255, 0, 0), 3, "Blue"),
        ((255, 255, 255), 7, "White"),
    ]

    passed = 0
    for pixel, expected_idx, name in test_cases:
        detected_idx = detector.detect_color(pixel)
        status = "✓" if detected_idx == expected_idx else "✗"
        print(f"  {status} {name}: expected {expected_idx}, got {detected_idx}")
        if detected_idx == expected_idx:
            passed += 1

    if passed == len(test_cases):
        print("✓ Color detection test PASSED")
        return True
    else:
        print("✗ Color detection test FAILED")
        return False


def test_multiple_messages():
    """Test queueing multiple messages"""
    print("\n=== Testing Multiple Messages ===")

    encoder = MessageEncoder()
    decoder = MessageDecoder()

    messages = [
        "First message",
        "Second message",
        "Third message"
    ]

    for msg in messages:
        encoder.queue_message(msg)

    print(f"Queued {len(messages)} messages")
    print(f"Queue size: {len(encoder.message_queue)}")

    # Generate 200 frames to process all messages
    for i in range(200):
        grid = encoder.create_frame()
        decoder.process_frame(grid)

    decoded = decoder.get_full_message()
    print(f"Decoded length: {len(decoded)} characters")

    # Check if all messages were at least partially captured
    success = all(msg[:5] in decoded for msg in messages)

    if success:
        print("✓ Multiple messages test PASSED")
        return True
    else:
        print("✗ Multiple messages test FAILED")
        return False


def test_frame_rate():
    """Test frame generation timing"""
    print("\n=== Testing Frame Rate ===")

    import time
    from config import TARGET_FPS, FRAME_TIME_MS

    encoder = MessageEncoder()
    encoder.queue_message("Test message")

    frame_times = []

    for i in range(100):
        start = time.time()
        encoder.create_frame()
        elapsed = time.time() - start
        frame_times.append(elapsed * 1000)  # Convert to ms

    avg_time = np.mean(frame_times)
    print(f"Average frame generation time: {avg_time:.2f}ms")
    print(f"Target frame time: {FRAME_TIME_MS}ms")
    print(f"Frames generated per second: {1000.0 / avg_time:.1f}")

    # Should be able to generate frames much faster than FRAME_TIME_MS
    if avg_time < FRAME_TIME_MS / 2:
        print("✓ Frame rate test PASSED")
        return True
    else:
        print("✗ Frame rate test FAILED (generation too slow)")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("NO-MARGIN-VIS TEST SUITE")
    print("=" * 60)

    results = {
        'Grid/Byte Conversion': test_grid_byte_conversion(),
        'Color Detection': test_color_detection(),
        'Frame Rate': test_frame_rate(),
        'Encoding/Decoding': test_encoding_decoding(),
        'Multiple Messages': test_multiple_messages(),
    }

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
