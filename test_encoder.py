#!/usr/bin/env python3
"""
Simple test to verify encoder/decoder functionality
"""
from src.video_tunnel.video_encoder import VideoDataEncoder
from src.video_tunnel.video_decoder import VideoDataDecoder


def test_encode_decode():
    """Test basic encoding and decoding"""
    encoder = VideoDataEncoder(width=640, height=480)
    decoder = VideoDataDecoder(width=640, height=480)

    # Test data
    test_data = b"Hello, Video Tunnel! This is a test message. " * 100

    print(f"Original data length: {len(test_data)} bytes")
    print(f"Frame capacity: {encoder.bytes_per_frame} bytes")

    # Encode
    frame = encoder.encode_frame(test_data)
    print(f"Encoded frame shape: {frame.shape}")

    # Decode
    decoded_data = decoder.decode_frame(frame)
    print(f"Decoded data length: {len(decoded_data)} bytes")

    # Verify
    if decoded_data == test_data:
        print("✓ SUCCESS: Data matches!")
        return True
    else:
        print("✗ FAILURE: Data mismatch!")
        print(f"Expected: {test_data[:100]}")
        print(f"Got: {decoded_data[:100]}")
        return False


def test_large_data():
    """Test encoding data that requires multiple frames"""
    encoder = VideoDataEncoder(width=640, height=480)
    decoder = VideoDataDecoder(width=640, height=480)

    # Create larger test data
    test_data = b"X" * (encoder.bytes_per_frame * 3 + 12345)
    print(f"\nTesting large data: {len(test_data)} bytes")

    # Encode to multiple frames
    frames = list(encoder.encode_stream(test_data))
    print(f"Encoded to {len(frames)} frames")

    # Decode from frames
    decoded_chunks = list(decoder.decode_stream(frames))
    decoded_data = b"".join(decoded_chunks)

    print(f"Decoded {len(decoded_data)} bytes")

    if decoded_data == test_data:
        print("✓ SUCCESS: Large data stream works!")
        return True
    else:
        print("✗ FAILURE: Large data mismatch!")
        return False


if __name__ == "__main__":
    print("Testing Video Encoder/Decoder")
    print("=" * 50)

    success = True
    success &= test_encode_decode()
    success &= test_large_data()

    print("=" * 50)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
