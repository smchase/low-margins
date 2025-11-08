"""
Video Encoder - Encodes binary data into video frames
Each frame stores data in RGB pixel values
"""
import numpy as np
import struct


class VideoDataEncoder:
    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize the video encoder

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps

        # Calculate bytes per frame (3 bytes per pixel for RGB)
        # Reserve first row for metadata (length, checksum)
        self.metadata_pixels = width
        self.data_pixels = (width * height) - self.metadata_pixels
        self.bytes_per_frame = self.data_pixels * 3

    def encode_frame(self, data: bytes) -> np.ndarray:
        """
        Encode binary data into a single video frame

        Args:
            data: Binary data to encode (max bytes_per_frame bytes)

        Returns:
            numpy array of shape (height, width, 3) with dtype uint8
        """
        if len(data) > self.bytes_per_frame:
            raise ValueError(f"Data too large for frame: {len(data)} > {self.bytes_per_frame}")

        # Create empty frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Calculate checksum (simple sum modulo 256)
        checksum = sum(data) % 256

        # Encode metadata in first row
        # First 4 pixels: data length (32-bit integer)
        length_bytes = struct.pack('>I', len(data))
        frame[0, 0] = [length_bytes[0], length_bytes[1], length_bytes[2]]
        frame[0, 1] = [length_bytes[3], checksum, 0]

        # Encode data starting from second row
        data_with_padding = data + b'\x00' * (self.bytes_per_frame - len(data))

        # Convert bytes to pixels
        data_array = np.frombuffer(data_with_padding, dtype=np.uint8)
        data_pixels = data_array.reshape(-1, 3)

        # Fill frame with data (skip first row used for metadata)
        pixels_per_row = self.width
        for i, pixel_data in enumerate(data_pixels):
            row = (i // pixels_per_row) + 1  # +1 to skip metadata row
            col = i % pixels_per_row
            if row < self.height:
                frame[row, col] = pixel_data

        return frame

    def encode_stream(self, data: bytes):
        """
        Generator that yields frames for streaming large data

        Args:
            data: Binary data to encode

        Yields:
            Video frames as numpy arrays
        """
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + self.bytes_per_frame]
            yield self.encode_frame(chunk)
            offset += self.bytes_per_frame

        # Send empty frame to signal end of data
        yield self.encode_frame(b'')
