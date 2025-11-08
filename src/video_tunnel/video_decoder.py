"""
Video Decoder - Decodes binary data from video frames
Extracts data from RGB pixel values
"""
import numpy as np
import struct


class VideoDataDecoder:
    def __init__(self, width=640, height=480):
        """
        Initialize the video decoder

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.width = width
        self.height = height

        # Calculate bytes per frame (must match encoder)
        self.metadata_pixels = width
        self.data_pixels = (width * height) - self.metadata_pixels
        self.bytes_per_frame = self.data_pixels * 3

    def decode_frame(self, frame: np.ndarray) -> bytes:
        """
        Decode binary data from a video frame

        Args:
            frame: numpy array of shape (height, width, 3) with dtype uint8

        Returns:
            Decoded binary data

        Raises:
            ValueError: If frame is corrupted or checksum fails
        """
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        # Extract metadata from first row
        length_bytes = bytes([frame[0, 0, 0], frame[0, 0, 1], frame[0, 0, 2], frame[0, 1, 0]])
        data_length = struct.unpack('>I', length_bytes)[0]
        stored_checksum = frame[0, 1, 1]

        if data_length > self.bytes_per_frame:
            raise ValueError(f"Invalid data length in frame: {data_length}")

        # Extract data from frame (skip first row)
        data_pixels = []
        pixels_per_row = self.width

        num_pixels_needed = (data_length + 2) // 3 + 1  # Round up division

        for i in range(num_pixels_needed):
            row = (i // pixels_per_row) + 1  # +1 to skip metadata row
            col = i % pixels_per_row
            if row < self.height:
                data_pixels.append(frame[row, col])

        # Convert pixels back to bytes
        data_array = np.array(data_pixels, dtype=np.uint8).flatten()
        data = bytes(data_array[:data_length])

        # Verify checksum
        calculated_checksum = sum(data) % 256
        if calculated_checksum != stored_checksum:
            raise ValueError(f"Checksum mismatch: {calculated_checksum} != {stored_checksum}")

        return data

    def decode_stream(self, frames):
        """
        Generator that decodes data from a stream of frames

        Args:
            frames: Iterator of video frames

        Yields:
            Decoded binary data chunks
        """
        for frame in frames:
            try:
                data = self.decode_frame(frame)
                if len(data) == 0:
                    # Empty frame signals end of stream
                    break
                yield data
            except ValueError as e:
                # Skip corrupted frames
                print(f"Warning: Skipping corrupted frame: {e}")
                continue
