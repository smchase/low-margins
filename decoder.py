"""
Decoder: Convert grid frames back to readable messages
"""

import numpy as np
import struct
from config import FRAME_SIZE, GRID_SIZE
from color_utils import grid_to_bytes


class MessageDecoder:
    """Converts grid frames back into text messages"""

    def __init__(self):
        self.decoded_text = ""
        self.frame_count = 0
        self.last_message_id = -1
        self.message_buffers = {}  # Map message_id -> accumulated data

    def process_frame(self, grid_data):
        """
        Process a received grid frame.

        Args:
            grid_data: numpy array (128, 128) with values 0-15

        Returns:
            decoded_text: str of any new text decoded from this frame
        """
        # Convert grid to bytes
        frame_bytes = grid_to_bytes(grid_data)

        # Parse header
        frame_counter = struct.unpack('>I', frame_bytes[0:4])[0]
        message_id = struct.unpack('>H', frame_bytes[4:6])[0]
        message_length = struct.unpack('>H', frame_bytes[6:8])[0]

        # Detect new message (message ID changed)
        if message_id != self.last_message_id:
            self.last_message_id = message_id
            self.message_buffers[message_id] = bytearray()
            print(f"\n[NEW MESSAGE ID: {message_id}, Total length: {message_length} bytes]")

        # Extract payload
        payload = frame_bytes[8:FRAME_SIZE]

        # Add to message buffer
        self.message_buffers[message_id].extend(payload)

        # Try to decode the current message
        new_text = self._try_decode_message(message_id, message_length)

        self.frame_count += 1
        return new_text

    def _try_decode_message(self, message_id, expected_length):
        """
        Try to extract valid UTF-8 text from the message buffer.
        Returns newly decoded text.
        """
        new_text = ""
        buffer = self.message_buffers.get(message_id, b'')

        # Track how much we've decoded from this message before
        if not hasattr(self, '_decoded_lengths'):
            self._decoded_lengths = {}

        previously_decoded = self._decoded_lengths.get(message_id, 0)

        # Try to decode up to expected_length bytes
        data_to_decode = bytes(buffer[:expected_length])

        try:
            decoded = data_to_decode.decode('utf-8')

            # Only return the newly decoded part
            new_decoded = decoded[previously_decoded:]
            new_text = new_decoded
            self.decoded_text += new_text

            # Update how much we've decoded from this message
            self._decoded_lengths[message_id] = len(decoded)

        except UnicodeDecodeError:
            # Partial UTF-8 sequence, wait for more data
            pass

        return new_text

    def get_full_message(self):
        """Get the entire decoded message so far"""
        return self.decoded_text

    def get_stats(self):
        """Return decoding statistics"""
        return {
            'frames': self.frame_count,
            'messages': len(self.message_buffers),
            'message_length': len(self.decoded_text),
        }


if __name__ == "__main__":
    from color_utils import bytes_to_grid

    decoder = MessageDecoder()

    # Create a test frame with some data
    test_data = b'Hello from frame' + b'\x00' * (FRAME_SIZE - 16)
    test_grid = bytes_to_grid(test_data)

    text = decoder.process_frame(test_grid)
    print(f"Decoded: {repr(text)}")
    print(f"Full message: {repr(decoder.get_full_message())}")
