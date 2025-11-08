"""
Encoder: Convert messages to grid frames for transmission
"""

import numpy as np
import struct
from config import FRAME_SIZE, GRID_SIZE, FRAME_HEADER
from color_utils import bytes_to_grid


class MessageEncoder:
    """Converts text messages into a stream of grid frames"""

    def __init__(self):
        self.frame_counter = 0
        self.message_queue = []
        self.current_message_id = 0
        self.current_message = None
        self.current_message_offset = 0

    def queue_message(self, text):
        """
        Queue a message for transmission.
        Message will be split into frames automatically.

        Args:
            text: String message to transmit
        """
        message_bytes = text.encode('utf-8')
        self.message_queue.append(message_bytes)
        print(f"Queued message: {len(message_bytes)} bytes")

    def create_frame(self):
        """
        Generate the next frame to display.

        Returns:
            numpy array (128, 128) with values 0-15
        """
        # Get next message if not currently sending one
        if self.current_message is None:
            if self.message_queue:
                self.current_message = self.message_queue.pop(0)
                self.current_message_offset = 0
                print(f"Starting transmission of message (ID: {self.current_message_id}, {len(self.current_message)} bytes)")
            else:
                # No message: display idle pattern
                return self._create_idle_pattern(self.frame_counter)

        # Build frame payload
        payload = bytearray()

        # Header: frame counter (4 bytes) + message id (2 bytes) + message length (2 bytes)
        payload.extend(struct.pack('>I', self.frame_counter))  # Frame counter
        payload.extend(struct.pack('>H', self.current_message_id))  # Message ID
        payload.extend(struct.pack('>H', len(self.current_message)))  # Total message length

        # Data: copy from current message
        bytes_to_copy = FRAME_SIZE - len(payload)
        remaining = len(self.current_message) - self.current_message_offset

        if remaining > 0:
            # Copy actual message data
            copy_size = min(bytes_to_copy, remaining)
            payload.extend(self.current_message[self.current_message_offset:self.current_message_offset + copy_size])
            self.current_message_offset += copy_size

            # Check if message is done
            if self.current_message_offset >= len(self.current_message):
                print(f"Message (ID: {self.current_message_id}) transmission complete")
                self.current_message = None
                self.current_message_id += 1

        # Pad payload to FRAME_SIZE with zeros
        while len(payload) < FRAME_SIZE:
            payload.append(0)

        # Convert bytes to grid
        payload = bytes(payload[:FRAME_SIZE])
        frame = bytes_to_grid(payload)

        self.frame_counter += 1
        return frame

    def _create_idle_pattern(self, frame_num):
        """Create a test pattern when no message is being sent"""
        frame = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        # Checkerboard pattern that cycles
        cycle = (frame_num // 10) % 16
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if (row + col) % 2 == 0:
                    frame[row, col] = cycle
                else:
                    frame[row, col] = (cycle + 8) % 16

        return frame


if __name__ == "__main__":
    encoder = MessageEncoder()
    encoder.queue_message("Hello World")

    # Generate first few frames
    for i in range(5):
        frame = encoder.create_frame()
        print(f"Frame {i}: shape={frame.shape}, unique colors={len(np.unique(frame))}")
