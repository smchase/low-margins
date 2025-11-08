"""
Virtual camera buffer for testing visual data link without two physical devices.
"""

import cv2
import numpy as np
import threading
from collections import deque


class VirtualCameraCapture:
    """
    Virtual camera that reads from an in-memory buffer.
    Implements cv2.VideoCapture interface.
    """

    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.is_open = True

    def write_frame(self, frame):
        """Write a frame to the virtual camera buffer (called by TX)"""
        with self.lock:
            self.buffer.append(frame.copy())

    def read(self):
        """Read a frame from the virtual camera buffer (called by RX)"""
        with self.lock:
            if len(self.buffer) == 0:
                # Return a blank frame if buffer is empty
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            return True, self.buffer[-1].copy()  # Return most recent frame

    def isOpened(self):
        return self.is_open

    def release(self):
        self.is_open = False
        with self.lock:
            self.buffer.clear()

    def set(self, prop_id, value):
        """Dummy method for compatibility"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
