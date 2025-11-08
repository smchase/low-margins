"""
Stream Receiver - Receives video stream using FFmpeg from network
"""
import subprocess
import numpy as np
import select
import sys


class VideoStreamReceiver:
    def __init__(self, port, width=640, height=480):
        """
        Initialize video stream receiver

        Args:
            port: Port to listen on
            width: Frame width
            height: Frame height
        """
        self.port = port
        self.width = width
        self.height = height
        self.frame_size = width * height * 3  # RGB24
        self.process = None

    def start(self):
        """
        Start the FFmpeg process for receiving stream
        """
        # FFmpeg command to receive stream and output raw video to stdout
        command = [
            'ffmpeg',
            '-i', f'udp://0.0.0.0:{self.port}?overrun_nonfatal=1&fifo_size=50000000',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-',  # Write to stdout
            '-loglevel', 'error'
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * 10
        )

        print(f"Started receiving on port {self.port}", flush=True)
        print(f"[DEBUG] FFmpeg command: {' '.join(command)}", flush=True)

    def receive_frame(self) -> np.ndarray:
        """
        Receive a single frame

        Returns:
            numpy array of shape (height, width, 3) with dtype uint8

        Raises:
            RuntimeError: If stream is not started or has ended
        """
        if self.process is None:
            raise RuntimeError("Stream not started. Call start() first.")

        try:
            # Read one frame worth of data
            raw_frame = self.process.stdout.read(self.frame_size)

            if len(raw_frame) != self.frame_size:
                raise RuntimeError("Stream ended or incomplete frame received")

            # Convert bytes to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 3))

            return frame

        except Exception as e:
            raise RuntimeError(f"Error receiving frame: {e}")

    def receive_frames(self):
        """
        Generator that yields frames as they are received

        Yields:
            Video frames as numpy arrays
        """
        while True:
            try:
                frame = self.receive_frame()
                yield frame
            except RuntimeError:
                break

    def stop(self):
        """
        Stop the receiving process
        """
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            print("Stopped receiving")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
