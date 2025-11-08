"""
Stream Sender - Sends video stream using FFmpeg over network
"""
import subprocess
import numpy as np
import sys
import signal


class VideoStreamSender:
    def __init__(self, host, port, width=640, height=480, fps=30, bitrate='2M'):
        """
        Initialize video stream sender

        Args:
            host: Target host IP address
            port: Target port number
            width: Frame width
            height: Frame height
            fps: Frames per second
            bitrate: Video bitrate (e.g., '2M' for 2 Mbps)
        """
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.process = None

    def start(self):
        """
        Start the FFmpeg process for streaming
        """
        # FFmpeg command to read raw video from stdin and stream over UDP
        command = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-b:v', self.bitrate,
            '-f', 'mpegts',
            f'udp://{self.host}:{self.port}?pkt_size=1316',
            '-loglevel', 'error'
        ]

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        print(f"Started streaming to {self.host}:{self.port}")

    def send_frame(self, frame: np.ndarray):
        """
        Send a single frame

        Args:
            frame: numpy array of shape (height, width, 3) with dtype uint8
        """
        if self.process is None:
            raise RuntimeError("Stream not started. Call start() first.")

        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        try:
            # Write frame to FFmpeg stdin
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("FFmpeg process has terminated")

    def send_frames(self, frames):
        """
        Send multiple frames from an iterator

        Args:
            frames: Iterator of video frames
        """
        for frame in frames:
            self.send_frame(frame)

    def stop(self):
        """
        Stop the streaming process
        """
        if self.process:
            self.process.stdin.close()
            self.process.wait(timeout=5)
            self.process = None
            print("Stopped streaming")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
