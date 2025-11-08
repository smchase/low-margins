"""
Video Tunnel - Bidirectional data communication via video encoding
"""
import sys
import threading
import time
from .video_encoder import VideoDataEncoder
from .video_decoder import VideoDataDecoder
from .stream_sender import VideoStreamSender
from .stream_receiver import VideoStreamReceiver


class VideoTunnel:
    def __init__(self, remote_host, send_port, recv_port, width=640, height=480, fps=30):
        """
        Initialize bidirectional video tunnel

        Args:
            remote_host: Remote host IP address
            send_port: Port to send data to
            recv_port: Port to receive data on
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.remote_host = remote_host
        self.send_port = send_port
        self.recv_port = recv_port
        self.width = width
        self.height = height
        self.fps = fps

        self.encoder = VideoDataEncoder(width, height, fps)
        self.decoder = VideoDataDecoder(width, height)

        self.sender = None
        self.receiver = None

        self.running = False
        self.send_thread = None
        self.recv_thread = None

    def _send_worker(self, input_stream):
        """
        Worker thread that reads from input stream and sends as video

        Args:
            input_stream: Input stream to read from (e.g., sys.stdin.buffer)
        """
        try:
            self.sender = VideoStreamSender(
                self.remote_host,
                self.send_port,
                self.width,
                self.height,
                self.fps
            )
            self.sender.start()

            # Calculate chunk size for optimal frame packing
            chunk_size = self.encoder.bytes_per_frame
            total_bytes = 0
            frame_count = 0

            print(f"[SEND] Waiting for input data (chunk size: {chunk_size} bytes)...", file=sys.stderr)

            while self.running:
                # Read chunk from input
                chunk = input_stream.read(chunk_size)

                if not chunk:
                    # End of input - send empty frame to signal completion
                    print(f"[SEND] End of input. Sent {frame_count} frames, {total_bytes} bytes total", file=sys.stderr)
                    self.sender.send_frame(self.encoder.encode_frame(b''))
                    break

                # Encode and send
                frame = self.encoder.encode_frame(chunk)
                self.sender.send_frame(frame)

                total_bytes += len(chunk)
                frame_count += 1
                print(f"[SEND] Frame {frame_count}: {len(chunk)} bytes (total: {total_bytes} bytes)", file=sys.stderr)

                # Control frame rate
                time.sleep(1.0 / self.fps)

        except Exception as e:
            print(f"[SEND] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            if self.sender:
                self.sender.stop()

    def _recv_worker(self, output_stream):
        """
        Worker thread that receives video and writes to output stream

        Args:
            output_stream: Output stream to write to (e.g., sys.stdout.buffer)
        """
        try:
            # Give sender time to start first
            print(f"[RECV] Waiting 2 seconds for sender to start...", file=sys.stderr)
            time.sleep(2)

            self.receiver = VideoStreamReceiver(
                self.recv_port,
                self.width,
                self.height
            )
            self.receiver.start()

            total_bytes = 0
            frame_count = 0
            print(f"[RECV] Waiting for frames...", file=sys.stderr)

            for frame in self.receiver.receive_frames():
                if not self.running:
                    break

                try:
                    data = self.decoder.decode_frame(frame)

                    if len(data) == 0:
                        # Empty frame signals end of stream
                        print(f"[RECV] End of stream. Received {frame_count} frames, {total_bytes} bytes total", file=sys.stderr)
                        break

                    output_stream.write(data)
                    output_stream.flush()

                    total_bytes += len(data)
                    frame_count += 1
                    print(f"[RECV] Frame {frame_count}: {len(data)} bytes (total: {total_bytes} bytes)", file=sys.stderr)

                except ValueError as e:
                    # Skip corrupted frames
                    print(f"[RECV] Warning - corrupted frame: {e}", file=sys.stderr)
                    continue

        except Exception as e:
            print(f"[RECV] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            if self.receiver:
                self.receiver.stop()

    def start(self, input_stream=None, output_stream=None):
        """
        Start the bidirectional tunnel

        Args:
            input_stream: Input stream to read from (default: sys.stdin.buffer)
            output_stream: Output stream to write to (default: sys.stdout.buffer)
        """
        if input_stream is None:
            input_stream = sys.stdin.buffer

        if output_stream is None:
            output_stream = sys.stdout.buffer

        self.running = True

        # Start sender thread
        self.send_thread = threading.Thread(
            target=self._send_worker,
            args=(input_stream,),
            daemon=True
        )
        self.send_thread.start()

        # Start receiver thread
        self.recv_thread = threading.Thread(
            target=self._recv_worker,
            args=(output_stream,),
            daemon=True
        )
        self.recv_thread.start()

        print(f"Tunnel started: sending to {self.remote_host}:{self.send_port}, receiving on :{self.recv_port}", file=sys.stderr)

    def wait(self):
        """
        Wait for both threads to complete
        """
        if self.send_thread:
            self.send_thread.join()
        if self.recv_thread:
            self.recv_thread.join()

    def stop(self):
        """
        Stop the tunnel
        """
        self.running = False

        if self.sender:
            self.sender.stop()
        if self.receiver:
            self.receiver.stop()

        if self.send_thread:
            self.send_thread.join(timeout=2)
        if self.recv_thread:
            self.recv_thread.join(timeout=2)

        print("Tunnel stopped", file=sys.stderr)
