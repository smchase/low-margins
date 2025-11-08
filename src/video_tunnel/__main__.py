#!/usr/bin/env python3
"""
Video Tunnel - Stream data encoded as video between computers on a local network
"""
import argparse
import signal
import sys
from .tunnel import VideoTunnel


def main():
    parser = argparse.ArgumentParser(
        description='Video Tunnel - Stream data encoded as video over local network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Computer A (192.168.1.10) - connects to Computer B at 192.168.1.20
  cat file.txt | video-tunnel connect 192.168.1.20 > received.txt

  # Computer B (192.168.1.20) - listens and sends back to Computer A
  echo "Hello" | video-tunnel listen 192.168.1.10 > output.txt

  # Bidirectional communication (both need each other's IPs)
  # Machine A (192.168.1.10):
  video-tunnel connect 192.168.1.20

  # Machine B (192.168.1.20):
  video-tunnel listen 192.168.1.10
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Connect command (initiator)
    connect_parser = subparsers.add_parser('connect', help='Connect to remote host')
    connect_parser.add_argument('host', help='Remote host IP address')
    connect_parser.add_argument('--send-port', type=int, default=5000, help='Port to send to (default: 5000)')
    connect_parser.add_argument('--recv-port', type=int, default=5001, help='Port to receive on (default: 5001)')
    connect_parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    connect_parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    connect_parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')

    # Listen command (responder)
    listen_parser = subparsers.add_parser('listen', help='Listen for incoming connection')
    listen_parser.add_argument('remote_host', help='Remote host IP address to send back to')
    listen_parser.add_argument('--send-port', type=int, default=5001, help='Port to send to (default: 5001)')
    listen_parser.add_argument('--recv-port', type=int, default=5000, help='Port to receive on (default: 5000)')
    listen_parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    listen_parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    listen_parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Determine remote host based on mode
    if args.command == 'connect':
        remote_host = args.host
        print(f"Connecting to {remote_host}...", file=sys.stderr)
    else:  # listen
        remote_host = args.remote_host

        # Get local IP to display to user
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()

        print(f"Listening on {local_ip}:{args.recv_port}", file=sys.stderr)
        print(f"Sending to {remote_host}:{args.send_port}", file=sys.stderr)

    # Create tunnel
    tunnel = VideoTunnel(
        remote_host=remote_host,
        send_port=args.send_port,
        recv_port=args.recv_port,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...", file=sys.stderr)
        tunnel.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start tunnel
    try:
        tunnel.start()
        tunnel.wait()
    except KeyboardInterrupt:
        tunnel.stop()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        tunnel.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
