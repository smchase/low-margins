"""
Client: Simple CLI to send messages to transmitter
"""

import requests
import sys
import argparse


def send_message(server_url, message):
    """Send a message to the transmitter"""
    try:
        response = requests.post(
            f"{server_url}/send",
            json={'message': message},
            timeout=5
        )
        result = response.json()
        print(f"✓ Message sent: {result}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_status(server_url):
    """Check transmitter status"""
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        result = response.json()
        print(f"Status: {result}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def interactive_mode(server_url):
    """Interactive prompt for sending messages"""
    print(f"Connected to {server_url}")
    print("Type messages to send (Ctrl+C to exit)")
    print()

    while True:
        try:
            message = input("> ")
            if message:
                send_message(server_url, message)
                print()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send messages to NO-MARGIN-VIS transmitter")
    parser.add_argument(
        "--server",
        default="http://localhost:5000",
        help="Server URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--message", "-m",
        help="Message to send (if not provided, starts interactive mode)"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Check server status"
    )

    args = parser.parse_args()

    if args.status:
        check_status(args.server)
    elif args.message:
        send_message(args.server, args.message)
    else:
        interactive_mode(args.server)
