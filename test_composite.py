#!/usr/bin/env python3
"""
Test receiver with composite camera (real camera + TX grid overlay).

The TX grid is rendered smaller and centered on top of the real camera feed.
This allows testing on a single machine with just one camera.

Usage:
    python test_composite.py
"""

import sys
from composite_camera import CompositeCamera
from visual_data_link import main as vdl_main


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Visual Data Link receiver with composite camera'
    )
    parser.add_argument(
        '--grid-size', type=int, default=16,
        help='Grid size (default: 16)'
    )
    parser.add_argument(
        '--cell-size', type=int, default=40,
        help='Cell size in pixels (default: 40 - overlay size)'
    )
    args, unknown = parser.parse_known_args()

    print("\n" + "="*70)
    print("VISUAL DATA LINK - RECEIVER WITH COMPOSITE CAMERA")
    print("="*70)
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Cell size: {args.cell_size}px (overlay)")
    print()
    print("The TX grid will appear smaller and centered on your camera feed (solid, no transparency).")
    print("Press 'c' to calibrate, then RX will lock and decode the grid.")
    print("="*70 + "\n")

    try:
        # Create composite camera (real camera with TX grid on top)
        cap = CompositeCamera(
            grid_size=args.grid_size,
            cell_size=args.cell_size
        )

        # Run the receiver with composite camera
        sys.argv = ['test_composite.py', '--mode', 'rx', '--grid-size', str(args.grid_size)]

        vdl_main(camera=cap)

    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and available.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        if 'cap' in locals():
            cap.release()


if __name__ == "__main__":
    main()
