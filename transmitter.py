"""
Transmitter: Displays grid and serves messages via HTTP
"""

import cv2
import numpy as np
import threading
import time
from flask import Flask, request, jsonify
from encoder import MessageEncoder
from color_utils import draw_grid
from config import (
    GRID_SIZE, DISPLAY_WIDTH, DISPLAY_HEIGHT, CELL_WIDTH, CELL_HEIGHT,
    TARGET_FPS, FRAME_TIME_MS, SERVER_HOST, SERVER_PORT
)


class TransmitterApp:
    def __init__(self):
        self.encoder = MessageEncoder()
        self.running = True
        self.current_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.lock = threading.Lock()

        # Flask app
        self.flask_app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.flask_app.route('/send', methods=['POST'])
        def send_message():
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({'error': 'No message provided'}), 400

            message = data['message']
            with self.lock:
                self.encoder.queue_message(message)

            return jsonify({
                'status': 'queued',
                'message_length': len(message),
                'queue_size': len(self.encoder.message_queue)
            })

        @self.flask_app.route('/status', methods=['GET'])
        def status():
            with self.lock:
                return jsonify({
                    'frame_counter': self.encoder.frame_counter,
                    'queue_size': len(self.encoder.message_queue),
                    'running': self.running
                })

        @self.flask_app.route('/stop', methods=['POST'])
        def stop():
            self.running = False
            return jsonify({'status': 'stopping'})

    def display_thread(self):
        """Thread that generates and displays frames at 20fps"""
        print(f"Display thread started - {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} @ {TARGET_FPS}fps")

        # Create a fullscreen window
        window_name = "NO-MARGIN-VIS Transmitter"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        cv2.moveWindow(window_name, 0, 0)

        frame_time = FRAME_TIME_MS / 1000.0

        while self.running:
            frame_start = time.time()

            # Get next frame
            with self.lock:
                self.current_grid = self.encoder.create_frame()

            # Create display image
            display_image = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            draw_grid(display_image, self.current_grid, CELL_WIDTH, CELL_HEIGHT)

            # Add frame counter overlay
            with self.lock:
                frame_num = self.encoder.frame_counter
            cv2.putText(
                display_image,
                f"Frame: {frame_num}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Display
            cv2.imshow(window_name, display_image)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        cv2.destroyAllWindows()

    def start(self, debug=False):
        """Start transmitter"""
        print(f"Starting transmitter on {SERVER_HOST}:{SERVER_PORT}")

        # Start display thread
        display_t = threading.Thread(target=self.display_thread, daemon=True)
        display_t.start()

        # Start Flask server
        try:
            self.flask_app.run(host=SERVER_HOST, port=SERVER_PORT, debug=debug, threaded=True)
        except KeyboardInterrupt:
            self.running = False
            display_t.join(timeout=2)


if __name__ == "__main__":
    app = TransmitterApp()

    print("=" * 60)
    print("NO-MARGIN-VIS TRANSMITTER")
    print("=" * 60)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"Cell size: {CELL_WIDTH}x{CELL_HEIGHT} pixels")
    print(f"FPS: {TARGET_FPS}")
    print(f"\nServer: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"Send message: POST /send")
    print(f"  Payload: {{'message': 'your text here'}}")
    print(f"\nPress 'q' in display window to quit")
    print("=" * 60)

    app.start(debug=False)
