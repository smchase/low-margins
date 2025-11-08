"""
Transmitter: Displays grid and serves messages via HTTP
"""

import cv2
import numpy as np
import threading
import time
import base64
from flask import Flask, request, jsonify, render_template_string
from encoder import MessageEncoder
from color_utils import draw_grid, COLOR_PALETTE
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

        @self.flask_app.route('/')
        def index():
            """Serve web UI"""
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>NO-MARGIN-VIS Transmitter</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { font-family: monospace; background: #000; color: #0f0; overflow: hidden; }
                    #container { display: flex; flex-direction: column; height: 100vh; }
                    #grid {
                        flex: 1;
                        display: flex;
                        flex-wrap: wrap;
                        align-content: flex-start;
                        border: 2px solid #0f0;
                        background: #000;
                    }
                    .cell {
                        flex-grow: 1;
                        flex-basis: calc(100% / 64);
                        aspect-ratio: 1;
                        background: #000;
                    }
                    #controls {
                        background: #111;
                        padding: 15px;
                        border-top: 2px solid #0f0;
                        display: flex;
                        gap: 10px;
                        align-items: center;
                    }
                    #stats {
                        color: #0f0;
                        font-size: 14px;
                        min-width: 300px;
                    }
                    #message-input {
                        flex: 1;
                        padding: 10px;
                        font-size: 16px;
                        font-family: monospace;
                        background: #000;
                        color: #0f0;
                        border: 1px solid #0f0;
                    }
                    button {
                        padding: 10px 20px;
                        font-size: 16px;
                        cursor: pointer;
                        background: #0f0;
                        color: #000;
                        border: none;
                        font-weight: bold;
                    }
                    button:hover { background: #00ff00aa; }
                </style>
            </head>
            <body>
                <div id="container">
                    <div id="grid"></div>
                    <div id="controls">
                        <div id="stats">Frames: 0 | Queue: 0</div>
                        <input id="message-input" type="text" placeholder="Enter message...">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
                <script>
                    const GRID_SIZE = 64;
                    const COLOR_MAP = {
                        0: '#000000', 1: '#FF0000', 2: '#00FF00', 3: '#0000FF',
                        4: '#00FFFF', 5: '#FF00FF', 6: '#FFFF00', 7: '#FFFFFF',
                        8: '#800000', 9: '#008000', 10: '#000080', 11: '#FFA500',
                        12: '#800080', 13: '#FFC0CB', 14: '#00FF7F', 15: '#008080'
                    };

                    function createGrid() {
                        const gridDiv = document.getElementById('grid');
                        for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
                            const cell = document.createElement('div');
                            cell.className = 'cell';
                            cell.id = 'cell-' + i;
                            cell.style.backgroundColor = '#000000';
                            gridDiv.appendChild(cell);
                        }
                    }

                    function updateGrid() {
                        fetch('/grid')
                            .then(r => r.json())
                            .then(data => {
                                for (let i = 0; i < data.grid.length; i++) {
                                    const colorIdx = data.grid[i];
                                    document.getElementById('cell-' + i).style.backgroundColor = COLOR_MAP[colorIdx];
                                }
                                document.getElementById('stats').innerHTML =
                                    'Frames: ' + data.frames + ' | Queue: ' + data.queue + ' | FPS: ' + (data.fps || 20).toFixed(1);
                            });
                    }

                    function sendMessage() {
                        const msg = document.getElementById('message-input').value;
                        if (!msg) return;

                        fetch('/send', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: msg})
                        }).then(r => r.json())
                          .then(data => {
                              document.getElementById('message-input').value = '';
                              console.log('Sent:', data);
                          });
                    }

                    document.getElementById('message-input').addEventListener('keypress', e => {
                        if (e.key === 'Enter') sendMessage();
                    });

                    createGrid();
                    setInterval(updateGrid, 50);
                </script>
            </body>
            </html>
            '''
            return render_template_string(html)

        @self.flask_app.route('/grid')
        def get_grid():
            """Get current grid as JSON"""
            with self.lock:
                grid_flat = self.current_grid.flatten().tolist()
                return jsonify({
                    'grid': grid_flat,
                    'frames': self.encoder.frame_counter,
                    'queue': len(self.encoder.message_queue),
                    'fps': TARGET_FPS
                })

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

        # Try to create display window, but continue if display unavailable
        window_name = "NO-MARGIN-VIS Transmitter"
        display_available = False

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            cv2.moveWindow(window_name, 0, 0)
            display_available = True
            print("Display window created successfully")
        except cv2.error as e:
            print(f"WARNING: Display not available - running in headless mode")
            print(f"  (This is normal on headless/remote systems)")
            print(f"  Grid frames are still being generated at {TARGET_FPS}fps")
            print(f"  Connect via HTTP to send/receive messages")

        frame_time = FRAME_TIME_MS / 1000.0

        while self.running:
            frame_start = time.time()

            # Get next frame
            with self.lock:
                self.current_grid = self.encoder.create_frame()

            # Only try to display if display is available
            if display_available:
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
                try:
                    cv2.imshow(window_name, display_image)
                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                except cv2.error:
                    # Display was lost (e.g., window closed)
                    display_available = False

            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        if display_available:
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
