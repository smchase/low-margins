"""
Transmitter: Displays grid and serves messages via HTTP
"""

import cv2
import numpy as np
import threading
import time
from flask import Flask, request, jsonify, render_template_string
from encoder import MessageEncoder
from color_utils import draw_grid, COLOR_PALETTE
from config import (
    GRID_SIZE, DISPLAY_WIDTH, DISPLAY_HEIGHT, CELL_WIDTH, CELL_HEIGHT,
    TARGET_FPS, FRAME_TIME_MS, SERVER_HOST, SERVER_PORT, GRID_BORDER_THICKNESS,
    GRID_BORDER_COLOR
)


class TransmitterApp:
    def __init__(self, test_message="HELLO TEST"):
        self.encoder = MessageEncoder()
        self.running = True
        self.current_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.lock = threading.Lock()

        # For simple test mode: queue the test message once
        self.encoder.queue_message(test_message)

        # Flask app
        self.flask_app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.flask_app.route('/')
        def index():
            """Serve web UI"""
            import json

            def bgr_to_hex(bgr):
                b, g, r = bgr
                return f"#{r:02X}{g:02X}{b:02X}"

            color_map = {idx: bgr_to_hex(color) for idx, color in COLOR_PALETTE.items()}

            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>NO-MARGIN-VIS Transmitter</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    html, body { height: 100%; width: 100%; }
                    body { font-family: monospace; background: #fff; color: #000; overflow: hidden; }
                    #container { display: flex; flex-direction: column; height: 100%; width: 100%; }
                    #grid {
                        flex: 1;
                        min-height: 0;
                        aspect-ratio: 1;
                        width: 100%;
                        max-width: 100%;
                        max-height: 100%;
                        display: grid;
                        border: 32px solid #0f0;
                        background: #fff;
                        overflow: visible;
                        gap: 0;
                        margin: auto;
                        box-shadow: 0 0 0 4px #0f0, inset 0 0 0 4px #0f0;
                        position: relative;
                    }
                    .cell {
                        background: #fff;
                        border: 0;
                    }
                    .corner-marker {
                        position: absolute;
                        width: 30px;
                        height: 30px;
                        background: #000;
                        z-index: 10;
                    }
                    .corner-marker.top-left { top: -15px; left: -15px; }
                    .corner-marker.top-right { top: -15px; right: -15px; }
                    .corner-marker.bottom-left { bottom: -15px; left: -15px; }
                    .corner-marker.bottom-right { bottom: -15px; right: -15px; }
                    #controls {
                        flex-shrink: 0;
                        background: #fff;
                        padding: 10px;
                        border-top: 4px solid #0f0;
                        display: flex;
                        gap: 10px;
                        align-items: center;
                        height: auto;
                    }
                    #stats {
                        color: #000;
                        font-size: 12px;
                        min-width: 250px;
                        white-space: nowrap;
                    }
                    #message-input {
                        flex: 1;
                        padding: 8px;
                        font-size: 14px;
                        font-family: monospace;
                        background: #fff;
                        color: #000;
                        border: 1px solid #999;
                        min-width: 200px;
                    }
                    button {
                        padding: 8px 15px;
                        font-size: 14px;
                        cursor: pointer;
                        background: #0f0;
                        color: #000;
                        border: none;
                        font-weight: bold;
                        flex-shrink: 0;
                    }
                    button:hover { background: #00dd00; }
                </style>
            </head>
            <body>
                <div id="container">
                    <div id="grid">
                        <div class="corner-marker top-left"></div>
                        <div class="corner-marker top-right"></div>
                        <div class="corner-marker bottom-left"></div>
                        <div class="corner-marker bottom-right"></div>
                    </div>
                    <div id="controls">
                        <div id="stats">Frames: 0 | Queue: 0</div>
                        <input id="message-input" type="text" placeholder="Enter message...">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
                <script>
                    let GRID_SIZE = 0;  // Will be set dynamically
                    const COLOR_MAP = __COLOR_MAP__;

                    function createGrid(gridSize) {
                        const gridDiv = document.getElementById('grid');
                        // Remove only cells, keep corner markers
                        const cells = gridDiv.querySelectorAll('.cell');
                        cells.forEach(cell => cell.remove());

                        GRID_SIZE = gridSize;

                        // Update CSS grid to match actual size
                        gridDiv.style.gridTemplateColumns = `repeat(${gridSize}, 1fr)`;
                        gridDiv.style.gridTemplateRows = `repeat(${gridSize}, 1fr)`;

                        for (let i = 0; i < gridSize * gridSize; i++) {
                            const cell = document.createElement('div');
                            cell.className = 'cell';
                            cell.id = 'cell-' + i;
                            cell.style.backgroundColor = '#FFFFFF';
                            gridDiv.appendChild(cell);
                        }
                    }

                    function updateGrid() {
                        fetch('/grid')
                            .then(r => r.json())
                            .then(data => {
                                // Initialize grid on first update if not done
                                if (GRID_SIZE === 0) {
                                    const gridSize = Math.sqrt(data.grid.length);
                                    createGrid(gridSize);
                                }

                                // Update grid colors
                                for (let i = 0; i < data.grid.length; i++) {
                                    const colorIdx = data.grid[i];
                                    const cell = document.getElementById('cell-' + i);
                                    if (cell) {
                                        cell.style.backgroundColor = COLOR_MAP[colorIdx];
                                    }
                                }

                                document.getElementById('stats').innerHTML =
                                    'Frames: ' + data.frames + ' | Queue: ' + data.queue;
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

                    setInterval(updateGrid, 50);
                </script>
            </body>
            </html>
            '''
            html = html.replace('__COLOR_MAP__', json.dumps(color_map))
            return render_template_string(html)

        @self.flask_app.route('/grid')
        def get_grid():
            """Get current grid as JSON"""
            with self.lock:
                grid_flat = self.current_grid.flatten().tolist()
                return jsonify({
                    'grid': grid_flat,
                    'frames': self.encoder.global_frame_counter,
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
                    'frame_counter': self.encoder.global_frame_counter,
                    'queue_size': len(self.encoder.message_queue),
                    'running': self.running
                })

        @self.flask_app.route('/stop', methods=['POST'])
        def stop():
            self.running = False
            return jsonify({'status': 'stopping'})

    def display_loop(self):
        """Generate and display frames at 20fps (runs on main thread)."""
        print(f"Display loop started - {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} @ {TARGET_FPS}fps")

        # Try to create display window, but continue if display unavailable
        window_name = "NO-MARGIN-VIS Transmitter"
        display_available = False

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            cv2.moveWindow(window_name, 0, 0)
            display_available = True
            print("✓ Display window created successfully")
        except Exception as e:
            print(f"⚠ Display not available: {type(e).__name__}: {e}")
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
                cv2.rectangle(
                    display_image,
                    (GRID_BORDER_THICKNESS // 2, GRID_BORDER_THICKNESS // 2),
                    (DISPLAY_WIDTH - GRID_BORDER_THICKNESS // 2 - 1, DISPLAY_HEIGHT - GRID_BORDER_THICKNESS // 2 - 1),
                    GRID_BORDER_COLOR,
                    GRID_BORDER_THICKNESS
                )

                # Add frame counter overlay
                with self.lock:
                    frame_num = self.encoder.global_frame_counter
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
                except Exception:
                    # Display was lost (e.g., window closed)
                    display_available = False

            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        if display_available:
            cv2.destroyAllWindows()

    def _run_server(self, debug):
        """Run Flask server in a background thread."""
        try:
            self.flask_app.run(host=SERVER_HOST, port=SERVER_PORT, debug=debug, threaded=True)
        except Exception as exc:
            print(f"Flask server stopped: {exc}")

    def start(self, debug=False):
        """Start transmitter"""
        print(f"Starting transmitter on {SERVER_HOST}:{SERVER_PORT}")

        server_thread = threading.Thread(
            target=self._run_server, args=(debug,), daemon=True, name="TransmitterHTTP"
        )
        server_thread.start()

        try:
            self.display_loop()
        except KeyboardInterrupt:
            print("Interrupted, shutting down transmitter...")
        finally:
            self.running = False
            server_thread.join(timeout=1)


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
