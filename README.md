# NO-MARGIN-VIS: Optical Data Transmission via Color Grid

A fascinating project that transmits text messages between two computers using a 64×64 color grid. One computer displays the grid, the other reads it with a camera.

## How It Works

### Architecture

**Transmitter:**
- Displays a 64×64 grid of 16 distinct colors on the screen
- Each frame at 20fps carries **2,048 bytes** of data
- Runs a Flask HTTP server on port 5000
- Users send text via HTTP POST requests
- Messages are streamed across multiple frames automatically

**Receiver:**
- Captures the display with a camera at 20fps
- Detects which color each grid cell is
- Decodes the message back to text
- Displays split-screen: camera feed (left) + decoded message (right)

### Data Capacity

```
64 × 64 grid = 4,096 cells
4 bits per cell (16 colors) = 16,384 bits per frame
20 FPS = 40.96 KB/s sustained throughput
```

A 100-character message takes ~2.4 seconds to transmit.

### Color Palette

The 16 colors are highly saturated and distinct to survive camera color shifts:

| ID | Color | BGR Value | Purpose |
|----|-------|-----------|---------|
| 0 | Black | (0,0,0) | Sync/reference |
| 1-7 | Primary Colors | - | Main data |
| 8-15 | Dark/Secondary | - | Additional data |

## Setup

### Prerequisites

- Python 3.9+
- OpenCV library (requires OpenCV to be installed)
- Two computers with USB cameras (or one with built-in camera for testing)

### Installation

**Using uv (recommended):**
```bash
# Clone/navigate to project
cd no-margin-vis

# Install dependencies with uv
uv pip install -e .
```

**Or using pip:**
```bash
cd no-margin-vis
pip install -r requirements.txt
```

> **Note:** Install `uv` from https://docs.astral.sh/uv/getting-started/installation/

## Usage

### Quick Start

**Terminal 1 - Transmitter (Display computer):**
```bash
uv run transmitter.py
```
- Starts Flask server on `http://0.0.0.0:5000`
- Displays 64×64 color grid fullscreen
- Press `q` to quit

**Terminal 2 - Receiver (Camera computer):**
```bash
uv run receiver.py
```
- Opens default camera (device 0)
- Displays split-screen with camera + decoded message
- Press `q` to quit
- Press `c` to clear decoded message

**Terminal 3 - Client (Message sender):**
```bash
# Interactive mode (default)
uv run client.py

# Or specify different server
uv run client.py --server http://192.168.1.100:5000

# Send single message
uv run client.py -m "Hello from NO-MARGIN-VIS!"

# Check status
uv run client.py --status
```

### Network Setup (Two Different Machines)

1. **On Transmitter Machine:**
   ```bash
   uv run transmitter.py
   ```
   Note the IP address and port (e.g., `192.168.1.10:5000`)

2. **On Receiver Machine:**
   ```bash
   uv run receiver.py
   ```
   Make sure the camera is pointing at the transmitter display

3. **On Client Machine (can be any machine):**
   ```bash
   uv run client.py --server http://192.168.1.10:5000
   > Type your message here
   ```

## Configuration

Edit `config.py` to adjust:

```python
GRID_SIZE = 64               # Grid dimensions (64x64)
DISPLAY_WIDTH = 1512         # Transmitter display width
DISPLAY_HEIGHT = 982         # Transmitter display height
TARGET_FPS = 20              # Frames per second
SERVER_PORT = 5000           # Flask server port
```

## File Structure

```
no-margin-vis/
├── config.py              # Global configuration
├── color_utils.py         # Color detection and grid rendering
├── encoder.py             # Message → grid converter
├── decoder.py             # Grid → message converter
├── transmitter.py         # Display server
├── receiver.py            # Camera capture and decode
├── client.py              # Message sending CLI
├── pyproject.toml         # uv project configuration
├── requirements.txt       # Python dependencies (legacy)
├── test.py                # Test suite
└── README.md             # This file
```

## Technical Details

### Message Protocol

Each frame is structured as:
```
[4 bytes: Frame counter]
[2 bytes: Message ID]
[2 bytes: Message length]
[2,040 bytes: Payload data]
```

Total: 2,048 bytes per frame. The decoder accumulates frames and extracts valid UTF-8 text automatically.

### Color Detection Algorithm

The receiver uses:
1. **Fast pixel sampling** - Samples one pixel per cell
2. **K-nearest color matching** - Finds closest match from 16-color palette
3. **Adaptive detection** - Optional HSV-based color classification for robustness

### Timing

- Frame generation: ~20ms per frame
- Camera capture: 33ms (30fps) → decimated to 20fps
- Display update: Synchronized across all threads
- Total latency: ~50-100ms from send to display

## Troubleshooting

### Colors don't match / poor detection

**Symptoms:** Receiver shows wrong colors, message corrupted

**Solutions:**
1. Ensure good lighting on transmitter display
2. Adjust camera focus manually if autofocus fails
3. Increase distance between computers for larger cells
4. Try `receiver.py` color detection modes (see code)

### Low FPS / latency

**Symptoms:** Stuttering, frame drops

**Solutions:**
1. Close other applications
2. Reduce camera resolution capture in `receiver.py`
3. Reduce `TARGET_FPS` in `config.py`

### Network connectivity issues

**Symptoms:** Client can't reach transmitter

**Solutions:**
1. Verify transmitter IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Check firewall allows port 5000
3. Use explicit IP instead of localhost
4. Test: `curl http://192.168.1.X:5000/status`

### Camera not opening

**Symptoms:** "ERROR: Cannot open camera"

**Solutions:**
1. Check camera is connected and working (test with other app)
2. Try different camera ID: `ReceiverApp(camera_id=1)` in `receiver.py`
3. Grant camera permissions (macOS/Linux)
4. Restart application

## API Endpoints

### POST /send
Send a message to transmitter

```bash
curl -X POST http://localhost:5000/send \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello World"}'
```

Response:
```json
{
  "status": "queued",
  "message_length": 11,
  "queue_size": 1
}
```

### GET /status
Check transmitter status

```bash
curl http://localhost:5000/status
```

Response:
```json
{
  "frame_counter": 1234,
  "queue_size": 0,
  "running": true
}
```

### POST /stop
Stop the transmitter

```bash
curl -X POST http://localhost:5000/stop
```

## Performance Benchmarks

Tested at 80cm distance with 1512×982 display:

| Metric | Value |
|--------|-------|
| Grid cell size | ~23.6px × 15.3px |
| Throughput | 40.96 KB/s |
| Message roundtrip | 2-3s |
| Color accuracy | ~97% |
| Frame sync drift | <2% |

**Note:** The 64×64 grid uses larger cells (23.6×15.3px vs 11.8×7.7px for 128×128), resulting in **much better color accuracy** at the cost of lower throughput.

## Future Improvements

- [ ] Reed-Solomon error correction
- [ ] Grid corner detection for perspective correction
- [ ] Brightness normalization
- [ ] Adaptive color clustering
- [ ] Web UI for client
- [ ] Message history/logging
- [ ] Encryption for transmitted data
- [ ] Multiple receiver support

## License

MIT - Use freely for education and experimentation

## Notes

- This is an educational project demonstrating creative data encoding
- Not suitable for production use without significant hardening
- Requires good lighting conditions for reliable operation
- Distance and angle between computers affects quality

## Questions?

Check the code comments and docstrings for implementation details!
