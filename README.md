# Video Tunnel

Stream arbitrary data between two computers on a local network by encoding it as video. This project uses FFmpeg to encode binary data into video frames, transmit them over UDP, and decode them back into the original data on the receiving end.

## Concept

Instead of traditional network protocols, this application:
1. Takes arbitrary binary data
2. Encodes it into RGB pixel values in video frames
3. Uses FFmpeg to compress and stream the video over UDP
4. Decodes the video frames back into binary data on the receiving end
5. Supports bidirectional communication

Each frame stores data in its RGB pixel values, with metadata in the first row for length and checksums.

## Requirements

- Python 3.10 or higher
- FFmpeg installed and available in PATH
- Two computers on the same local network (WiFi)
- `uv` for Python package management

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd low-margins
```

2. Install dependencies:

**With uv (recommended):**
```bash
uv pip install -e .
```

**Or with pip:**
```bash
python3 -m pip install -e .
```

**Or install dependencies directly:**
```bash
python3 -m pip install numpy opencv-python
```

## Usage

The application works in two modes: **connect** (initiator) and **listen** (responder). Both machines need to know each other's IP addresses.

### Finding Your IP Address

**macOS/Linux:**
```bash
ifconfig | grep "inet "
# or
ip addr show
```

**Windows:**
```cmd
ipconfig
```

Look for your local network IP (usually starts with 192.168.x.x or 10.x.x.x).

### Basic Usage

**Computer A (192.168.1.10):**
```bash
video-tunnel connect 192.168.1.20
```

**Computer B (192.168.1.20):**
```bash
video-tunnel listen 192.168.1.10
```

Now both computers can send and receive data through stdin/stdout. Type messages and press Enter to send.

### Pipe Data Through

**Send a file from A to B:**

Computer A:
```bash
cat myfile.txt | video-tunnel connect 192.168.1.20
```

Computer B:
```bash
video-tunnel listen 192.168.1.10 > received.txt
```

**Bidirectional file transfer:**

Computer A:
```bash
cat send.txt | video-tunnel connect 192.168.1.20 > received.txt
```

Computer B:
```bash
cat response.txt | video-tunnel listen 192.168.1.10 > received.txt
```

### Advanced Options

```bash
# Custom ports
video-tunnel connect 192.168.1.20 --send-port 6000 --recv-port 6001

# Higher quality video (more data per frame, requires more bandwidth)
video-tunnel connect 192.168.1.20 --width 1280 --height 720

# Adjust frame rate
video-tunnel connect 192.168.1.20 --fps 60
```

**Note:** Both sides must use the same width, height, and FPS settings!

## How It Works

### Data Encoding

1. Binary data is split into chunks (based on frame capacity)
2. Each chunk is encoded into a video frame:
   - First row stores metadata (data length, checksum)
   - Remaining pixels store data (3 bytes per pixel in RGB)
3. Frames are sent to FFmpeg for H.264 encoding

### Network Streaming

- FFmpeg encodes video with H.264 codec (ultrafast preset, zero latency)
- Video is streamed over UDP using MPEGTS container
- Low-latency configuration for near real-time transmission

### Data Decoding

1. FFmpeg receives and decodes video stream
2. Raw RGB frames are extracted
3. Metadata is read to determine data length
4. Checksum is verified
5. Binary data is reconstructed and output

### Frame Capacity

With default settings (640x480):
- Pixels per frame: 307,200
- Minus metadata row: 306,560 pixels
- Data capacity: 919,680 bytes (~900 KB per frame)

At 30 FPS, theoretical maximum throughput: ~27 MB/s (before video compression)

## Architecture

```
┌─────────────┐                           ┌─────────────┐
│  Computer A │                           │  Computer B │
├─────────────┤                           ├─────────────┤
│             │                           │             │
│  stdin ─────┼──┐                    ┌──┼───── stdout │
│             │  │                    │  │             │
│             │  ▼                    ▼  │             │
│   Encoder   │  Data                Data│   Decoder   │
│      │      │   │                    │ │      ▲      │
│      ▼      │   ▼                    │ │      │      │
│   FFmpeg ───┼──► Video ═══════════╗ │ │   FFmpeg    │
│             │   (UDP port 5000)   ║ └─┼──────┘      │
│             │                     ║   │             │
│   FFmpeg ◄──┼──┐ Video ═══════════╝   │  Encoder    │
│      │      │  │ (UDP port 5001)      │      │      │
│      ▼      │  │                      │      ▼      │
│   Decoder   │  └──────────────────────┼──── stdin   │
│      │      │                         │             │
│      ▼      │                         │             │
│  stdout ────┼──────────────────────────┼────────────►│
└─────────────┘                         └─────────────┘
```

## Troubleshooting

### "Connection refused" or "No data received"

- Ensure both machines are on the same network
- Check firewall settings allow UDP traffic on the specified ports
- Verify FFmpeg is installed: `ffmpeg -version`
- Try pinging the remote host: `ping <remote-ip>`

### Poor performance or data corruption

- Reduce resolution: `--width 320 --height 240`
- Lower frame rate: `--fps 15`
- Check network quality and bandwidth
- Ensure WiFi signal is strong on both machines

### "FFmpeg process has terminated"

- Check FFmpeg error output in terminal
- Try running with verbose FFmpeg logging (modify `stream_sender.py` to remove `-loglevel error`)
- Ensure sufficient network bandwidth for video streaming

## Limitations

- Both sides need to know each other's IP addresses
- UDP protocol means no automatic reliability (packets may be lost)
- Performance depends on network bandwidth and quality
- Video compression overhead reduces effective throughput
- Currently no encryption (data is encoded but not secured)

## Future Improvements

- Automatic peer discovery on local network
- Add forward error correction for packet loss
- Implement encryption for secure communication
- Support for different video codecs
- Adaptive bitrate based on network conditions
- GUI interface for easier usage

## License

MIT

## Contributing

Pull requests welcome! This is an experimental project exploring novel data transmission methods.
