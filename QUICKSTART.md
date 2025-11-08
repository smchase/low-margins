# Quick Start Guide

## Installation

You already have everything installed! The project is set up with:
- Python 3.13.7
- uv package manager
- FFmpeg 8.0
- All dependencies (numpy, opencv-python)

## Testing the Application

### 1. Test Encoder/Decoder (No Network Required)

```bash
uv run python test_encoder.py
```

This verifies that data can be encoded into video frames and decoded back correctly.

### 2. Test Network Streaming (Two Terminal Windows)

You'll need two terminal windows to test the bidirectional communication.

**Terminal 1 (Listener):**
```bash
uv run python -m video_tunnel listen 127.0.0.1
```

**Terminal 2 (Connector):**
```bash
uv run python -m video_tunnel connect 127.0.0.1
```

Now you can type messages in either terminal and they'll be transmitted as video to the other side!

### 3. Test File Transfer

**Terminal 1 (Receive file):**
```bash
uv run python -m video_tunnel listen 127.0.0.1 > received.txt
```

**Terminal 2 (Send file):**
```bash
cat README.md | uv run python -m video_tunnel connect 127.0.0.1
```

After sending, press Ctrl+C in both terminals and check `received.txt`.

## Using on Two Different Computers

### Step 1: Find Your IP Addresses

On both computers, run:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Let's say:
- Computer A: `192.168.1.10`
- Computer B: `192.168.1.20`

### Step 2: Start the Tunnel

**On Computer A (192.168.1.10):**
```bash
uv run python -m video_tunnel connect 192.168.1.20
```

**On Computer B (192.168.1.20):**
```bash
uv run python -m video_tunnel listen 192.168.1.10
```

### Step 3: Communicate!

Now both computers can send and receive data. Type messages, pipe files, or stream data!

## How It Works

1. Your data is split into chunks
2. Each chunk is encoded into RGB pixel values of video frames
3. FFmpeg compresses the frames using H.264 codec
4. Video is streamed over UDP to the other computer
5. The receiver decodes frames back into binary data

With default settings (640x480 @ 30fps), each frame can carry ~900KB of data, giving theoretical throughput of ~27 MB/s before video compression.

## Troubleshooting

### Firewall Issues
Make sure UDP ports 5000 and 5001 are open on both machines.

### Connection Refused
- Verify both computers are on the same WiFi network
- Check IP addresses are correct
- Try pinging the other machine: `ping <other-ip>`

### Poor Performance
- Lower resolution: `--width 320 --height 240`
- Reduce frame rate: `--fps 15`
- Check WiFi signal strength

## Advanced Usage

### Custom Ports
```bash
uv run python -m video_tunnel connect 192.168.1.20 --send-port 6000 --recv-port 6001
```

### Higher Quality (More Data Per Frame)
```bash
uv run python -m video_tunnel connect 192.168.1.20 --width 1280 --height 720
```

Both sides must use matching settings!

## What's Next?

Try these experiments:
- Stream a video through the tunnel: `cat movie.mp4 | video-tunnel ...`
- Compress data first for more throughput: `tar czf - bigfolder/ | video-tunnel ...`
- Use with other tools: `nc -l 8080 | video-tunnel ...`

Have fun encoding data as video! 🎥📡
