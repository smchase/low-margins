# 🌊 Scrolling Barcode Data Transmission

Transmit data between laptops using webcams pointed at screens with **Matrix-style scrolling colored barcodes**.

## 🚀 Quick Start

### On Laptop 1 (Transmitter):
```bash
# Transmit a message
python barcode_encode.py "Hello from the Matrix!"

# Press 'f' for fullscreen
# Press 'q' to quit
```

### On Laptop 2 (Receiver):
```bash
# Start receiving
python barcode_decode.py

# Point camera at the transmitting screen
# Press 'q' to quit, SPACE to save current buffer
```

## 📊 Performance

- **Throughput**: ~7-72 KB/sec (depending on bar height and camera)
- **Bar encoding**: 16 colors = 4 bits per bar
- **FPS**: 30 frames per second
- **Sync patterns**: Automatic frame detection

## 🎮 Keyboard Controls

### Encoder (barcode_encode.py):
- `f` - Toggle fullscreen
- `SPACE` - Pause/resume
- `q` or `ESC` - Quit

### Decoder (barcode_decode.py):
- `SPACE` - Capture and save current buffer
- `q` or `ESC` - Quit

## 🎨 How It Works

1. **Encoding**: Data is converted to 4-bit values (0-15), each mapped to a distinct color
2. **Sync Pattern**: Red-White alternating pattern marks data boundaries
3. **Scrolling**: Horizontal color bars scroll down the screen continuously
4. **Decoding**: Camera scans horizontal lines, matches colors, reconstructs data

## 🔧 Parameters

### Encoder:
- `bar_height`: Height of each bar in pixels (default: 4) - smaller = more throughput
- `fps`: Frames per second (default: 30)
- `screen_width/height`: Display resolution (default: 1920x1080)

### Decoder:
- `num_scan_lines`: Number of horizontal lines to sample (default: 5)
- `num_samples`: Samples per line (default: 100)

## 💡 Tips for Best Results

1. **Maximize brightness** on transmitting screen
2. **Reduce ambient lighting** for better color detection
3. **Position camera directly** facing screen (not at angle)
4. **Increase bar_height** (e.g., 8-10px) if colors aren't detected well
5. **Fullscreen mode** (`f` key) gives best results
6. Watch the **green scan lines** in decoder to ensure good alignment

## 🆚 Comparison to Static Grid

| Feature | Static Grid | Scrolling Barcode |
|---------|------------|-------------------|
| Throughput | ~2 KB/frame | ~7-72 KB/sec continuous |
| Setup | Manual capture | Continuous stream |
| Cool factor | 3/10 | 11/10 Matrix vibes |

## 📝 Example Data Streams

```bash
# Stream a file
python barcode_encode.py "$(cat message.txt)"

# Stream lots of data (will loop)
python barcode_encode.py "$(head -c 1000 /dev/urandom | base64)"

# Custom parameters for max speed
# Edit barcode_encode.py and change:
#   bar_height=2  # Thinner bars = more throughput
#   fps=60        # Higher framerate if your monitor supports it
```

## 🐛 Troubleshooting

**Colors not detected?**
- Increase `bar_height` in encoder
- Check camera exposure/white balance
- Reduce ambient light

**Getting garbage data?**
- Ensure camera can see full screen width
- Reduce scroll speed (increase `bar_height`)
- Check for SYNC patterns in decoder output

**Low throughput?**
- Decrease `bar_height` (minimum ~2px)
- Increase FPS if supported
- Use fullscreen mode
- Improve lighting conditions

Enjoy your cyberpunk data tunnel! 🚀

