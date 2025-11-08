# Codec Composite Camera Test

Transmit a **748×64 FP16 tensor** through the visual data link using a real camera and the codec system with **8-color encoding** and **spatial slicing**.

## Quick Start

```bash
python test_codec_composite.py
```

## How It Works

1. **Creates test tensor**: Random 748×64 FP16 tensor
2. **Encodes with codec**: 4 grids of 748×64 values [0-15]
3. **Spatial slicing**: Each 748×64 grid → 12 slices of 64×64 (rows 0-63, 64-127, ..., 704-767)
4. **8-color encoding**: Each slice → 2 color frames (8 colors = 3 bits/cell)
   - Frame 0: Lower 3 bits (values 0-7) using 8 distinct colors
   - Frame 1: Upper bit (values 0-1) using 2 colors (black/red)
5. **Visual transmission**: 64×64 color grid per frame (cell_size=30px → 1920×1920 image)
6. **Receives with color detection**: Detects which of 8 colors each cell is
7. **Reconstructs slices**: Combines 2 color frames → 64×64 values [0-15]
8. **Reconstructs grids**: Stacks 12 slices → 748×64 grid
9. **Decodes and verifies**: Reconstructs tensor and checks lossless transmission

**Total transmission**: 4 grids × 12 slices × 2 frames = **96 frames** (~96 seconds at 1 FPS)

**Colors used**: Black, Red, Green, Yellow, Blue, Magenta, Cyan, White (8 corners of RGB cube)

## Controls

| Key | Action |
|-----|--------|
| `c` | **Calibrate** - Lock onto the grid |
| `p` | **Start transmission** - Begin transmitting first grid |
| `n` | **Next frame** - Capture current color frame and move to next |
| `f` | Toggle horizontal flip |
| `+/=` | Increase border offset |
| `-` | Decrease border offset |
| `q` | Quit |

## Workflow

### Step 1: Calibrate
- Start the test - you'll immediately see a flashing grid overlaid on your camera feed
- Point camera at the screen (showing the flashing grid)
- Press **'c'** to calibrate
- Green outline appears when locked ✓

### Step 2: Start Auto-Transmission
- Press **'p'** to start automatic transmission at 1 FPS
- Frames auto-advance every second
- Receiver auto-detects frame changes and captures them
- Status shows progress: "Grid 1/4, Frame 0/2 - AUTO TX"

### Step 3: Automatic Capture & Completion
- **No manual intervention needed!**
- Frames are automatically captured as they change
- After 96 seconds, all frames transmitted
- Green "DONE" frame appears
- Automatic decoding begins

### Step 4: Automatic Decoding
When done frame is detected:
```
======================================================================
✓ DONE FRAME DETECTED - DECODING...
======================================================================
Grid 1: reconstructed (748, 64), range [0, 15]
Grid 2: reconstructed (748, 64), range [0, 15]
Grid 3: reconstructed (748, 64), range [0, 15]
Grid 4: reconstructed (748, 64), range [0, 15]

Stacked grids: (4, 748, 64)
Decoded tensor: (748, 64), dtype=float16

======================================================================
✓✓✓ SUCCESS! Tensor transmitted losslessly!
======================================================================
Original sample: [ 0.9946  0.5654  0.9917 -1.344  -0.10864]
Decoded sample:  [ 0.9946  0.5654  0.9917 -1.344  -0.10864]
Match: True
```

## Technical Details

### Grid Size
- **Tensor**: 748×64 (47,872 FP16 values)
- **Visual grid**: 64×64 cells (fixed for camera readability)
- **Spatial slices**: 12 slices per codec grid (64 rows each, last slice padded to 768 rows)
- **Cell size**: 30 pixels → 1920×1920 pixel image
- Each cell shows one of 8 colors (3 bits per cell)

### Transmission Time
- **Calibration**: One-time (~3 seconds)
- **Data transmission**: 96 frames at 1 FPS (4 grids × 12 slices × 2 color frames)
- **Total**: ~99 seconds (~1.5 minutes)

### Codec Settings
```python
codec(rows=748, cols=64, min_val=0, max_val=15)
# K=16 (base-16 encoding)
# D=4 grids needed
# Each value: 4 bits
```

### 8-Color Encoding
Each value [0-15] is encoded across 2 frames:

**Frame 0** (Lower 3 bits, 8 colors):
```
Value 7 → 111 → White
Value 3 → 011 → Yellow
Value 0 → 000 → Black
```

**Frame 1** (Upper bit, 2 colors):
```
Bit 1 → Red
Bit 0 → Black
```

**Color Palette**:
- 0: Black (0,0,0)
- 1: Red (255,0,0)
- 2: Green (0,255,0)
- 3: Yellow (255,255,0)
- 4: Blue (0,0,255)
- 5: Magenta (255,0,255)
- 6: Cyan (0,255,255)
- 7: White (255,255,255)

## Troubleshooting

### "Calibration failed"
- Make sure grid is visible and flashing
- Grid might be too small - check camera can see the whole pattern
- Try adjusting lighting

### Grid too large for camera
- Script automatically scales to fit 90% of camera frame
- Original: 1024×1024 px (with 2px cells)

### Decoding errors
- Re-calibrate with 'c'
- Adjust border offset with +/- keys
- Make sure camera is stable when pressing 'n'

### "MISMATCH - Transmission had errors"
- Captured pattern has bit errors
- Try re-capturing affected grid
- Check lighting and camera stability

## Files

- [test_codec_composite.py](test_codec_composite.py) - Main test script
- [codec_composite_camera.py](codec_composite_camera.py) - Composite camera with codec
- [codec.py](codec.py) - FP16 ↔ integer grids encoder/decoder
- [visual_main.py](visual_main.py) - Receiver class (used for detection)

## See Also

- [CODEC_TRANSMISSION.md](CODEC_TRANSMISSION.md) - Full codec transmission documentation
- [COMPOSITE_CAMERA.md](COMPOSITE_CAMERA.md) - Composite camera approach
