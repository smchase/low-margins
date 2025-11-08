# Composite Camera - Real Camera + TX Grid Overlay

## Concept

Instead of using a fully virtual camera or needing two physical devices, the **composite camera** captures from your real camera but renders the TX grid **smaller and centered on top** of the camera feed.

This way:
- ✓ Single camera
- ✓ Real camera input
- ✓ TX grid visible on screen
- ✓ RX can calibrate and detect

---

## How It Works

```
Real Camera Feed
├─ User/environment in background
│
└─ TX Grid Overlaid on Top
   ├─ Smaller than camera frame (centered)
   ├─ Solid/opaque (fully visible, no blending)
   ├─ Green border around it for visibility
   └─ Auto-starts in calibration mode (flashing)
```

The receiver sees the composite image - real camera feed with grid on top.

---

## Visual Example

```
Camera sees:
┌─────────────────────────────────────┐
│                                     │
│   Your room / environment           │
│                                     │
│      ┌───────────────┐              │
│      │███████████████│              │
│      │███ TX Grid ███│ Solid/opaque
│      │███ (smaller)███│              │
│      └───────────────┘              │
│                                     │
└─────────────────────────────────────┘

RX locks onto the grid and decodes it!
```

---

## Usage

### Simple Run
```bash
python test_composite.py
```

Then:
1. You'll see your camera feed with TX grid overlaid (flashing)
2. Press `c` in the window to calibrate
3. After ~3 seconds, TX stops flashing and shows the actual message
4. RX detects and decodes: "HELLO WORLD" ✓

### With Parameters

```bash
# Smaller grid (less obstructing, harder to detect)
python test_composite.py --cell-size 30

# Larger grid (more visible, easier to detect)
python test_composite.py --cell-size 60

# Different grid size
python test_composite.py --grid-size 20
```

---

## Parameters

### `--grid-size` (default: 16)
Grid size (16x16, 20x20, etc.)

### `--cell-size` (default: 40)
Size of each grid cell in pixels
- Smaller (30) = less obstructing, harder to detect
- Larger (60) = more visible, easier to detect

---

## How to Test

### Step 1: Run
```bash
python test_composite.py
```

### Step 2: Watch
- You see your camera feed
- TX grid appears smaller in the middle
- Grid is flashing (calibration mode)

### Step 3: Calibrate
- Click the window
- Press `c`
- RX locks onto the grid (green outline appears in the window)

### Step 4: Decode
- After ~3 seconds, TX stops flashing
- Grid now shows the actual "HELLO WORLD" message
- RX detects and decodes it
- You see "DECODED: 'HELLO WORLD'" with "✓ MATCH!"

### Step 5: Exit
- Press `q` to quit

---

## Files

| File | Purpose |
|------|---------|
| `composite_camera.py` | Composite camera implementation |
| `test_composite.py` | Test script to run it |
| `visual_data_link.py` | Modified to accept camera (unchanged RX logic) |

---

## Implementation Details

### CompositeCamera Class

```python
cap = CompositeCamera(grid_size=16, cell_size=40)

# Implements cv2.VideoCapture interface:
ret, frame = cap.read()  # Real camera feed + TX grid
cap.set(prop_id, value)  # Pass through to real camera
cap.isOpened()           # Check if camera is open
cap.release()            # Release camera
```

### What `read()` Does

1. **Capture** from real camera: `camera.read()`
2. **Render** TX grid: `tx.render()`
3. **Convert** to BGR: `cvtColor(grid, GRAY2BGR)`
4. **Place** grid on top of camera frame (solid, no blending)
5. **Return**: Composite image

### Placement

The TX grid is placed directly on top of the camera frame with full opacity:

```python
# Direct placement - fully opaque, no blending
composite[y:y+grid_h, x:x+grid_w] = grid_bgr
```

This ensures the grid is solid and clearly visible without any semi-transparency.

---

## Advantages

✓ **Single camera** - No need for two devices
✓ **Real input** - Camera feed is real, not synthetic
✓ **Visual feedback** - You see the grid on your screen
✓ **Simple** - Just run the script
✓ **No threading** - Unlike virtual camera, no background thread needed
✓ **Same RX logic** - Uses exact same receiver code as real hardware

---

## Troubleshooting

### Camera not found
```
Error: Could not open camera
```
- Make sure camera is connected
- Try: `python test_composite.py`

### Grid not visible
- Increase size: `--cell-size 60`
- Make sure grid is in focus in camera

### Grid too visible (can't see camera)
- Decrease size: `--cell-size 30`
- Move camera so grid doesn't cover important area

### RX can't calibrate
- Grid might be too small
- Increase size: `--cell-size 60`
- Make sure grid is in focus in camera
- Try moving the camera closer to your monitor

### Slow/lag
- Reduce grid size: `--cell-size 30`
- This is normal - just compositing takes CPU

---

## Comparison of Approaches

| Feature | Pure Virtual | Composite Real | Real Hardware |
|---------|---|---|---|
| **Needs camera** | ✗ | ✓ | ✓✓ |
| **Real input** | ✗ | ✓ | ✓✓ |
| **Single device** | ✓ | ✓ | ✗ |
| **Realistic** | ✓✓ | ✓✓ | ✓✓✓ |
| **Background thread** | ✓ | ✗ | ✗ |
| **Complexity** | Medium | Low | Low |

**Choose Composite Real if:**
- You have one camera
- You want real input with minimal setup
- You don't want to manage threads

---

## Summary

**Composite camera = Real camera + TX grid overlay**

One command to test:
```bash
python test_composite.py
```

No threads, no virtual buffers, no two devices.
Just your camera with the TX grid rendered on top.

Simple. Clean. Effective.
