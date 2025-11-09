# Camera Communication Clock Test

This directory contains a clock synchronization test for camera-based distributed training communication.

## Overview

The test implements an 11-second cycle with precise clock synchronization:

1. **Compute Phase 1** (0.5s): Root generates test tensor
2. **Transmit Phase 1** (5s): Root transmits tensor to worker via camera
3. **Compute Phase 2** (0.5s): Worker processes tensor
4. **Transmit Phase 2** (5s): Worker transmits response back to root

## Files

- `test_utils.py` - Core utilities for tensor conversion, clock sync, and phase management
- `clock_test_root.py` - Root node implementation
- `clock_test_worker.py` - Worker node implementation
- `test_codec_conversion.py` - Unit tests for tensor encoding/decoding

## Running the Test

### Prerequisites

1. Two computers with cameras, positioned to see each other's screens
2. Python environment with required packages (cv2, numpy)

### Test Procedure

1. **Verify codec conversion** (optional):
   ```bash
   python test_codec_conversion.py
   ```

2. **Start both nodes**:
   - On root computer: `python clock_test_root.py`
   - On worker computer: `python clock_test_worker.py`

3. **Calibrate cameras**:
   - Follow the calibration instructions on each screen
   - Ensure good lighting and clear view of the transmission window

4. **Begin synchronized test**:
   - Press SPACE on both computers at approximately the same time
   - The system will automatically sync to the next 11-second boundary

5. **Monitor progress**:
   - Watch the console output for cycle status and timing accuracy
   - The test runs for 10 complete cycles (110 seconds)
   - **Visual overlay**: The top-right corner of the camera window shows:
     - Node type (ROOT or WORKER)
     - Current cycle number
     - Current phase (e.g., "Transmit to Worker", "Worker Compute")
     - Time remaining in current phase
     - Frame progress during transmission/reception

## Visual Status Display

The camera window shows real-time status in the top-right corner:

```
ROOT | Cycle 3/10 | Transmit [5/9] | 2.3s
```

This indicates:
- **ROOT**: Node type (ROOT or WORKER)
- **Cycle 3/10**: Current cycle out of total cycles
- **Transmit [5/9]**: Current phase and frame progress
- **2.3s**: Time remaining in current phase

## Technical Details

- **Data size**: 670 float16 values (matching model parameter count)
- **Encoding**: K=8 (3-bit values), 6 grids per float16
- **Transmission**: 9 frames per tensor, 150ms per frame
- **Frame size**: 15×30 pixels (450 pixels total)

## Success Criteria

- Zero data corruption over 10+ cycles
- Clock drift < 100ms between nodes
- All frames transmitted within time windows
- Consistent phase transitions

## Troubleshooting

1. **Calibration fails**: Ensure good lighting, adjust camera angle
2. **Poor reception**: Check for reflections, adjust screen brightness
3. **Clock drift**: Ensure both computers have accurate system time
4. **Frame drops**: Reduce CPU load from other applications

## Next Steps

Once the clock test passes consistently:
1. Replace test tensors with actual model parameters/gradients
2. Add gradient computation in worker phase
3. Add parameter updates in root phase
4. Scale to multiple workers
