# Tensor Verification Test

This test verifies that camera-based tensor transmission is working correctly with deterministic validation.

## What it Does

1. **Root generates** a deterministic tensor using seed 42
2. **Root sends** the tensor to worker (9 frames)
3. **Worker verifies** it received the exact tensor expected
4. **Worker transforms** the tensor (multiply by 2) and sends it back
5. **Root verifies** it received the correctly transformed tensor

Both sides know what to expect because they use deterministic generation.

## How to Run

### Terminal 1 (Root):
```bash
python tensor_verification_test.py root
```

### Terminal 2 (Worker):
```bash
python tensor_verification_test.py worker
```

Press SPACE in both terminals at the same time to start.

## What You'll See

### Success Case:
```
ROOT:
✓ SUCCESS: Received tensor matches expected transform!

WORKER:
✓ SUCCESS: Received exact tensor from root!
```

### Failure Case:
```
ROOT:
✗ FAILURE: 42/670 values differ
  Maximum error: 0.123456

WORKER:
✗ FAILURE: 35/670 values differ from expected
  Maximum error: 0.234567
```

## Test Details

- **Tensor size**: 670 float16 values (matching model parameters)
- **Encoding**: K=8 (3-bit values), 6 grids per float16
- **Frames**: 9 frames per tensor (75 values per frame)
- **Transform**: Multiply by 2.0 (with clipping to prevent overflow)

## Quick Local Test

To verify encoding/decoding works without cameras:
```bash
python test_tensor_roundtrip.py
```

This should show all tests passing before you run the camera test.
