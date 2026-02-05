# Camera Communicator for Distributed Training

This provides a drop-in replacement for `NetworkCommunicator` that uses camera-based communication instead of network sockets.

## Architecture

The `CameraCommunicator` class implements the exact same interface as `NetworkCommunicator`:
- `setup(num_workers)` - Initialize communication
- `send_gradients(gradients)` - Worker sends gradients to root
- `receive_gradients()` - Root receives gradients from workers
- `send_parameters(parameters)` - Root sends parameters to workers  
- `receive_parameters()` - Worker receives parameters from root
- `close()` - Clean up resources

## Usage with state_network_camera.py

The modified `state_network_camera.py` supports both communication types:

### Network Communication (Original)
```bash
# Terminal 1 - Root:
python state_network_camera.py --mode root --comm network --num-steps 30

# Terminal 2 - Worker:
python state_network_camera.py --mode worker --comm network --root-host localhost --num-steps 30
```

### Camera Communication (New)
```bash
# Terminal 1 - Root:
python state_network_camera.py --mode root --comm camera --num-steps 30

# Terminal 2 - Worker:
python state_network_camera.py --mode worker --comm camera --num-steps 30
```

**Important**: 
1. Complete calibration on both nodes
2. Press SPACE on both terminals at approximately the same time
3. The system will sync to the next COMPUTE_G phase automatically

## Key Differences

### Timing
- **Network**: 4-second cycles (fast communication)
- **Camera**: 8-second cycles (2 frames @ 1s each for gradients/parameters)

### Setup
- **Network**: Automatic connection via TCP/IP
- **Camera**: Requires calibration and manual synchronization

### Capacity
- **Network**: Unlimited data size
- **Camera**: 466 float16 values per frame (model needs 2 frames)

## Testing the Communicator

Test just the camera communicator standalone:

```bash
# Terminal 1:
python camera_communicator.py root

# Terminal 2:  
python camera_communicator.py worker
```

This will:
1. Complete camera calibration
2. Root sends test parameters (2 frames)
3. Worker receives parameters
4. Worker sends test gradients (2 frames)
5. Root receives gradients

## Integration Example

```python
# Your existing code using NetworkCommunicator:
from network import NetworkCommunicator
comm = NetworkCommunicator(is_root=True, root_host='localhost')

# Switch to camera by changing one line:
from camera_communicator import CameraCommunicator
comm = CameraCommunicator(is_root=True)

# Rest of the code stays exactly the same!
```

## Protocol Details

The camera communicator:
1. Automatically detects model size during first parameter send
2. Uses the efficient codec (K=8, 3-bit encoding) from tensor verification
3. Maintains proper frame timing (1s display per frame)
4. Handles tensor shapes automatically

## Limitations

- Currently supports only 1 worker (camera sees one screen at a time)
- Requires good camera alignment and lighting
- Slower than network communication due to frame timing
- Maximum ~466 float16 values per second

## Benefits

- No network configuration needed
- Works across air-gapped systems
- Visual confirmation of data transmission
- Uses proven tensor encoding from verification tests
