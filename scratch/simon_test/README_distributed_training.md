# Distributed Training with Camera Communication

This implements distributed neural network training using camera-based communication instead of network sockets.

## System Overview

The system uses a 5-state cycle that repeats every 8 seconds:

1. **COMPUTE_G** (0.5s): Both nodes compute gradients on their data
2. **COMMUNICATE_G** (3s): Worker sends gradients to root via camera
3. **COMPUTE_THETA** (0.5s): Root averages gradients and updates parameters
4. **COMMUNICATE_THETA** (3s): Root sends new parameters to worker via camera
5. **COMPUTE_EVAL** (1s): Root evaluates model performance

## Architecture

- **Root Node**: 
  - Trains on first half of MNIST dataset
  - Collects gradients from worker
  - Computes parameter updates using Adam optimizer
  - Sends updated parameters to worker
  - Evaluates model accuracy

- **Worker Node**:
  - Trains on second half of MNIST dataset
  - Sends gradients to root
  - Receives updated parameters from root

## Camera Communication

- Model has 670 float16 parameters
- Each frame can hold 466 float16 values (2800 pixels ÷ 6 pixels per float)
- Requires 2 frames to transmit parameters or gradients
- Each frame displays for 1 second

## Running the System

### Prerequisites
1. Two computers with cameras positioned to see each other's screens
2. PyTorch and dependencies installed
3. Good lighting conditions

### Terminal 1 (Root):
```bash
python simon_test/distributed_root.py --num-steps 20 --learning-rate 0.01
```

### Terminal 2 (Worker):
```bash
python simon_test/distributed_worker.py --num-steps 20 --learning-rate 0.01
```

### Starting Training:
1. Run both commands
2. Complete camera calibration on both nodes
3. Press SPACE on both terminals at approximately the same time
4. System will automatically synchronize to the next cycle boundary

## Expected Output

Root will show:
```
[12:34:56.789] Step 1: Computing gradients (state: compute_g, t=0.2s)
[12:34:57.290] Step 1: Receiving gradients from worker (state: communicate_g, t=0.0s)
[12:34:57.340]   Received frame 1/2
[12:35:00.290] Step 1: Computing parameter update (state: compute_theta, t=0.0s)
[12:35:00.790] Step 1: Transmitting updated parameters (state: communicate_theta, t=0.0s)
[12:35:03.790] Step 1: Evaluating model (state: compute_eval, t=0.0s)
  Step 1/20: Accuracy=15.2%, Avg Loss=2.301
```

Worker will show:
```
[12:34:56.789] Step 1: Computing gradients (state: compute_g, t=0.2s)
  Computed gradients with norm: 0.5432, loss: 2.298
[12:34:57.290] Step 1: Transmitting gradients to root (state: communicate_g, t=0.0s)
[12:35:00.290] Step 1: Root computing parameter update (state: compute_theta, t=0.0s)
[12:35:00.790] Step 1: Receiving updated parameters (state: communicate_theta, t=0.0s)
```

## Training Progress

You should see:
- Accuracy gradually increasing on root node
- Loss decreasing on both nodes
- Successful frame transmissions (2 frames per communication phase)
- Both nodes staying synchronized through the cycles

## Final Results

After training:
- Root saves model to `distributed_model_root.pth`
- Worker saves model to `distributed_model_worker.pth`
- Both models should have identical parameters (verify with a comparison script)

## Troubleshooting

1. **Poor accuracy**: Ensure cameras are well-aligned and calibrated
2. **Frame errors**: Check lighting conditions, reduce screen glare
3. **Timing issues**: Both nodes must start within the same cycle
4. **Reception failures**: Verify both screens show transmission window clearly

## Technical Details

- Uses same codec as tensor verification tests (K=8, 3-bit encoding)
- Implements gradient averaging across nodes
- Uses Adam optimizer with gradient clipping
- Splits MNIST dataset 50/50 between nodes
