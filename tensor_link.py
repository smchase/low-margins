import time
import numpy as np
import torch
from camera import Camera, Frame
from codec import codec


class TensorLink:
    def __init__(self, camera: Camera, frame_delay_ms: int = 100):
        self.camera = camera
        self.frame_delay_ms = frame_delay_ms
        
        # Special sync patterns (using values beyond binary 0/1 for detection)
        # Since camera supports 0-7, we use higher values for control signals
        self.START_PATTERN = 7  # All 7s = START signal
        self.END_PATTERN = 6    # All 6s = END signal
        
        print(f"TensorLink initialized")
    
    def _send_sync_frame(self, pattern_value: int, duration_ms: int = 500) -> None:
        """Send a sync frame (START or END pattern) for specified duration."""
        sync_frame = Frame(data=np.full((16, 16), pattern_value, dtype=np.int64))
        
        # Send sync pattern for duration to ensure it's captured
        start_time = time.time()
        while (time.time() - start_time) * 1000 < duration_ms:
            self.camera.send(sync_frame)
            self.camera.update()
            time.sleep(0.03)  # ~30fps
    
    def _wait_for_sync_frame(self, pattern_value: int, timeout_sec: int = 60) -> bool:
        """Wait for a sync frame (START or END). Returns True if detected, False on timeout."""
        print(f"Waiting for sync signal (pattern={pattern_value})...")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_sec:
            frame = self.camera.receive()
            
            # Check if this is the sync pattern (most cells match the pattern)
            matches = np.sum(frame.data == pattern_value)
            total_cells = frame.data.size
            
            # If >90% of cells match the pattern, it's a sync frame
            if matches > (0.9 * total_cells):
                print(f"✓ Sync signal detected!")
                return True
            
            # Update display to keep responsive
            self.camera.update()
            time.sleep(0.03)
        
        print(f"✗ Timeout waiting for sync signal")
        return False
    
    def _wait_for_sync_end(self, pattern_value: int, timeout_sec: int = 10) -> bool:
        """Wait for sync frame to disappear (transition away from pattern). Returns True if transitioned."""
        print(f"Waiting for sync signal to end (pattern={pattern_value})...")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_sec:
            frame = self.camera.receive()
            
            # Check if frame is NOT the sync pattern anymore
            matches = np.sum(frame.data == pattern_value)
            total_cells = frame.data.size
            
            # If <50% of cells match, sync has ended
            if matches < (0.5 * total_cells):
                print(f"✓ Sync signal ended, ready for data!")
                return True
            
            # Update display to keep responsive
            self.camera.update()
            time.sleep(0.03)
        
        print(f"✗ Timeout waiting for sync to end")
        return False
    
    def send_tensor(self, tensor: torch.Tensor) -> None:
        if len(tensor.shape) != 2:
            raise ValueError(f"Tensor must be 2D, got shape {tensor.shape}")
        
        rows, cols = tensor.shape
        
        # Convert to numpy and handle dtype
        # For bfloat16, we preserve raw bits by viewing as uint16, then as float16
        # For other types, convert to float16
        if tensor.dtype == torch.bfloat16:
            # View bfloat16 as uint16 (preserves raw bits), then view as float16 for codec
            tensor_np = tensor.detach().cpu().view(torch.int16).numpy().view(np.uint16).view(np.float16)
        else:
            # Convert to float16 normally
            tensor_np = tensor.detach().cpu().numpy().astype(np.float16)
        
        # Create codec for this tensor's size
        c = codec(rows=rows, cols=cols, min_val=0, max_val=1)
        
        # Encode entire tensor to binary grids
        grids = c.encode(tensor_np)
        total_frames = grids.shape[0]
        
        print(f"Sending {rows}×{cols} tensor:")
        print(f"  Total frames: {total_frames}")
        
        # HANDSHAKE: Send START signal
        print("Sending START signal...")
        self._send_sync_frame(self.START_PATTERN, duration_ms=2000)
        time.sleep(0.3)  # Brief pause after START
        
        # Send each grid as a camera frame
        for i in range(total_frames):
            grid = grids[i].astype(np.int64)
            frame = Frame(data=grid)
            self.camera.send(frame)
            self.camera.update()
            
            if i < total_frames - 1:
                time.sleep(self.frame_delay_ms / 1000.0)
            
            if (i + 1) % 10 == 0 or i == total_frames - 1:
                print(f"  Sent frame {i + 1}/{total_frames}")
        
        # HANDSHAKE: Send END signal
        print("Sending END signal...")
        self._send_sync_frame(self.END_PATTERN, duration_ms=2000)
        
        print("✓ Tensor transmission complete!")
    
    def receive_tensor(self, rows: int, cols: int, dtype: torch.dtype = torch.bfloat16, timeout_sec: int = 60) -> torch.Tensor:
        """
        Receive a PyTorch tensor via camera over multiple frames.
        Waits for START signal before beginning reception.
        
        Args:
            rows: Number of rows in the tensor
            cols: Number of columns in the tensor
            dtype: Target dtype (default: torch.bfloat16)
            timeout_sec: How long to wait for START signal (default: 60s)
            
        Returns:
            Tensor of shape (rows, cols) with specified dtype
        """
        # Create codec for this tensor's size
        c = codec(rows=rows, cols=cols, min_val=0, max_val=1)
        total_frames = c.grids_needed()
        
        print(f"Receiving {rows}×{cols} tensor:")
        print(f"  Total frames: {total_frames}")
        
        # HANDSHAKE: Wait for START signal
        if not self._wait_for_sync_frame(self.START_PATTERN, timeout_sec=timeout_sec):
            raise TimeoutError("Failed to receive START signal")
        
        # CRITICAL: Wait for START signal to END (transition away)
        # This ensures sender has stopped sending START and is ready to send data
        if not self._wait_for_sync_end(self.START_PATTERN, timeout_sec=5):
            raise TimeoutError("START signal never ended")
        
        time.sleep(0.2)  # Brief pause after START ends, before reading data
        
        # STEP 1: Capture ALL frames until END signal
        print("Capturing all frames until END signal...")
        all_frames = []
        capture_start = time.time()
        
        while True:
            frame = self.camera.receive()
            
            # Check if this is END signal
            matches = np.sum(frame.data == self.END_PATTERN)
            total_cells = frame.data.size
            
            if matches > (0.9 * total_cells):
                print(f"✓ END signal detected after {len(all_frames)} captures!")
                break
            
            # Store frame
            all_frames.append(frame.data.copy())
            
            # Timeout check (safety)
            if (time.time() - capture_start) > 300:  # 5 min timeout
                raise TimeoutError("Timeout waiting for END signal")
            
            # Brief delay between captures
            time.sleep(0.01)
            self.camera.update()
        
        print(f"Captured {len(all_frames)} total frames")
        
        # STEP 2: Deduplicate consecutive identical frames
        print("Deduplicating consecutive frames...")
        unique_frames = []
        prev = None
        
        for frame in all_frames:
            if prev is None:
                # First frame
                unique_frames.append(frame)
                prev = frame
            else:
                # Check if different from previous
                if not np.array_equal(frame, prev):
                    unique_frames.append(frame)
                    prev = frame
                # else: skip duplicate
        
        print(f"Deduplicated to {len(unique_frames)} unique frames")
        
        # STEP 3: Verify we have the right number of frames
        if len(unique_frames) != total_frames:
            print(f"⚠ Warning: Expected {total_frames} frames, got {len(unique_frames)}")
            print(f"  This might cause decoding errors!")
        
        # STEP 4: Decode frames into tensor
        print("Decoding frames into tensor...")
        
        # Collect all frames and threshold to binary
        received_grids = []
        for i in range(total_frames):
            if i >= len(unique_frames):
                raise ValueError(f"Ran out of frames! Expected {total_frames}, got {len(unique_frames)}")
            
            frame_data = unique_frames[i]
            
            # Threshold to binary: 0-3 → 0, 4-7 → 1
            binary_data = (frame_data >= 4).astype(np.int64)
            received_grids.append(binary_data)
        
        # Decode all grids at once
        grids_array = np.array(received_grids)
        tensor_np = c.decode(grids_array)
        
        # Convert to PyTorch tensor with proper dtype
        if dtype == torch.bfloat16:
            # View float16 as uint16, then as int16, then interpret as bfloat16
            tensor = torch.from_numpy(tensor_np.view(np.uint16).view(np.int16)).view(torch.bfloat16)
        else:
            # Normal conversion for float16 or other types
            tensor = torch.from_numpy(tensor_np)
            if dtype != torch.float16:
                tensor = tensor.to(dtype)
        
        print("✓ Tensor reception complete!")
        return tensor


if __name__ == "__main__":
    """
    Example usage demonstrating tensor transmission.
    
    Usage:
        Computer A (sender):
            python tensor_link.py
            # Follow calibration instructions (press T, then R)
            # Press 'S' to send a random tensor
        
        Computer B (receiver):
            python tensor_link.py
            # Follow calibration instructions (press R, then T)
            # Press 'R' to receive the tensor
    """
    print("=" * 70)
    print("TENSOR LINK - Example Usage")
    print("=" * 70)
    

    # Create camera
    cam = Camera(test_mode=False)
    
    # Run calibration
    print("\n1. Running calibration...")
    if not cam.calibrate():
        print("Calibration cancelled")
        cam.cleanup()
        exit(0)
    
    # Create tensor link
    link = TensorLink(cam)
    
    print("\n2. Calibration complete! Ready for tensor transmission.")
    print("   Press 'S' to send a random tensor")
    print("   Press 'R' to receive a tensor")
    print("   Press 'Q' to quit\n")
    
    # Event loop
    while True:
        key = cam.update()
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            # Send a random tensor (try different sizes!)
            print("\n" + "=" * 70)
            torch.manual_seed(5828)
            # Example: send a 32×32 bfloat16 tensor
            test_tensor = torch.randn(32, 32, dtype=torch.bfloat16)
            print(f"Sending test tensor:")
            print(f"  Shape: {test_tensor.shape}")
            print(f"  Dtype: {test_tensor.dtype}")
            print(f"  Mean: {test_tensor.mean().item():.4f}")
            print(f"  Std: {test_tensor.std().item():.4f}")
            
            link.send_tensor(test_tensor)
            
            print("=" * 70 + "\n")
        elif key == ord('r') or key == ord('R'):
            # Receive a tensor
            # NOTE: You need to know the shape and dtype in advance!
            # In a real system, you'd transmit shape metadata first
            print("\n" + "=" * 70)
            print("Receiving tensor...")
            
            # Must match what sender sent
            received = link.receive_tensor(rows=32, cols=32, dtype=torch.bfloat16)
            
            print(f"Received tensor:")
            print(f"  Shape: {received.shape}")
            print(f"  Dtype: {received.dtype}")
            print(f"  Mean: {received.mean().item():.4f}")
            print(f"  Std: {received.std().item():.4f}")
            print("=" * 70 + "\n")
    
    # Cleanup
    print("\nShutting down...")
    cam.cleanup()

