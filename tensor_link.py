import time
import numpy as np
import torch
from camera import Camera, Frame
from codec import codec


class TensorLink:
    def __init__(self, camera: Camera, frame_delay_ms: int = 100):
        self.camera = camera
        self.frame_delay_ms = frame_delay_ms
        
        self.tile_size = 16
        self.codec = codec(rows=self.tile_size, cols=self.tile_size, min_val=0, max_val=1)
        
        # Special sync patterns (using values beyond binary 0/1 for detection)
        # Since camera supports 0-7, we use higher values for control signals
        self.START_PATTERN = 7  # All 7s = START signal
        self.END_PATTERN = 6    # All 6s = END signal
        
        print(f"TensorLink initialized: {self.codec.grids_needed()} frames per 16×16 tile")
    
    def grids_needed_per_tile(self) -> int:
        return self.codec.grids_needed()
    
    def _calculate_num_tiles(self, rows: int, cols: int) -> tuple:
        tiles_rows = (rows + self.tile_size - 1) // self.tile_size
        tiles_cols = (cols + self.tile_size - 1) // self.tile_size
        return tiles_rows, tiles_cols
    
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
    
    def send_tensor(self, tensor: torch.Tensor) -> None:
        if len(tensor.shape) != 2:
            raise ValueError(f"Tensor must be 2D, got shape {tensor.shape}")
        
        rows, cols = tensor.shape
        tiles_rows, tiles_cols = self._calculate_num_tiles(rows, cols)
        
        # Store original dtype to preserve it
        self._last_sent_dtype = tensor.dtype
        
        # Convert to numpy and handle dtype
        # For bfloat16, we preserve raw bits by viewing as uint16, then as float16
        # For other types, convert to float16
        if tensor.dtype == torch.bfloat16:
            # View bfloat16 as uint16 (preserves raw bits), then view as float16 for codec
            tensor_np = tensor.detach().cpu().view(torch.int16).numpy().view(np.uint16).view(np.float16)
        else:
            # Convert to float16 normally
            tensor_np = tensor.detach().cpu().numpy().astype(np.float16)
        
        # Pad to multiple of tile_size
        padded_rows = tiles_rows * self.tile_size
        padded_cols = tiles_cols * self.tile_size
        
        if rows < padded_rows or cols < padded_cols:
            padded = np.zeros((padded_rows, padded_cols), dtype=np.float16)
            padded[:rows, :cols] = tensor_np
            tensor_np = padded
        
        # Calculate total frames
        total_tiles = tiles_rows * tiles_cols
        frames_per_tile = self.codec.grids_needed()
        total_frames = total_tiles * frames_per_tile
        
        print(f"Sending {rows}×{cols} tensor:")
        print(f"  Tiles: {tiles_rows}×{tiles_cols} = {total_tiles} tiles")
        print(f"  Total frames: {total_frames}")
        
        # HANDSHAKE: Send START signal
        print("Sending START signal...")
        self._send_sync_frame(self.START_PATTERN, duration_ms=2000)
        time.sleep(0.3)  # Brief pause after START
        
        frame_count = 0
        
        # Process each tile
        for tile_row in range(tiles_rows):
            for tile_col in range(tiles_cols):
                # Extract tile
                r_start = tile_row * self.tile_size
                r_end = r_start + self.tile_size
                c_start = tile_col * self.tile_size
                c_end = c_start + self.tile_size
                
                tile = tensor_np[r_start:r_end, c_start:c_end]
                
                # Encode tile to binary grids
                grids = self.codec.encode(tile)
                
                # Send each grid as a camera frame
                for i in range(grids.shape[0]):
                    grid = grids[i].astype(np.int64)
                    frame = Frame(data=grid)
                    self.camera.send(frame)
                    self.camera.update()
                    
                    frame_count += 1
                    
                    if frame_count < total_frames:
                        time.sleep(self.frame_delay_ms / 1000.0)
                    
                    if frame_count % 10 == 0 or frame_count == total_frames:
                        print(f"  Sent frame {frame_count}/{total_frames}")
        
        # HANDSHAKE: Send END signal
        print("Sending END signal...")
        self._send_sync_frame(self.END_PATTERN, duration_ms=2000)
        
        print("✓ Tensor transmission complete!")
    
    def receive_tensor(self, rows: int, cols: int, dtype: torch.dtype = torch.bfloat16, timeout_sec: int = 60) -> torch.Tensor:
        """
        Receive a PyTorch tensor via camera over multiple frames.
        Waits for START signal before beginning reception.
        
        Args:
            rows: Original number of rows in the tensor
            cols: Original number of columns in the tensor
            dtype: Target dtype (default: torch.bfloat16)
            timeout_sec: How long to wait for START signal (default: 60s)
            
        Returns:
            Tensor of shape (rows, cols) with specified dtype
        """
        tiles_rows, tiles_cols = self._calculate_num_tiles(rows, cols)
        total_tiles = tiles_rows * tiles_cols
        frames_per_tile = self.codec.grids_needed()
        total_frames = total_tiles * frames_per_tile
        
        print(f"Receiving {rows}×{cols} tensor:")
        print(f"  Tiles: {tiles_rows}×{tiles_cols} = {total_tiles} tiles")
        print(f"  Total frames: {total_frames}")
        
        # HANDSHAKE: Wait for START signal
        if not self._wait_for_sync_frame(self.START_PATTERN, timeout_sec=timeout_sec):
            raise TimeoutError("Failed to receive START signal")
        
        time.sleep(0.3)  # Brief pause after detecting START
        
        # Padded dimensions
        padded_rows = tiles_rows * self.tile_size
        padded_cols = tiles_cols * self.tile_size
        
        # Reconstruct tensor from tiles
        reconstructed = np.zeros((padded_rows, padded_cols), dtype=np.float16)
        
        frame_count = 0
        
        for tile_row in range(tiles_rows):
            for tile_col in range(tiles_cols):
                # Receive frames for this tile
                received_grids = []
                for i in range(frames_per_tile):
                    frame = self.camera.receive()
                    
                    # Threshold to binary: 0-3 → 0, 4-7 → 1
                    # This handles noise and ensures values are in codec range [0, 1]
                    binary_data = (frame.data >= 4).astype(np.int64)
                    received_grids.append(binary_data)
                    
                    frame_count += 1
                    
                    if frame_count % 10 == 0 or frame_count == total_frames:
                        print(f"  Received frame {frame_count}/{total_frames}")
                
                # Decode tile
                grids_array = np.array(received_grids)
                tile = self.codec.decode(grids_array)
                
                # Place tile in reconstructed tensor
                r_start = tile_row * self.tile_size
                r_end = r_start + self.tile_size
                c_start = tile_col * self.tile_size
                c_end = c_start + self.tile_size
                
                reconstructed[r_start:r_end, c_start:c_end] = tile
        
        # Crop to original size
        tensor_np = reconstructed[:rows, :cols]
        
        # Convert to PyTorch tensor with proper dtype
        if dtype == torch.bfloat16:
            # View float16 as uint16, then as int16, then interpret as bfloat16
            tensor = torch.from_numpy(tensor_np.view(np.uint16).view(np.int16)).view(torch.bfloat16)
        else:
            # Normal conversion for float16 or other types
            tensor = torch.from_numpy(tensor_np)
            if dtype != torch.float16:
                tensor = tensor.to(dtype)
        
        # HANDSHAKE: Wait for END signal (optional, for confirmation)
        print("Waiting for END signal...")
        if self._wait_for_sync_frame(self.END_PATTERN, timeout_sec=10):
            print("✓ END signal confirmed")
        else:
            print("⚠ END signal not detected (might be ok)")
        
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

