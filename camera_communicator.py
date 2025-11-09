import time
import torch
import cv2
import numpy as np
from typing import List, Optional
from camera import Camera, Frame, ROWS, COLS, SECONDS_PER_FRAME, RECEIVE_OFFSET
from simon_test.distributed_utils import tensors_to_frames, frames_to_tensors, get_parameter_shapes, log_timing, FLOATS_PER_FRAME


class CameraCommunicator:
    """
    Camera-based communicator that implements the same interface as NetworkCommunicator.
    
    Root node transmits parameters and receives gradients.
    Worker nodes transmit gradients and receive parameters.
    """
    
    def __init__(self, is_root: bool, root_host: str = 'localhost', grad_port: int = 5000, param_port: int = 5001):
        """
        Initialize camera communicator.
        
        Args:
            is_root: True if this is the root node, False if worker
            root_host: Ignored for camera communication (kept for compatibility)
            grad_port: Ignored for camera communication
            param_port: Ignored for camera communication
        """
        self.is_root = is_root
        self.camera = None
        self.param_shapes = None
        self.num_frames = None
        self.calibration_done = False
        
        # For tracking state
        self.num_workers = None
        
    def setup(self, num_workers: Optional[int] = None):
        """
        Set up camera communication.
        
        Args:
            num_workers: Required for root node - number of workers to expect
        """
        if self.is_root and num_workers is None:
            raise ValueError("Root must specify num_workers")
        
        self.num_workers = num_workers if self.is_root else 1
        
        # Initialize camera
        self.camera = Camera()
        
        print(f"{'Root' if self.is_root else 'Worker'}: Setting up camera communication")
        print(f"{'Root' if self.is_root else 'Worker'}: Please complete calibration...")
        
        if not self.camera.calibrate():
            raise RuntimeError("Camera calibration failed")
        
        self.calibration_done = True
        print(f"{'Root' if self.is_root else 'Worker'}: Camera ready! Press SPACE to continue...")
        
        # Wait for user to press space
        while True:
            self.camera.update()
            if (cv2.waitKey(30) & 0xFF) == ord(' '):
                break
        
        print(f"{'Root' if self.is_root else 'Worker'}: Camera communication established!")
    
    def _transmit_frames(self, frames: List[Frame]):
        """Transmit frames with proper timing."""
        start_time = time.time()
        
        for i, frame in enumerate(frames):
            # Transmit frame
            self.camera.transmit(frame)
            
            # Keep displaying until next frame time
            next_frame_time = (i + 1) * SECONDS_PER_FRAME
            while (time.time() - start_time) < next_frame_time:
                self.camera.update()
                cv2.waitKey(10)
    
    def _receive_frames(self, num_frames: int) -> List[Frame]:
        """Receive frames with proper timing."""
        frames = []
        start_time = time.time()
        
        for i in range(num_frames):
            # Wait for frame to be displayed
            target_time = i * SECONDS_PER_FRAME + RECEIVE_OFFSET
            while (time.time() - start_time) < target_time:
                self.camera.update()
                cv2.waitKey(10)
            
            # Capture frame
            frame = self.camera.receive()
            frames.append(frame)
        
        return frames
    
    def send_gradients(self, gradients: List[torch.Tensor]):
        """
        Worker sends gradients to root.
        
        Args:
            gradients: List of gradient tensors
        """
        if self.is_root:
            raise RuntimeError("Root should not send gradients")
        
        # Get shapes if not already stored
        if self.param_shapes is None:
            self.param_shapes = [tuple(g.shape) if g is not None else None for g in gradients]
            # Also set num_frames for consistency
            total_params = sum(np.prod(shape) for shape in self.param_shapes if shape is not None)
            self.num_frames = (total_params + FLOATS_PER_FRAME - 1) // FLOATS_PER_FRAME
        
        # Convert to frames
        frames = tensors_to_frames(gradients)
        
        print(f"Worker: Transmitting {len(frames)} gradient frames")
        self._transmit_frames(frames)
    
    def receive_gradients(self) -> List[List[torch.Tensor]]:
        """
        Root receives gradients from all workers.
        
        Returns:
            List of gradient lists, one per worker
        """
        if not self.is_root:
            raise RuntimeError("Only root should receive gradients")
        
        all_gradients = []
        
        # For camera communication, we only support single worker for now
        if self.num_workers != 1:
            raise NotImplementedError("Camera communication currently supports only 1 worker")
        
        # If we don't know frame count yet, calculate it from model
        if self.num_frames is None:
            # Get model to determine parameter shapes and frame count
            from train.model import MLP
            model = MLP()
            self.param_shapes = [tuple(p.shape) for p in model.parameters()]
            total_params = sum(np.prod(shape) for shape in self.param_shapes)
            self.num_frames = (total_params + FLOATS_PER_FRAME - 1) // FLOATS_PER_FRAME
            print(f"Root: Detected model has {total_params} parameters, need {self.num_frames} frames")
        
        print(f"Root: Receiving {self.num_frames} gradient frames from worker")
        frames = self._receive_frames(self.num_frames)
        
        # Convert back to tensors
        gradients = frames_to_tensors(frames, self.param_shapes)
        all_gradients.append(gradients)
        
        return all_gradients
    
    def send_parameters(self, parameters: List[torch.Tensor]):
        """
        Root sends parameters to all workers.
        
        Args:
            parameters: List of parameter tensors
        """
        if not self.is_root:
            raise RuntimeError("Only root should send parameters")
        
        # Store shapes for later use
        if self.param_shapes is None:
            self.param_shapes = [tuple(p.shape) for p in parameters]
        
        # Convert to frames
        frames = tensors_to_frames(parameters)
        self.num_frames = len(frames)
        
        print(f"Root: Transmitting {len(frames)} parameter frames")
        self._transmit_frames(frames)
    
    def receive_parameters(self) -> List[torch.Tensor]:
        """
        Worker receives parameters from root.
        
        Returns:
            List of parameter tensors
        """
        if self.is_root:
            raise RuntimeError("Root should not receive parameters")
        
        # For first receive, we need to get shapes from the model
        if self.param_shapes is None:
            # Import here to avoid circular dependency
            from train.model import MLP
            model = MLP()
            self.param_shapes = [tuple(p.shape) for p in model.parameters()]
        
        # Determine number of frames based on model
        if self.num_frames is None:
            # Calculate based on total parameters
            total_params = sum(np.prod(shape) for shape in self.param_shapes)
            self.num_frames = (total_params + FLOATS_PER_FRAME - 1) // FLOATS_PER_FRAME
        
        print(f"Worker: Receiving {self.num_frames} parameter frames from root")
        frames = self._receive_frames(self.num_frames)
        
        # Convert back to tensors
        parameters = frames_to_tensors(frames, self.param_shapes)
        return parameters
    
    def close(self):
        """Close camera connections."""
        if self.camera is not None:
            cv2.destroyAllWindows()
            print(f"{'Root' if self.is_root else 'Worker'}: Camera communication closed")


# For testing the communicator standalone
if __name__ == "__main__":
    import sys
    from train.model import MLP
    
    if len(sys.argv) < 2:
        print("Usage: python camera_communicator.py [root|worker]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Create a test model to get parameter shapes
    model = MLP().bfloat16()
    test_params = [p.clone() for p in model.parameters()]
    
    if mode == "root":
        try:
            comm = CameraCommunicator(is_root=True)
            comm.setup(num_workers=1)
            
            # Store parameter shapes
            comm.param_shapes = get_parameter_shapes(model)
            
            print("\nRoot: Sending test parameters...")
            comm.send_parameters(test_params)
            print("Root: Parameters sent!")
            
            print("\nRoot: Waiting for gradients...")
            gradients = comm.receive_gradients()
            print(f"Root: Received {len(gradients)} gradient sets")
            print(f"Root: First gradient set has {len(gradients[0])} tensors")
            
            print("\nRoot: Test complete!")
            comm.close()
            
        except Exception as e:
            print(f"Root ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elif mode == "worker":
        try:
            comm = CameraCommunicator(is_root=False)
            comm.setup()
            
            # Store parameter shapes
            comm.param_shapes = get_parameter_shapes(model)
            
            print("\nWorker: Waiting for parameters...")
            params = comm.receive_parameters()
            print(f"Worker: Received {len(params)} parameters")
            print(f"Worker: First parameter shape: {params[0].shape}")
            
            print("\nWorker: Sending test gradients...")
            test_grads = [torch.randn_like(p) * 0.1 for p in test_params]
            comm.send_gradients(test_grads)
            print("Worker: Gradients sent!")
            
            print("\nWorker: Test complete!")
            comm.close()
            
        except Exception as e:
            print(f"Worker ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
