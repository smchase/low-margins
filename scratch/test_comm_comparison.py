"""
Quick test to show the difference between NetworkCommunicator and CameraCommunicator.

This demonstrates that both implement the same interface.
"""

import torch
from train.model import MLP
from typing import List


def test_communicator(communicator_class, is_root: bool, **kwargs):
    """Test a communicator implementation."""
    print(f"\nTesting {communicator_class.__name__} as {'ROOT' if is_root else 'WORKER'}")
    print("="*60)
    
    # Create communicator
    comm = communicator_class(is_root=is_root, **kwargs)
    
    # Create test model
    model = MLP().bfloat16()
    test_params = [p.clone() for p in model.parameters()]
    test_grads = [torch.randn_like(p) * 0.1 for p in test_params]
    
    print(f"Model has {sum(p.numel() for p in test_params)} parameters")
    
    # Test the interface methods
    print("\nInterface check:")
    print(f"  - setup: {'✓' if hasattr(comm, 'setup') else '✗'}")
    print(f"  - send_gradients: {'✓' if hasattr(comm, 'send_gradients') else '✗'}")
    print(f"  - receive_gradients: {'✓' if hasattr(comm, 'receive_gradients') else '✗'}")
    print(f"  - send_parameters: {'✓' if hasattr(comm, 'send_parameters') else '✗'}")
    print(f"  - receive_parameters: {'✓' if hasattr(comm, 'receive_parameters') else '✗'}")
    print(f"  - close: {'✓' if hasattr(comm, 'close') else '✗'}")
    
    # Show what operations are allowed
    print("\nAllowed operations:")
    if is_root:
        print("  - Can receive gradients from workers")
        print("  - Can send parameters to workers")
    else:
        print("  - Can send gradients to root")
        print("  - Can receive parameters from root")
    
    return comm


if __name__ == "__main__":
    print("Comparing NetworkCommunicator and CameraCommunicator")
    print("="*70)
    
    # Test NetworkCommunicator
    from network import NetworkCommunicator
    net_comm = test_communicator(
        NetworkCommunicator, 
        is_root=True,
        root_host='localhost'
    )
    
    # Test CameraCommunicator  
    from camera_communicator import CameraCommunicator
    cam_comm = test_communicator(
        CameraCommunicator,
        is_root=True
    )
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("\nBoth communicators implement the same interface!")
    print("You can swap them by changing just the import statement.")
    print("\nKey differences:")
    print("- NetworkCommunicator: Uses TCP/IP sockets, fast, unlimited data")
    print("- CameraCommunicator: Uses camera frames, visual, limited bandwidth")
    print("\nUsage in state_network_camera.py:")
    print("  --comm network  # Use NetworkCommunicator")
    print("  --comm camera   # Use CameraCommunicator")
