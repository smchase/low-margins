import socket
import pickle
import struct
import torch
from typing import List, Optional


class NetworkCommunicator:
    """
    Handles network communication between root and worker nodes.
    
    Root node runs as a server that receives gradients and sends parameters.
    Worker nodes run as clients that send gradients and receive parameters.
    """
    
    def __init__(self, is_root: bool, root_host: str = 'localhost', grad_port: int = 5000, param_port: int = 5001):
        """
        Initialize network communicator.
        
        Args:
            is_root: True if this is the root node, False if worker
            root_host: Hostname/IP of the root node
            grad_port: Port for gradient communication (workers send to root)
            param_port: Port for parameter communication (root sends to workers)
        """
        self.is_root = is_root
        self.root_host = root_host
        self.grad_port = grad_port
        self.param_port = param_port
        
        # Sockets for root
        self.grad_server_socket = None  # Root listens for gradients
        self.param_server_socket = None  # Root sends parameters
        
        # Sockets for workers
        self.grad_client_socket = None  # Worker sends gradients
        self.param_client_socket = None  # Worker receives parameters
        
        # Connection tracking
        self.grad_connections = []  # Root: list of worker connections
        self.param_connections = []  # Root: list of worker connections
    
    def setup(self, num_workers: Optional[int] = None):
        """
        Set up network connections.
        
        Args:
            num_workers: Required for root node - number of workers to expect
        """
        if self.is_root:
            if num_workers is None:
                raise ValueError("Root must specify num_workers")
            self._setup_root(num_workers)
        else:
            self._setup_worker()
    
    def _setup_root(self, num_workers: int):
        """Set up root as server to receive gradients and send parameters"""
        print(f"Root: Setting up network server on {self.root_host}:{self.grad_port} (gradients) and {self.root_host}:{self.param_port} (parameters)")
        
        # Set up gradient server (receives from workers)
        self.grad_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.grad_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.grad_server_socket.bind(('0.0.0.0', self.grad_port))
        self.grad_server_socket.listen(num_workers)
        print(f"Root: Listening for gradient connections on port {self.grad_port}")
        
        # Accept gradient connections from all workers
        for i in range(num_workers):
            conn, addr = self.grad_server_socket.accept()
            self.grad_connections.append(conn)
            print(f"Root: Worker {i+1}/{num_workers} connected for gradients from {addr}")
        
        # Set up parameter server (sends to workers)
        self.param_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.param_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.param_server_socket.bind(('0.0.0.0', self.param_port))
        self.param_server_socket.listen(num_workers)
        print(f"Root: Listening for parameter connections on port {self.param_port}")
        
        # Accept parameter connections from all workers
        for i in range(num_workers):
            conn, addr = self.param_server_socket.accept()
            self.param_connections.append(conn)
            print(f"Root: Worker {i+1}/{num_workers} connected for parameters from {addr}")
        
        print("Root: All workers connected!")
    
    def _setup_worker(self):
        """Set up worker as client to send gradients and receive parameters"""
        print(f"Worker: Connecting to root at {self.root_host}")
        
        # Connect to root's gradient server
        self.grad_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.grad_client_socket.connect((self.root_host, self.grad_port))
        print(f"Worker: Connected to gradient server at {self.root_host}:{self.grad_port}")
        
        # Connect to root's parameter server
        self.param_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.param_client_socket.connect((self.root_host, self.param_port))
        print(f"Worker: Connected to parameter server at {self.root_host}:{self.param_port}")
    
    def send_gradients(self, gradients: List[torch.Tensor]):
        """
        Worker sends gradients to root.
        
        Args:
            gradients: List of gradient tensors
        """
        if self.is_root:
            raise RuntimeError("Root should not send gradients")
        
        # Convert to CPU and serialize
        cpu_gradients = [g.cpu() if g is not None else None for g in gradients]
        data = pickle.dumps(cpu_gradients)
        
        # Send length first, then data
        self._send_data(self.grad_client_socket, data)
    
    def receive_gradients(self) -> List[List[torch.Tensor]]:
        """
        Root receives gradients from all workers.
        
        Returns:
            List of gradient lists, one per worker
        """
        if not self.is_root:
            raise RuntimeError("Only root should receive gradients")
        
        all_gradients = []
        for i, conn in enumerate(self.grad_connections):
            data = self._receive_data(conn)
            gradients = pickle.loads(data)
            all_gradients.append(gradients)
            # print(f"Root: Received gradients from worker {i+1}")
        
        return all_gradients
    
    def send_parameters(self, parameters: List[torch.Tensor]):
        """
        Root sends parameters to all workers.
        
        Args:
            parameters: List of parameter tensors
        """
        if not self.is_root:
            raise RuntimeError("Only root should send parameters")
        
        # Convert to CPU and serialize
        cpu_parameters = [p.cpu() for p in parameters]
        data = pickle.dumps(cpu_parameters)
        
        # Send to all workers
        for i, conn in enumerate(self.param_connections):
            self._send_data(conn, data)
            # print(f"Root: Sent parameters to worker {i+1}")
    
    def receive_parameters(self) -> List[torch.Tensor]:
        """
        Worker receives parameters from root.
        
        Returns:
            List of parameter tensors
        """
        if self.is_root:
            raise RuntimeError("Root should not receive parameters")
        
        data = self._receive_data(self.param_client_socket)
        parameters = pickle.loads(data)
        # print(f"Worker: Received parameters from root")
        return parameters
    
    def _send_data(self, sock: socket.socket, data: bytes):
        """Send data with length prefix"""
        # Send 4-byte length prefix
        length = len(data)
        sock.sendall(struct.pack('!I', length))
        # Send actual data
        sock.sendall(data)
    
    def _receive_data(self, sock: socket.socket) -> bytes:
        """Receive data with length prefix"""
        # Receive 4-byte length prefix
        length_data = self._recv_exactly(sock, 4)
        length = struct.unpack('!I', length_data)[0]
        # Receive actual data
        data = self._recv_exactly(sock, length)
        return data
    
    def _recv_exactly(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket connection broken")
            data.extend(packet)
        return bytes(data)
    
    def close(self):
        """Close all network connections"""
        if self.is_root:
            for conn in self.grad_connections:
                conn.close()
            for conn in self.param_connections:
                conn.close()
            if self.grad_server_socket:
                self.grad_server_socket.close()
            if self.param_server_socket:
                self.param_server_socket.close()
            print("Root: All connections closed")
        else:
            if self.grad_client_socket:
                self.grad_client_socket.close()
            if self.param_client_socket:
                self.param_client_socket.close()
            print("Worker: All connections closed")


# Example usage for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python network.py [root|worker] [root_host]")
        print("  root: Run as root node")
        print("  worker: Run as worker node (specify root_host)")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "root":
        # Test as root
        comm = NetworkCommunicator(is_root=True)
        comm.setup(num_workers=1)
        
        print("\nRoot: Waiting for gradients...")
        gradients = comm.receive_gradients()
        print(f"Root: Received {len(gradients)} gradient sets")
        print(f"Root: First gradient set has {len(gradients[0])} tensors")
        
        print("\nRoot: Sending test parameters...")
        test_params = [torch.randn(10, 10) for _ in range(3)]
        comm.send_parameters(test_params)
        print("Root: Parameters sent!")
        
        comm.close()
        
    elif mode == "worker":
        root_host = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
        
        # Test as worker
        comm = NetworkCommunicator(is_root=False, root_host=root_host)
        comm.setup()
        
        print("\nWorker: Sending test gradients...")
        test_grads = [torch.randn(10, 10) for _ in range(3)]
        comm.send_gradients(test_grads)
        print("Worker: Gradients sent!")
        
        print("\nWorker: Waiting for parameters...")
        params = comm.receive_parameters()
        print(f"Worker: Received {len(params)} parameters")
        print(f"Worker: First parameter shape: {params[0].shape}")
        
        comm.close()
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

