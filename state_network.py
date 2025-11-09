import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from train.model import MLP
from train.worker import Worker
from network import NetworkCommunicator
from enum import Enum
import time
import argparse


class DistributedState(Enum):
    COMPUTE_G = "compute_g"
    COMMUNICATE_G = "communicate_g"
    COMPUTE_THETA = "compute_theta"
    COMMUNICATE_THETA = "communicate_theta"
    COMPUTE_EVAL = "compute_eval"


# State durations in seconds
STATE_DURATIONS = {
    DistributedState.COMPUTE_G: 0.5,          # 50ms
    DistributedState.COMMUNICATE_G: 1,       # 100ms
    DistributedState.COMPUTE_THETA: 0.5,      # 50ms
    DistributedState.COMMUNICATE_THETA: 1,   # 100ms
    DistributedState.COMPUTE_EVAL: 1,        # 1s
}

CYCLE_DURATION = sum(STATE_DURATIONS.values())  # 1 second total


def get_current_state_from_time():
    """Determine which state we should be in based on unix time modulo cycle duration"""
    current_time = time.time()
    cycle_time = current_time % CYCLE_DURATION
    
    t1 = STATE_DURATIONS[DistributedState.COMPUTE_G]
    t2 = t1 + STATE_DURATIONS[DistributedState.COMMUNICATE_G]
    t3 = t2 + STATE_DURATIONS[DistributedState.COMPUTE_THETA]
    t4 = t3 + STATE_DURATIONS[DistributedState.COMMUNICATE_THETA]
    
    if cycle_time < t1:
        return DistributedState.COMPUTE_G, cycle_time
    elif cycle_time < t2:
        return DistributedState.COMMUNICATE_G, cycle_time - t1
    elif cycle_time < t3:
        return DistributedState.COMPUTE_THETA, cycle_time - t2
    elif cycle_time < t4:
        return DistributedState.COMMUNICATE_THETA, cycle_time - t3
    else:
        return DistributedState.COMPUTE_EVAL, cycle_time - t4


class Node:
    def __init__(self, node_id, is_root, data_indices, communicator, learning_rate=0.01, device='cpu'):
        self.node_id = node_id
        self.is_root = is_root
        self.communicator = communicator
        self.learning_rate = learning_rate
        self.device = device
        
        self.model = MLP().bfloat16().to(device)
        
        # All nodes need data and a Worker instance
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        data_subset = Subset(train_dataset, data_indices)
        
        self.worker = Worker(
            model=self.model,
            data_subset=data_subset,
            batch_size=1024,
            learning_rate=learning_rate,
            device=device,
            root=is_root,
            worker_id=node_id
        )
        
        if is_root:
            # Use Adam optimizer instead of manual SGD!
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.collected_gradients = []
        
        # Storage for computed gradients
        self.local_gradients = None
        self.local_loss = None
    
    def execute_compute_g(self):
        """Compute gradients - all nodes do this"""
        start = time.perf_counter()
        self.local_gradients, self.local_loss = self.worker.run_step()
        compute_time = time.perf_counter() - start
        return compute_time
    
    def execute_communicate_g(self):
        """Communicate gradients"""
        start = time.perf_counter()
        
        if self.is_root:
            # Root collects gradients from all workers
            self.collected_gradients = [self.local_gradients]
            worker_grads = self.communicator.receive_gradients()
            self.collected_gradients.extend(worker_grads)
            # Verify we collected gradients
            grad_norm = sum(torch.norm(g).item() for g in self.collected_gradients[0] if g is not None)
            # print(f"  Root collected {len(self.collected_gradients)} gradient sets, grad_norm={grad_norm:.4f}")
        else:
            # Workers send their gradients to root
            self.communicator.send_gradients(self.local_gradients)
            grad_norm = sum(torch.norm(g).item() for g in self.local_gradients if g is not None)
            # print(f"  Worker {self.node_id} sent gradients, grad_norm={grad_norm:.4f}")
        
        comm_time = time.perf_counter() - start
        return comm_time
    
    def execute_compute_theta(self):
        """Root computes new parameters"""
        start = time.perf_counter()
        
        if self.is_root:
            averaged_grads = self._average_gradients(self.collected_gradients)
            
            self.optimizer.zero_grad()
            for param, grad in zip(self.model.parameters(), averaged_grads):
                if grad is not None:
                    param.grad = grad.bfloat16()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        compute_time = time.perf_counter() - start
        return compute_time
    
    def execute_communicate_theta(self):
        """Communicate updated parameters"""
        start = time.perf_counter()
        
        if self.is_root:
            # Root sends updated parameters to all workers
            updated_params = [param.clone().detach() for param in self.model.parameters()]
            param_norm = torch.norm(updated_params[0]).item()
            self.communicator.send_parameters(updated_params)
            # print(f"  Root sent updated parameters, param_norm={param_norm:.4f}")
        else:
            # Workers receive updated parameters from root
            updated_params = self.communicator.receive_parameters()
            self.worker.replace_parameters(updated_params)
            param_norm = torch.norm(updated_params[0]).item()
            # print(f"  Worker {self.node_id} received updated parameters, param_norm={param_norm:.4f}")
        
        comm_time = time.perf_counter() - start
        return comm_time
    
    def save_model(self, path='model_weights.pth'):
        torch.save(self.model.state_dict(), path)
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device).bfloat16()
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.float()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def _average_gradients(self, gradients_list):
        num_workers = len(gradients_list)
        averaged_gradients = []
        
        num_params = len(gradients_list[0])
        
        for param_idx in range(num_params):
            grad_sum = None
            for worker_grads in gradients_list:
                if worker_grads[param_idx] is not None:
                    if grad_sum is None:
                        grad_sum = worker_grads[param_idx].clone()
                    else:
                        grad_sum += worker_grads[param_idx]
            
            if grad_sum is not None:
                averaged_gradients.append(grad_sum / num_workers)
            else:
                averaged_gradients.append(None)
        
        return averaged_gradients


def run_node(is_root, root_host, num_workers, num_steps, learning_rate=0.01, seed=42):
    """
    Run either root or worker node.
    
    Args:
        is_root: True for root node, False for worker
        root_host: IP address of root node
        num_workers: Number of worker nodes (not including root)
        num_steps: Number of training steps
        learning_rate: Learning rate for training
        seed: Random seed for reproducibility
    """
    # Set manual seed so all processes start with same model weights
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set up network communicator
    communicator = NetworkCommunicator(is_root=is_root, root_host=root_host)
    if is_root:
        print(f"Root: Setting up network with {num_workers} workers...")
        communicator.setup(num_workers=num_workers)
    else:
        print(f"Worker: Connecting to root at {root_host}...")
        communicator.setup()
    
    # Partition dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataset_size = len(train_dataset)
    # Total number of nodes including root
    total_nodes = num_workers + 1
    partition_size = dataset_size // total_nodes
    
    # Determine this node's data partition
    if is_root:
        node_idx = 0
        role = "Root"
    else:
        # For now, assume single worker (node_idx = 1)
        # In multi-worker setup, this would need to be passed as argument
        node_idx = 1
        role = f"Worker {node_idx}"
    
    start_idx = node_idx * partition_size
    if node_idx == total_nodes - 1:
        end_idx = dataset_size
    else:
        end_idx = (node_idx + 1) * partition_size
    
    data_indices = list(range(start_idx, end_idx))
    
    # Create node
    node = Node(
        node_id=node_idx,
        is_root=is_root,
        data_indices=data_indices,
        communicator=communicator,
        learning_rate=learning_rate,
        device='cpu'
    )
    
    print(f"{role}: Starting with {len(data_indices)} training samples")
    print(f"{role}: Training for {num_steps} steps")
    print(f"{role}: Initialized with seed={seed}")
    
    # Verify initial weights are the same across all processes
    initial_param_norm = torch.norm(list(node.model.parameters())[0]).item()
    print(f"{role}: Initial param[0] norm = {initial_param_norm:.6f}")
    
    # Wait until we're at the start of a COMPUTE_G phase
    print(f"{role}: Waiting to synchronize to global clock...")
    while True:
        current_state, time_in_state = get_current_state_from_time()
        if current_state == DistributedState.COMPUTE_G and time_in_state < 0.1:  # Within first 100ms
            break
        time.sleep(0.05)  # Check every 50ms
    
    current_time = time.time()
    print(f"{role}: Synchronized! Starting training at unix time {current_time:.6f}")
    
    # Timing accumulators
    total_compute_g_time = 0.0
    total_communicate_g_time = 0.0
    total_compute_theta_time = 0.0
    total_communicate_theta_time = 0.0
    total_train_loss = 0.0
    
    # Load test dataset for evaluation (only root needs this)
    if is_root:
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    for step in range(num_steps):
        # Get the current cycle position based on unix time
        current_time = time.time()
        cycle_time = current_time % CYCLE_DURATION
        
        # Calculate when this cycle started (in absolute unix time)
        cycle_start = current_time - cycle_time
        
        # State 1: COMPUTE_G
        state_start = cycle_start
        state_end = state_start + STATE_DURATIONS[DistributedState.COMPUTE_G]
        
        # We should already be in COMPUTE_G from synchronization, but log entry
        entry_time = time.time()
        entry_offset_ms = (entry_time - state_start) * 1000
        print(f"{role}: [Step {step+1}] Entered COMPUTE_G at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        # Execute compute_g
        compute_g_time = node.execute_compute_g()
        total_compute_g_time += compute_g_time
        total_train_loss += node.local_loss
        
        # Block until state duration is complete
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 2: COMMUNICATE_G
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMMUNICATE_G]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        print(f"{role}: [Step {step+1}] Entered COMMUNICATE_G at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        communicate_g_time = node.execute_communicate_g()
        total_communicate_g_time += communicate_g_time
        
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 3: COMPUTE_THETA
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMPUTE_THETA]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        print(f"{role}: [Step {step+1}] Entered COMPUTE_THETA at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        compute_theta_time = node.execute_compute_theta()
        total_compute_theta_time += compute_theta_time
        
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 4: COMMUNICATE_THETA
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMMUNICATE_THETA]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        print(f"{role}: [Step {step+1}] Entered COMMUNICATE_THETA at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        communicate_theta_time = node.execute_communicate_theta()
        total_communicate_theta_time += communicate_theta_time
        
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 5: COMPUTE_EVAL
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMPUTE_EVAL]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        print(f"{role}: [Step {step+1}] Entered COMPUTE_EVAL at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        # Evaluate model (only root needs to do this)
        eval_start_time = time.time()
        if is_root:
            test_accuracy = node.evaluate(test_loader)
            eval_time = time.time() - eval_start_time

            avg_train_loss = total_train_loss / (step + 1)
            first_param_norm = torch.norm(list(node.model.parameters())[0]).item()
            
            print(f"\n{role}: Step {step+1}/{num_steps}")
            print(f"  Test Accuracy: {test_accuracy:.2f}%, param_norm={first_param_norm:.4f}")
        else:
            # Workers can just log their training loss
            avg_train_loss = total_train_loss / (step + 1)
            print(f"\n{role}: Step {step+1}/{num_steps}, Train Loss: {node.local_loss:.4f} (current), {avg_train_loss:.4f} (avg)")
        
        # Block until state duration is complete
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
    
    if is_root:
        print(f"\n{role}: Training complete!")
        node.save_model('model_weights.pth')
        print(f"{role}: Model saved to model_weights.pth")
        
        # Final evaluation
        final_accuracy = node.evaluate(test_loader)
        final_avg_train_loss = total_train_loss / num_steps
        print(f"{role}: Final Test Accuracy: {final_accuracy:.2f}%")
        print(f"{role}: Final Avg Train Loss: {final_avg_train_loss:.4f}")
    else:
        avg_train_loss = total_train_loss / num_steps
        print(f"\n{role}: Training complete! Avg Train Loss: {avg_train_loss:.4f}")
    
    # Close network connections
    communicator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed training with network communication')
    parser.add_argument('--mode', type=str, required=True, choices=['root', 'worker'],
                        help='Run as root or worker node')
    parser.add_argument('--root-host', type=str, default='localhost',
                        help='Hostname/IP of root node (required for worker)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of worker nodes (not including root)')
    parser.add_argument('--num-steps', type=int, default=30,
                        help='Number of training steps')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    is_root = (args.mode == 'root')
    
    print(f"\n{'='*60}")
    print(f"Starting distributed training")
    print(f"Mode: {args.mode}")
    if not is_root:
        print(f"Root host: {args.root_host}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Number of steps: {args.num_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print(f"Cycle duration: {CYCLE_DURATION}s")
    print(f"{'='*60}\n")
    
    run_node(
        is_root=is_root,
        root_host=args.root_host,
        num_workers=args.num_workers,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    print("\n=== Training complete! ===")

