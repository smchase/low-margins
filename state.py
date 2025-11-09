import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from train.model import MLP
from train.worker import Worker
from enum import Enum
import time


class DistributedState(Enum):
    COMPUTE_G = "compute_g"
    COMMUNICATE_G = "communicate_g"
    COMPUTE_THETA = "compute_theta"
    COMMUNICATE_THETA = "communicate_theta"
    COMPUTE_EVAL = "compute_eval"


# State durations in seconds
STATE_DURATIONS = {
    DistributedState.COMPUTE_G: 0.05,          # 500ms
    DistributedState.COMMUNICATE_G: 0.1,      # 4s
    DistributedState.COMPUTE_THETA: 0.05,      # 500ms
    DistributedState.COMMUNICATE_THETA: 0.1,  # 4s
    DistributedState.COMPUTE_EVAL: 0.7,       # 1s
}

CYCLE_DURATION = sum(STATE_DURATIONS.values())  # 10 seconds total


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
    def __init__(self, node_id, is_root, data_indices, grad_queue, param_queue, learning_rate=0.01, device='cpu'):
        self.node_id = node_id
        self.is_root = is_root
        self.grad_queue = grad_queue
        self.param_queue = param_queue
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
    
    def get_current_state(self, elapsed_time):
        """Determine which state we should be in based on elapsed time"""
        cycle_time = elapsed_time % CYCLE_DURATION
        
        if cycle_time < STATE_DURATIONS[DistributedState.COMPUTE_G]:
            return DistributedState.COMPUTE_G
        elif cycle_time < STATE_DURATIONS[DistributedState.COMPUTE_G] + STATE_DURATIONS[DistributedState.COMMUNICATE_G]:
            return DistributedState.COMMUNICATE_G
        elif cycle_time < STATE_DURATIONS[DistributedState.COMPUTE_G] + STATE_DURATIONS[DistributedState.COMMUNICATE_G] + STATE_DURATIONS[DistributedState.COMPUTE_THETA]:
            return DistributedState.COMPUTE_THETA
        else:
            return DistributedState.COMMUNICATE_THETA
    
    def execute_compute_g(self):
        """Compute gradients - all nodes do this"""
        start = time.perf_counter()
        self.local_gradients, self.local_loss = self.worker.run_step()
        compute_time = time.perf_counter() - start
        return compute_time
    
    def execute_communicate_g(self, num_workers):
        """Communicate gradients"""
        start = time.perf_counter()
        
        if self.is_root:
            # Root collects gradients from all workers
            self.collected_gradients = [self.local_gradients]
            for i in range(num_workers):
                worker_grads = self.grad_queue.get()
                self.collected_gradients.append(worker_grads)
            # Verify we collected gradients
            grad_norm = sum(torch.norm(g).item() for g in self.collected_gradients[0] if g is not None)
            # print(f"  Root collected {len(self.collected_gradients)} gradient sets, grad_norm={grad_norm:.4f}")
        else:
            # Workers send their gradients to root
            self.grad_queue.put(self.local_gradients)
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
    
    def execute_communicate_theta(self, num_workers):
        """Communicate updated parameters"""
        start = time.perf_counter()
        
        if self.is_root:
            # Root sends updated parameters to all workers
            updated_params = [param.clone().detach() for param in self.model.parameters()]
            param_norm = torch.norm(updated_params[0]).item()
            for _ in range(num_workers):
                self.param_queue.put(updated_params)
            # print(f"  Root sent updated parameters to {num_workers} workers, param_norm={param_norm:.4f}")
        else:
            # Workers receive updated parameters from root
            updated_params = self.param_queue.get()
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
    
    def _replace_parameters(self, new_parameters):
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), new_parameters):
                param.copy_(new_param)


def worker_process(worker_id, is_root, data_indices, grad_queue, param_queue, num_worker_nodes, num_steps, learning_rate=0.01, seed=42):
    # Set manual seed so all processes start with same model weights
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    node = Node(
        node_id=worker_id,
        is_root=is_root,
        data_indices=data_indices,
        grad_queue=grad_queue,
        param_queue=param_queue,
        learning_rate=learning_rate,
        device='cpu'
    )
    
    role = "Root" if is_root else f"Worker {worker_id}"
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
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
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
        
        # State 1: COMPUTE_G (500ms)
        state_start = cycle_start
        state_end = state_start + STATE_DURATIONS[DistributedState.COMPUTE_G]
        
        # We should already be in COMPUTE_G from synchronization, but log entry
        entry_time = time.time()
        entry_offset_ms = (entry_time - state_start) * 1000
        # print(f"{role}: [Step {step+1}] Entered COMPUTE_G at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        # Execute compute_g
        compute_g_time = node.execute_compute_g()
        total_compute_g_time += compute_g_time
        total_train_loss += node.local_loss
        
        # Block until state duration is complete
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 2: COMMUNICATE_G (4s)
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMMUNICATE_G]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        # print(f"{role}: [Step {step+1}] Entered COMMUNICATE_G at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        communicate_g_time = node.execute_communicate_g(num_worker_nodes)
        total_communicate_g_time += communicate_g_time
        
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 3: COMPUTE_THETA (500ms)
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMPUTE_THETA]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        # print(f"{role}: [Step {step+1}] Entered COMPUTE_THETA at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        compute_theta_time = node.execute_compute_theta()
        total_compute_theta_time += compute_theta_time
        
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 4: COMMUNICATE_THETA (4s)
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMMUNICATE_THETA]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        # print(f"{role}: [Step {step+1}] Entered COMMUNICATE_THETA at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        communicate_theta_time = node.execute_communicate_theta(num_worker_nodes)
        total_communicate_theta_time += communicate_theta_time
        
        current_time = time.time()
        if current_time < state_end:
            time.sleep(state_end - current_time)
        
        # State 5: COMPUTE_EVAL (1s)
        state_start = state_end
        state_end = state_start + STATE_DURATIONS[DistributedState.COMPUTE_EVAL]
        
        entry_time = time.time()
        cycle_time = entry_time % CYCLE_DURATION
        entry_offset_ms = (entry_time - state_start) * 1000
        # print(f"{role}: [Step {step+1}] Entered COMPUTE_EVAL at {entry_time:.6f} (cycle_time={cycle_time:.3f}s, offset: {entry_offset_ms:+.2f}ms)")
        
        # Evaluate model (only root needs to do this)
        eval_start_time = time.time()
        if is_root:
            test_accuracy = node.evaluate(test_loader)
            eval_time = time.time() - eval_start_time
            
            avg_train_loss = total_train_loss / (step + 1)
            first_param_norm = torch.norm(list(node.model.parameters())[0]).item()
            
            print(f"\n{role}: Step {step+1}/{num_steps}")
            # print(f"  Train Loss: {node.local_loss:.4f} (current), {avg_train_loss:.4f} (avg)")
            print(f"  Test Accuracy: {test_accuracy:.2f}%, param_norm={first_param_norm:.4f}")
            # print(f"  Eval Time: {eval_time*1000:.2f}ms")
            # print(f"  Actual Timing - Compute G: {compute_g_time*1000:.2f}ms, Communicate G: {communicate_g_time*1000:.2f}ms, Compute θ: {compute_theta_time*1000:.2f}ms, Communicate θ: {communicate_theta_time*1000:.2f}ms, Eval: {eval_time*1000:.2f}ms")
            # print(f"  Average Timing - Compute G: {total_compute_g_time/(step+1)*1000:.2f}ms, Communicate G: {total_communicate_g_time/(step+1)*1000:.2f}ms, Compute θ: {total_compute_theta_time/(step+1)*1000:.2f}ms, Communicate θ: {total_communicate_theta_time/(step+1)*1000:.2f}ms")
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


def run_distributed_training(num_workers=2, learning_rate=0.01, num_steps=50, seed=42):
    mp.set_start_method('spawn', force=True)
    
    grad_queue = mp.Queue()
    param_queue = mp.Queue()
    
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
    # All nodes (including root) do work, so partition data among all num_workers
    partition_size = dataset_size // num_workers
    
    data_partitions = []
    
    for i in range(num_workers):
        start_idx = i * partition_size
        if i == num_workers - 1:
            # Last partition gets any remaining data
            end_idx = dataset_size
        else:
            end_idx = (i + 1) * partition_size
        
        data_partitions.append(list(range(start_idx, end_idx)))
    
    num_worker_nodes = num_workers - 1  # Number of non-root workers
    
    print(f"Starting distributed training with {num_workers} nodes ({num_worker_nodes} workers + 1 root)")
    print(f"Training for {num_steps} steps")
    print(f"Each cycle takes {CYCLE_DURATION}s:")
    print(f"  - Compute G: {STATE_DURATIONS[DistributedState.COMPUTE_G]}s")
    print(f"  - Communicate G: {STATE_DURATIONS[DistributedState.COMMUNICATE_G]}s")
    print(f"  - Compute θ: {STATE_DURATIONS[DistributedState.COMPUTE_THETA]}s")
    print(f"  - Communicate θ: {STATE_DURATIONS[DistributedState.COMMUNICATE_THETA]}s")
    print(f"  - Compute Eval: {STATE_DURATIONS[DistributedState.COMPUTE_EVAL]}s")
    print(f"Total training time will be ~{num_steps * CYCLE_DURATION:.1f}s")
    print(f"Synchronizing all processes to global unix time clock...\n")
    
    processes = []
    for worker_id in range(num_workers):
        is_root = (worker_id == 0)
        
        p = mp.Process(
            target=worker_process,
            args=(worker_id, is_root, data_partitions[worker_id], grad_queue, param_queue, num_worker_nodes, num_steps, learning_rate, seed)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("\n=== Distributed training complete! ===")


if __name__ == "__main__":
    # Use 2 workers (1 root + 1 worker) for easier debugging
    # Use learning_rate=0.001 to match your working single-process code
    run_distributed_training(num_workers=2, learning_rate=0.1, num_steps=30)

