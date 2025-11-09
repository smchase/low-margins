import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from train.model import MLP
from train.worker import Worker
from enum import Enum


class WorkerState(Enum):
    INIT = "init"
    IDLE = "idle"
    COMPUTING = "computing"
    SENDING_GRAD = "sending_grad"
    WAITING_PARAMS = "waiting_params"


class RootState(Enum):
    INIT = "init"
    REQUESTING = "requesting"
    WAITING_GRADS = "waiting_grads"
    AVERAGING = "averaging"
    SENDING_PARAMS = "sending_params"


class Node:
    def __init__(self, node_id, root, data_indices, grad_queue, param_queue, request_queue, learning_rate=0.01, device='cpu'):
        self.node_id = node_id
        self.root = root
        self.grad_queue = grad_queue
        self.param_queue = param_queue
        self.request_queue = request_queue
        self.learning_rate = learning_rate
        self.device = device
        
        self.model = MLP().to(device)
        
        if root:
            self.state = RootState.INIT
            self.worker = None
            # Use Adam optimizer instead of manual SGD!
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.state = WorkerState.INIT
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
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
                batch_size=64,
                learning_rate=learning_rate,
                device=device,
                root=False,
                worker_id=node_id
            )
    
    def step_worker(self):
        if self.state == WorkerState.INIT:
            # Receive initial parameters from root
            initial_params = self.param_queue.get()
            self.worker.replace_parameters(initial_params)
            self.state = WorkerState.IDLE
            
        elif self.state == WorkerState.IDLE:
            request = self.request_queue.get()
            self.state = WorkerState.COMPUTING
            
        elif self.state == WorkerState.COMPUTING:
            self.gradients, self.loss = self.worker.run_step()
            self.state = WorkerState.SENDING_GRAD
            
        elif self.state == WorkerState.SENDING_GRAD:
            self.grad_queue.put(self.gradients)
            self.state = WorkerState.WAITING_PARAMS
            
        elif self.state == WorkerState.WAITING_PARAMS:
            updated_params = self.param_queue.get()
            self.worker.replace_parameters(updated_params)
            self.state = WorkerState.IDLE
    
    def step_root(self, num_workers):
        if self.state == RootState.INIT:
            initial_params = [param.clone().detach() for param in self.model.parameters()]
            for _ in range(num_workers):
                self.param_queue.put(initial_params)
            self.state = RootState.REQUESTING
            
        elif self.state == RootState.REQUESTING:
            for _ in range(num_workers):
                self.request_queue.put("COMPUTE_GRADIENT")
            self.state = RootState.WAITING_GRADS
            
        elif self.state == RootState.WAITING_GRADS:
            all_gradients = []
            for _ in range(num_workers):
                worker_grads = self.grad_queue.get()
                all_gradients.append(worker_grads)
            self.all_gradients = all_gradients
            self.state = RootState.AVERAGING
            
        elif self.state == RootState.AVERAGING:
            averaged_grads = self._average_gradients(self.all_gradients)
            
            self.optimizer.zero_grad()
            for param, grad in zip(self.model.parameters(), averaged_grads):
                if grad is not None:
                    param.grad = grad.clone()
            
            self.optimizer.step()
            
            updated_params = [param.clone().detach() for param in self.model.parameters()]
            
            self.updated_params = updated_params
            self.state = RootState.SENDING_PARAMS
            
        elif self.state == RootState.SENDING_PARAMS:
            for _ in range(num_workers):
                self.param_queue.put(self.updated_params)
            self.state = RootState.REQUESTING
    
    def save_model(self, path='model_weights.pth'):
        torch.save(self.model.state_dict(), path)
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                
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


def worker_process(worker_id, is_root, data_indices, grad_queue, param_queue, request_queue, ready_queue, num_worker_nodes, learning_rate=0.01):
    node = Node(
        node_id=worker_id,
        root=is_root,
        data_indices=data_indices,
        grad_queue=grad_queue,
        param_queue=param_queue,
        request_queue=request_queue,
        learning_rate=learning_rate,
        device='cpu'
    )
    
    if is_root:
        steps_per_worker = []
        for _ in range(num_worker_nodes):
            num_steps = ready_queue.get()
            steps_per_worker.append(num_steps)
        
        node.step_root(num_worker_nodes)
        print(f"Root: Sent initial parameters to {num_worker_nodes} workers")
        
        max_steps = max(steps_per_worker)
        print(f"Root: Training for {max_steps} steps with {num_worker_nodes} workers")
        
        for step in range(max_steps):
            # this is needed because you need to change this many states in the FSM
            node.step_root(num_worker_nodes)
            node.step_root(num_worker_nodes)
            node.step_root(num_worker_nodes)
            node.step_root(num_worker_nodes)
            
            if (step + 1) % 5 == 0 or step == 0:
                first_param_norm = torch.norm(list(node.model.parameters())[0]).item()
                print(f"Root: Step {step+1}/{max_steps}, param_norm={first_param_norm:.4f}")
        
        print(f"Root: Training complete!")
        node.save_model('model_weights.pth')
        print(f"Root: Model saved to model_weights.pth")
        
        print(f"Root: Running evaluation...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        accuracy = node.evaluate(test_loader)
        print(f"Root: Test Accuracy: {accuracy:.2f}%")
    
    else:
        num_steps = len(node.worker.data)
        ready_queue.put(num_steps)
        
        node.step_worker()
        print(f"Worker {worker_id}: Received initial parameters, training for {num_steps} steps")
        total_loss = 0.0
        
        for step in range(num_steps):
            node.step_worker()
            node.step_worker()
            node.step_worker()
            node.step_worker()
            
            total_loss += node.loss
            
            if (step + 1) % 5 == 0 or step == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Worker {worker_id}: Step {step+1}/{num_steps}, Loss: {node.loss:.4f}, Avg Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / num_steps
        print(f"Worker {worker_id}: Epoch complete! Avg Loss: {avg_loss:.4f}")


def run_distributed_training(num_workers=3, learning_rate=0.01):
    mp.set_start_method('spawn', force=True)
    
    grad_queue = mp.Queue()
    param_queue = mp.Queue()
    request_queue = mp.Queue()
    ready_queue = mp.Queue()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataset_size = len(train_dataset)
    num_worker_nodes = num_workers - 1
    partition_size = dataset_size // num_worker_nodes
    
    data_partitions = [[] for _ in range(num_workers)]
    data_partitions[0] = []
    
    for i in range(num_worker_nodes):
        start_idx = i * partition_size
        if i == num_worker_nodes - 1:
            end_idx = dataset_size
        else:
            end_idx = (i + 1) * partition_size
        
        data_partitions[i + 1] = list(range(start_idx, end_idx))
    
    worker_partition_sizes = [len(p) for p in data_partitions[1:]]
    
    processes = []
    for worker_id in range(num_workers):
        is_root = (worker_id == 0)
        
        p = mp.Process(
            target=worker_process,
            args=(worker_id, is_root, data_partitions[worker_id], grad_queue, param_queue, request_queue, ready_queue, num_worker_nodes, learning_rate)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    # Use 2 workers (1 root + 1 worker) for easier debugging
    # Use learning_rate=0.001 to match your working single-process code
    run_distributed_training(num_workers=3, learning_rate=0.001)

