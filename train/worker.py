import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from train.model import MLP


class Worker:
    def __init__(self, model, data_subset, batch_size=32, learning_rate=0.001, device='cpu', root=False, worker_id=0):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.root = root
        self.worker_id = worker_id
        
        self.data = DataLoader(
            data_subset,
            batch_size=batch_size,
            shuffle=True
        )
        self.data_iter = iter(self.data)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def run_step(self):
        self.model.train()
        try:
            inputs, labels = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data)
            inputs, labels = next(self.data_iter)
        
        inputs = inputs.to(self.device).bfloat16()
        labels = labels.to(self.device)
        
        self.model.zero_grad()
        
        outputs = self.model(inputs)
        outputs = outputs.float()
        loss = self.criterion(outputs, labels)
        
        loss.backward()
        
        gradients = [param.grad.float().clone() if param.grad is not None else None 
                     for param in self.model.parameters()]
        
        return gradients, loss.item()
    
    def replace_parameters(self, new_parameters):
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), new_parameters):
                param.copy_(new_param)
    
    def get_parameters(self):
        return [param.clone().detach() for param in self.model.parameters()]
    
    def average_gradients(self, gradients_list):
        if not self.root:
            raise RuntimeError("average_gradients can only be called on root worker")
        
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

