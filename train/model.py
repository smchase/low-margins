import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(1 * 8 * 8, num_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def create_model(device='cpu'):
    model = MLP().bfloat16()
    model.to(device)
    return model

if __name__ == "__main__":
    model = MLP().bfloat16()
    total_size_bytes = 0
    for n, p in model.named_parameters():
        size_bytes = p.numel() * p.element_size()
        total_size_bytes += size_bytes
        print(f"{n:20s} | shape: {str(p.shape):20s} | numel: {p.numel():6d} | size: {size_bytes:8d} bytes ({size_bytes/1024:.2f} KB)")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params}")
    print(f"Total size: {total_size_bytes} bytes ({total_size_bytes/1024:.2f} KB, {total_size_bytes/1024/1024:.2f} MB)")