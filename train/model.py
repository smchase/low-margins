import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(6 * 6 * 6, num_classes)
        
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
    model = MLP()
    model.to(device)
    return model
