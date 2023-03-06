import torch.nn as nn

class NN(nn.Module):
    def __init__(self, num_classes=10):
        super(NN,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    
    def forward(self, x):
        # Flatten starting from the second axis since NCHW - first axis corresponds to batch size
        return self.fc(x.flatten(start_dim=1))