import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 16 out channels from last conv layer, 24 by 24 pixel input size is reduced to 12 by 12 after maxpooling layer
        self.fc = nn.Linear(32*12*12, num_classes)
    
        
    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        out = self.fc(out)
        return out
