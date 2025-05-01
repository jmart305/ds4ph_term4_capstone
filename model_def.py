import torch
import torch.nn as nn

class EnhancedPredictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.2),
                
                nn.Linear(128, 6)
            )
            
        def forward(self, x):
            return self.net(x)