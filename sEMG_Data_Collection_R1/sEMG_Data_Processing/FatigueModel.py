import torch
import torch.nn as nn


class FatigueModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super(FatigueModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # regression? use num_classes=1
        self.softmax = nn.Softmax(dim=1)  # for classification
    
    def forward(self, x):
        # x: [batch, timesteps, input_size]
        out, (h, c) = self.lstm(x)  
        out = h[-1]  # last layer's hidden state
        out = self.fc(out)
        return self.softmax(out)