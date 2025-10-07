import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRegressor(nn.Module):
    def __init__(self, input_size=7):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)  # Output single fatigue score

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = x.permute(0, 2, 1)  # [batch, input_size, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [batch, 64]
        out = self.fc(x)
        out = torch.clamp(out, 1, 7)  # restrict output to fatigue range
        return out