import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output single fatigue score

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, (h, c) = self.lstm(x)
        out = h[-1]  # last hidden state
        out = self.fc(out)
        out = torch.clamp(out, 1, 7)
        return out