import torch
import torch.nn as nn

class RawLSTMRegressor(nn.Module):
    """LSTM model for temporal raw EMG data."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(RawLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Expecting shape: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        return self.fc(out)