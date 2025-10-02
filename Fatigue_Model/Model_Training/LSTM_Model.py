import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, num_classes=3):
        """
        LSTM-based fatigue detection model.
        Args:
            input_size (int): Number of features per timestep (7 in your feature extractor).
            hidden_size (int): Number of hidden units in LSTM.
            num_layers (int): Number of stacked LSTM layers.
            num_classes (int): Number of fatigue states (e.g., Not Fatigue, Early Fatigue, Fatigue).
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, (h, c) = self.lstm(x)
        out = h[-1]  # last layer’s hidden state
        out = self.fc(out)  # [batch_size, num_classes]
        return out