import torch
from torch.utils.data import Dataset

class EMGSequenceDataset(Dataset):
    """For CNN, LSTM, Transformer (sequence → label)."""
    def __init__(self, X_seq, y):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]