import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """For RFR (feature vectors → label)."""
    def __init__(self, X_feat, y):
        self.X = X_feat
        self.y = y

    def get_arrays(self):
        return self.X, self.y