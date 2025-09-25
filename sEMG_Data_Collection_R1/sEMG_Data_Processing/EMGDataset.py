import torch
from torch.utils.data import Dataset
import numpy as np

class TrainingDataset(Dataset):
    def __init__(self, df):


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]