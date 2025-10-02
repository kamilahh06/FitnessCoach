import torch
from torch.utils.data import Dataset
import numpy as np

class TrainingDataset(Dataset):
    def __init__(self, EMGWindows):
        self.features = []
        self.labels = []
        for window in EMGWindows:
            # Extract features for each window
            mean = np.mean(window)
            std = np.std(window)
            rms = np.sqrt(np.mean(window**2))
            mav = np.mean(np.abs(window))
            wl = np.sum(np.abs(np.diff(window)))
            zc = ((window[:-1] * window[1:]) < 0).sum()
            ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)
            
            feature_vector = [mean, std, rms, mav, wl, zc, ssc]
            self.features.append(feature_vector)
            self.labels.append(window.label)
        
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]