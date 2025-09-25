import numpy as np
from scipy.signal import butter, lfilter

class PreProcessor:
    def __init__(self, fs=1000, lowcut=20, highcut=450, order=4):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        
    def filter(self, data):
        filtered = self.bandpass_filtering(data)
        detrended = self.detrending(filtered)
        normalized = self.normalize(detrended)
        return normalized

    # --------------------
    # FILTERING
    # --------------------
    def bandpass_filtering(self, data):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def detrending(self, data):
        return data - np.mean(data)

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # --------------------
    # FEATURE EXTRACTION
    # --------------------
    def extract_features(self, data):
        mean = np.mean(data)
        std = np.std(data)
        rms = np.sqrt(np.mean(data**2))
        mav = np.mean(np.abs(data))
        wl = np.sum(np.abs(np.diff(data)))
        zc = ((data[:-1] * data[1:]) < 0).sum()
        ssc = np.sum(np.diff(np.sign(np.diff(data))) != 0)
        return mean, std, rms, mav, wl, zc, ssc