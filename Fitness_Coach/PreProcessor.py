import numpy as np
from scipy.signal import butter, lfilter


class PreProcessor:
    """ Class for preprocessing and feature extraction from EMG data.
    Attributes:
        data (numpy array): Raw EMG data.    
    """
    def __init__(self, window_size, step_size, fs=1000, lowcut=20, highcut=450, order=4):
        # Implement later
        self.window_size = window_size
        self.step_size = step_size 
  
    # FILTERING FUNCTIONS
    def bandpass_filtering(self, data, fs=1000, lowcut=20, highcut=450, order=4):
       """
       Apply a bandpass filter to the data.
       Outputs frequences between lowcut and highcut.
       """
       nyq = 0.5 * self.fs
       low = self.lowcut / nyq
       high = self.highcut / nyq
       b, a = butter(self.order, [low, high], btype='band')
       y = lfilter(b, a, data)
       return y
  
    # OTHER PREPROCESSING FUNCTIONS
    def detrending(self, data):
       return data - np.mean(data)
  
    def normalize(self, data):
       return (data - np.min(data)) / (np.max(data) - np.min(data))
   
   # FEATURE EXTRACTION
    def extract_features(self, data):
         mean = np.mean(data)
         std = np.std(data)
         rms = np.sqrt(np.mean(data**2))
         mav = np.mean(np.abs(data))
         wl = np.sum(np.abs(np.diff(data)))
         zc = ((data[:-1] * data[1:]) < 0).sum()
         ssc = np.sum(np.diff(np.sign(np.diff(data))) != 0)
         return mean, std, rms, mav, wl, zc, ssc
     
     def label_windows(self, df, labels):
        """
        labels: list of (start_time, end_time, label)
        returns: list of labels for each window
        """
        windows = []
        window_labels = []

        values = df['emg_value'].values
        times = df['time_ms'].values

        for start in range(0, len(values) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_times = times[start:end]

            # Check what label matches this window
            label = None
            for (t0, t1, l) in labels:
                if window_times[0] >= t0 and window_times[-1] <= t1:
                    label = l
                    break
            window_labels.append(label)
        
        return window_labels
  
  
  
  

