import numpy as np
from scipy.signal import butter, lfilter


class Processor:
   def __init__(self, fs=30.0, lowcut=0.1, highcut=3.0, order=5):
       # self.fs = fs
       # self.lowcut = lowcut
       # self.highcut = highcut
       # self.order = order
  
   # FILTERING FUNCTIONS
   def bandpass_filtering(self, data):
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
  
  
  
  

