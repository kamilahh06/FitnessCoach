import numpy as np
from scipy.signal import butter, lfilter

class PreProcessor:
    def __init__(self, fs=1000, lowcut=20, highcut=450, order=4):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        
    # Complete processing pipeline
    def full_process(self, data):
        filtered = self.filter(data)
        features = self.extract_features(filtered)
        freq_features = self.extract_frequency_features(filtered)
        nonlinear_features = self.extract_nonlinear_features(filtered)
        return features + freq_features + nonlinear_features
    
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
    
    def extract_frequency_features(self, data):
        fft_vals = np.fft.fft(data)
        fft_freq = np.fft.fftfreq(len(fft_vals), 1.0/self.fs)
        fft_power = np.abs(fft_vals)**2

        # Only take the positive frequencies
        pos_mask = fft_freq >= 0
        freqs = fft_freq[pos_mask]
        powers = fft_power[pos_mask]

        # Mean Frequency
        mean_freq = np.sum(freqs * powers) / np.sum(powers)

        # Median Frequency
        cumulative_power = np.cumsum(powers)
        total_power = cumulative_power[-1]
        median_freq = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]

        return mean_freq, median_freq
        
    # non linear estimates
    def extract_nonlinear_features(self, data):
        def lempel_ziv_complexity(s):
            i, k, l = 0, 1, 1
            c = 1
            n = len(s)
            while True:
                if s[i + k - 1] == s[l + k - 1]:
                    k += 1
                    if l + k > n:
                        c += 1
                        break
                else:
                    if k > 1:
                        i += 1
                        if i == l:
                            c += 1
                            l += k
                            i = 0
                        k = 1
                    else:
                        i += 1
                        if i == l:
                            c += 1
                            l += 1
                            i = 0
                if l + k > n:
                    break
            return c

        def sample_entropy(s, m=2, r=0.2 * np.std(s)):
            def _phi(m):
                x = np.array([s[i:i + m] for i in range(len(s) - m + 1)])
                C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
                return np.sum(C) / (len(s) - m + 1)
            return -np.log(_phi(m + 1) / _phi(m))

        def marginal_spectrum_entropy(s):
            fft_vals = np.fft.fft(s)
            psd = np.abs(fft_vals)**2
            psd_norm = psd / np.sum(psd)
            mse = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            return mse

        lz_complexity = lempel_ziv_complexity((data > np.mean(data)).astype(int))
        samp_entropy = sample_entropy(data)
        mse = marginal_spectrum_entropy(data)

        return lz_complexity, samp_entropy, mse