# import numpy as np, pandas as pd
# from scipy.signal import butter, lfilter, iirnotch, savgol_filter, welch
# import pywt

# class PreProcessor:
#     """
#     Preprocessing pipeline aligned with:
#     Cerqueira et al., *Sensors 2024, 24(24), 8081*
#     "A Comprehensive Dataset of Surface Electromyography and Self-Perceived Fatigue Levels"
#     """

#     def __init__(self, fs=1259, lowcut=20, highcut=450, order=4, notch_freq=60.0):
#         self.fs = fs
#         self.lowcut = lowcut
#         self.highcut = highcut
#         self.order = order
#         self.notch_freq = notch_freq

#     # ======================================================
#     # 🧹 Full Processing Pipeline
#     # ======================================================
#     def full_process(self, x, normalize_mode="zscore"):
#         """
#         Applies filtering, baseline correction, denoising, smoothing, and normalization.
#         """
#         x = np.asarray(x, dtype=float)
#         if x.size == 0:
#             return x

#         x = self.clean_data(x)
#         if self.is_artifact(x):
#             return np.zeros_like(x)

#         x = self.notch_filter(x)
#         x = self.bandpass_filtering(x)
#         x = self.wavelet_denoise(x)
#         x = self.savgol_smooth(x)
#         x = self.normalize(x, mode=normalize_mode)
#         return x

import numpy as np, pandas as pd
from scipy.signal import butter, lfilter, iirnotch, savgol_filter, welch
import pywt

class PreProcessor:
    def __init__(self, fs=1259, lowcut=20, highcut=450, order=4, notch_freq=60.0):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.notch_freq = notch_freq

    def full_process(self, x, normalize_mode="zscore"):
        x = np.asarray(x, dtype=float)
        if x.size == 0: return x
        x = self.clean_data(x)
        if self.is_artifact(x): return np.zeros_like(x)
        x = self.notch_filter(x)
        x = self.bandpass_filtering(x)
        x = self.wavelet_denoise(x)
        x = self.savgol_smooth(x)
        x = self.normalize(x, mode=normalize_mode)
        return x

    # ---- quick RMS helper for activity gating ----
    @staticmethod
    def window_rms(x):
        x = np.asarray(x, dtype=float)
        return float(np.sqrt(np.mean(x * x)))

    # ======================================================
    # 🔧 Cleaning & Filtering
    # ======================================================
    def clean_data(self, x):
        x = x.copy()
        x[np.isnan(x)] = 0
        # Replace exact zeros with interpolation
        if np.any(x == 0):
            x = pd.Series(x).replace(0, np.nan).interpolate(limit_direction="both").bfill().ffill().values
        # Remove spikes (3σ rule)
        z = np.abs((x - np.mean(x)) / (np.std(x) + 1e-8))
        x[z > 3] = np.nan
        x = pd.Series(x).interpolate(limit_direction="both").bfill().ffill().values
        return x

    def notch_filter(self, x, Q=30):
        b, a = iirnotch(self.notch_freq, Q, self.fs)
        return lfilter(b, a, x)

    def bandpass_filtering(self, x):
        """
        4th-order Butterworth 20–450 Hz bandpass filter
        (same as in Cerqueira et al. 2024).
        """
        nyq = 0.5 * self.fs
        low, high = self.lowcut / nyq, self.highcut / nyq
        b, a = butter(self.order, [low, high], btype="band")
        return lfilter(b, a, x)

    def wavelet_denoise(self, x):
        """
        Soft-thresholding with Daubechies-4 wavelet (db4), 4 levels.
        """
        try:
            coeffs = pywt.wavedec(x, "db4", level=4)
            for i in range(1, len(coeffs)):
                sigma = np.std(coeffs[i])
                coeffs[i] = pywt.threshold(coeffs[i], 0.6 * sigma, mode="soft")
            return pywt.waverec(coeffs, "db4")
        except Exception:
            return x

    def savgol_smooth(self, x, window_ms=200, polyorder=3):
        """
        Savitzky–Golay smoothing to reduce rapid fluctuations
        (as in MDPI paper Figures 9–10 for MNF/MDF trends).
        """
        try:
            window_length = max(5, int(self.fs * window_ms / 1000))
            if window_length % 2 == 0:
                window_length += 1
            return savgol_filter(x, window_length, polyorder)
        except Exception:
            return x

    # ======================================================
    # 🧮 Feature Extraction
    # ======================================================
    def is_artifact(self, x):
        rms = np.sqrt(np.mean(x * x))
        var = np.var(x)
        return (len(x) < 10) or (rms < 1e-5) or (var < 1e-8)

    def normalize(self, x, mode="zscore"):
        if mode == "zscore":
            return (x - np.mean(x)) / (np.std(x) + 1e-8)
        if mode == "minmax":
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        return x

    # ---- Time-Domain ----
    def extract_features(self, x):
        x = x - np.mean(x)
        mean = np.mean(x)
        std = np.std(x)
        rms = np.sqrt(np.mean(x ** 2))
        mav = np.mean(np.abs(x))
        wl = np.sum(np.abs(np.diff(x)))
        zc = np.sum(x[:-1] * x[1:] < 0)
        ssc = np.sum(np.diff(np.sign(np.diff(x))) != 0)
        return (mean, std, rms, mav, wl, zc, ssc)

    # ---- Frequency-Domain ----
    def extract_frequency_features(self, x):
        f, Pxx = welch(x, fs=self.fs, nperseg=min(1024, len(x)))
        Pxx = np.maximum(Pxx, 1e-12)
        mean_f = np.sum(f * Pxx) / np.sum(Pxx)
        cum = np.cumsum(Pxx)
        med_f = f[np.searchsorted(cum, cum[-1] / 2)]
        return (mean_f, med_f)

    # ---- Nonlinear ----
    def extract_nonlinear_features(self, x):
        return (
            self.fractal_dim_petrosian(x),
            0, # self.sample_entropy(x),
            self.spectral_entropy(x),
            self.hurst_exponent(x),
            self.lz_complexity(x)
        )

    # ---- Nonlinear helper metrics ----
    def fractal_dim_petrosian(self, x):
        diff = np.diff(x)
        N = len(x)
        Nz = np.sum(diff[:-1] * diff[1:] < 0) + 1e-8
        return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * Nz)))

    def sample_entropy(self, x, m=2, r=None):
        x = np.asarray(x, dtype=float)
        if r is None:
            r = 0.2 * np.std(x)
        N = len(x)
        if N <= m + 1:
            return 0.0
        def _phi(mm):
            X = np.array([x[i:i+mm] for i in range(N-mm+1)])
            C = np.sum(np.max(np.abs(X[:,None,:]-X[None,:,:]), axis=2) <= r, axis=0) - 1
            return np.sum(C) / (N-mm+1 - 1 + 1e-8)
        return -np.log((_phi(m+1) + 1e-12) / (_phi(m) + 1e-12))

    def spectral_entropy(self, x):
        f, Pxx = welch(x, fs=self.fs, nperseg=min(1024, len(x)))
        Pxx = np.maximum(Pxx, 1e-12)
        P = Pxx / np.sum(Pxx)
        return -np.sum(P * np.log(P + 1e-12))

    def hurst_exponent(self, x):
        x = np.asarray(x, dtype=float)
        if len(x) < 16:
            return 0.5
        y = np.cumsum(x - np.mean(x))
        n = len(y)
        sizes = np.unique(np.floor(np.logspace(1, np.log10(n / 2), num=8)).astype(int))
        RS = []
        for s in sizes:
            if s < 2:
                continue
            chunks = y[: (n // s) * s].reshape(-1, s)
            R = (np.max(chunks - chunks.mean(1, keepdims=True), axis=1) -
                 np.min(chunks - chunks.mean(1, keepdims=True), axis=1))
            S = np.std(chunks, axis=1) + 1e-12
            RS.append(np.mean(R / S))
        RS = np.array(RS)
        if len(RS) < 2:
            return 0.5
        H = np.polyfit(np.log(sizes[:len(RS)]), np.log(RS + 1e-12), 1)[0]
        return float(H)

    def lz_complexity(self, x):
        x = np.asarray(x, dtype=float)
        if len(x) < 4:
            return 0.0
        b = (x > np.median(x)).astype(int).tolist()
        s = ''.join(map(str, b))
        i = 0; c = 1; k = 1; l = 1; n = len(s)
        while True:
            if s[l - 1 + i] != s[i]:
                if k == 1:
                    c += 1
                    l += 1
                    i = 0
                    if l > n:
                        break
                else:
                    i = 0
                    k += 1
                    if l + k - 1 > n:
                        c += 1
                        break
            else:
                i += 1
                if i == k:
                    l += k
                    i = 0
                    k = 1
                    if l > n:
                        break
        return c / (n / np.log2(n + 1e-8))