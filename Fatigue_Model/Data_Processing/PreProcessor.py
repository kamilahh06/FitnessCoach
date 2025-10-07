import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch, medfilt, stft, welch
import pywt

class PreProcessor:
    """
    Comprehensive EMG preprocessing pipeline:
    """
    def __init__(self, fs=1000, lowcut=20, highcut=250, order=4, notch_freq=60.0):
        """
        Args:
            fs: Sampling frequency (Hz)
            lowcut: Low cutoff frequency for bandpass (Hz)
            highcut: High cutoff frequency for bandpass (Hz)
            order: Filter order
            notch_freq: Notch filter frequency (Hz)
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.notch_freq = notch_freq

    # MAIN PIPELINE
    def full_process(self, data, normalize_mode="zscore"):
        """
        Complete EMG preprocessing pipeline.
        Args:
            data: raw EMG data (1D array or DataFrame with 'emg_value' column)
            normalize_mode: "zscore", "minmax", or None
        """
        cleaned = self.clean_data(data)
        if self.is_artifact(cleaned):
            print("⚠️ Skipping segment: low amplitude or variance (artifact).")
            return np.zeros_like(cleaned)

        notch = self.notch_filter(cleaned)
        bandpassed = self.bandpass_filtering(notch)
        baseline_corrected = self.adaptive_baseline(bandpassed)
        wavelet_denoised = self.wavelet_denoise(baseline_corrected)
        median_smoothed = self.median_smooth(wavelet_denoised)
        normalized = self.normalize(median_smoothed, mode=normalize_mode)

        print("✅ Preprocessing complete — returning clean EMG signal.")
        return normalized

    # CLEANING
    def clean_data(self, df):
        """
        Cleans EMG data before filtering.
        - Handles 0s as missing
        - Interpolates small gaps
        - Removes outliers
        - Detects and compensates for half-wave rectification
        """
        if isinstance(df, pd.DataFrame):
            df = df.groupby("time_ms", as_index=False)["emg_value"].mean()
            df["emg_value"] = df["emg_value"].astype(float)

            # Handle zeros as missing
            zero_mask = df["emg_value"] == 0
            zero_ratio = zero_mask.mean()
            df.loc[zero_mask, "emg_value"] = np.nan
            df["emg_value"] = df["emg_value"].interpolate(method="linear", limit_direction="both")

            # Remove outliers (>3σ)
            z = np.abs((df["emg_value"] - df["emg_value"].mean()) / df["emg_value"].std())
            df.loc[z > 3, "emg_value"] = np.nan
            df["emg_value"] = df["emg_value"].interpolate(method="linear", limit_direction="both")

            # Detect half-wave rectification
            pos_ratio = np.mean(df["emg_value"] > 0)
            if pos_ratio > 0.9:
                df["emg_value"] = df["emg_value"] * 2.0
                print("⚙️ Detected half-wave rectification — amplitude doubled.")

            # Warn if too many zeros
            if zero_ratio > 0.4:
                print(f"⚠️ {zero_ratio*100:.1f}% zeros detected — possible electrode dropout.")

            data = df["emg_value"].values
        else:
            data = np.array(df, dtype=float)

        return data

    # FILTERING
    def notch_filter(self, data, Q=30):
        b, a = iirnotch(self.notch_freq, Q, self.fs)
        return lfilter(b, a, data)

    def bandpass_filtering(self, data):
        nyq = 0.5 * self.fs
        low, high = self.lowcut / nyq, self.highcut / nyq
        b, a = butter(self.order, [low, high], btype="band")
        return lfilter(b, a, data)

    def adaptive_baseline(self, data, window=None):
        """Adaptive baseline correction using rolling median."""
        if window is None:
            window = int(self.fs * 0.5)  # 0.5 s default
        # baseline = pd.Series(data).rolling(window, min_periods=1).median().fillna(method="bfill")
        baseline = pd.Series(data).rolling(window, min_periods=1).median().bfill()
        return data - baseline.values

    def wavelet_denoise(self, data):
        """Adaptive wavelet denoising (db4)."""
        try:
            level = max(1, int(np.log2(self.fs / self.lowcut)) - 1)
            coeffs = pywt.wavedec(data, "db4", level=level)
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i]) * 0.6, mode="soft")
            return pywt.waverec(coeffs, "db4")
        except Exception:
            return data

    def median_smooth(self, data, kernel=5):
        return medfilt(data, kernel_size=kernel)

    def normalize(self, data, mode="zscore"):
        """Normalize AFTER all filtering."""
        if mode == "zscore":
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        elif mode == "minmax":
            return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        else:
            return data

    # ARTIFACT DETECTION
    def is_artifact(self, data):
        rms = np.sqrt(np.mean(data**2))
        var = np.var(data)
        return rms < 0.005 or var < 1e-6

    # FEATURE EXTRACTION
    def extract_features(self, data):
        data = data - np.mean(data)  # ensure centered for ZC/SSC
        mean = np.mean(data)
        std = np.std(data)
        rms = np.sqrt(np.mean(data**2))
        mav = np.mean(np.abs(data))
        wl = np.sum(np.abs(np.diff(data)))
        zc = np.sum(data[:-1] * data[1:] < 0)
        ssc = np.sum(np.diff(np.sign(np.diff(data))) != 0)
        return mean, std, rms, mav, wl, zc, ssc

    def extract_frequency_features(self, data):
        """Use Welch’s method for stable frequency features."""
        f, Pxx = welch(data, fs=self.fs, nperseg=min(1024, len(data)))
        Pxx = np.maximum(Pxx, 1e-12)
        mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
        cumulative = np.cumsum(Pxx)
        median_freq = f[np.where(cumulative >= cumulative[-1] / 2)[0][0]]
        return mean_freq, median_freq

    def spectral_entropy_welch(self, data):
        """Improved spectral entropy based on Welch PSD."""
        f, Pxx = welch(data, fs=self.fs, nperseg=512)
        Pxx = Pxx / np.sum(Pxx)
        return -np.sum(Pxx * np.log2(Pxx + 1e-10))

    # NONLINEAR FEATURES
    def extract_nonlinear_features(self, data):
        data = np.asarray(data)
        if len(data) < 500:
            return np.nan, np.nan, np.nan

        def higuchi_fd(x, kmax=8):
            N = len(x)
            L = np.zeros(kmax)
            for k in range(1, kmax + 1):
                Lk = []
                for m in range(k):
                    idx1 = np.arange(m, N - k, k)
                    idx2 = np.arange(m + k, N, k)
                    if len(idx2) > len(idx1):
                        idx2 = idx2[:len(idx1)]
                    Lm = np.sum(np.abs(x[idx2] - x[idx1])) * (N - 1) / (len(idx1) * k)
                    Lk.append(Lm)
                L[k - 1] = np.mean(Lk)
            return np.polyfit(np.log(1.0 / np.arange(1, kmax + 1)), np.log(L), 1)[0]

        def sample_entropy(s, m=2, r=0.2):
            s = np.asarray(s)
            N = len(s)
            if np.std(s) == 0:
                return 0
            r *= np.std(s)
            count, total = 0, 0
            for i in range(N - m):
                template = s[i:i+m]
                dists = np.abs(s[i+1:i+m+2] - template[-1])
                total += 1
                if np.all(dists <= r):
                    count += 1
            return -np.log((count + 1e-8) / (total + 1e-8))

        try:
            fractal_dim = higuchi_fd(data)
            sampen = sample_entropy(data)
            spec_ent = self.spectral_entropy_welch(data)
        except Exception:
            return np.nan, np.nan, np.nan

        return fractal_dim, sampen, spec_ent

    # ADDITIONAL METHODS
    def extract_additional_time_features(self, data):
        iemg = np.sum(np.abs(data))
        var = np.var(data)
        return iemg, var

    def extract_instantaneous_frequencies(self, data):
        f, t, Zxx = stft(data, fs=self.fs, nperseg=512)
        power = np.abs(Zxx)**2
        mean_freqs = np.sum(f[:, None] * power, axis=0) / np.sum(power, axis=0)
        median_freqs = []
        for i in range(power.shape[1]):
            cum_power = np.cumsum(power[:, i])
            total = cum_power[-1]
            median_idx = np.where(cum_power >= total / 2)[0][0]
            median_freqs.append(f[median_idx])
        return t, mean_freqs, np.array(median_freqs)