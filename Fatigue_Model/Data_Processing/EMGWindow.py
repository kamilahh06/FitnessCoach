import numpy as np
import pandas as pd
from .PreProcessor import PreProcessor

class EMGWindow:
    """One window for one trial: processed signal, features, and the average fatigue label within the window."""
    def __init__(self, trial_dict, t0, t1, fs=1259, normalize_label_to_7=False):
        self.trial = trial_dict
        self.t0 = t0
        self.t1 = t1
        self.fs = fs
        self.normalize_label_to_7 = normalize_label_to_7
        self.processor = PreProcessor(fs=fs)

        # Slice raw EMG signal
        sig = self._slice_emg()

        # Preprocess & extract features
        clean = self.processor.full_process(sig, normalize_mode="zscore")
        self.features = self.processor.extract_features(clean)

        # Compute the label as the *average* of all fatigue values inside the window
        self.label = self._average_label(t0, t1)

    def _slice_emg(self):
        df = self.trial["emg_df"]
        tcol = self.trial["time_col"]
        ccol = self.trial["chosen_channel"]
        m = (df[tcol] >= self.t0) & (df[tcol] < self.t1)
        seg = df.loc[m, ccol].astype(float).values
        return seg

    def _average_label(self, t0, t1):
        """Compute average fatigue label within this window (interpolated if needed)."""
        labels = self.trial["label_df"].copy()
        mask = (labels["time"] >= t0) & (labels["time"] < t1)
        sub = labels.loc[mask, "label"].values

        if len(sub) == 0:
            # If no label points directly in window, interpolate around t0/t1
            label = np.interp((t0 + t1) / 2.0, labels["time"].values, labels["label"].values)
        else:
            label = np.mean(sub)

        label = float(np.clip(label, 0.0, 2.0))
        if self.normalize_label_to_7:
            # Map 0..2 -> 1..7 linearly
            label = 1 + label * 3.0
        return label