"""
Fatigue Model Package — Full pipeline from preprocessing → feature extraction → training.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, r2_score

# === Internal imports ===
from ..Data_Processing.pipeline import create_participant_list
from ..analysis import run_statistical_analysis
from ..Data_Processing.PreProcessor import PreProcessor
from ..Data_Processing.config import EMG_FS, DEFAULT_WINDOW_S, DEFAULT_STEP_S, LABEL_DIR, EMG_DIR

# === ML Imports ===
from .Models.CNN import CNNRegressor
from .Models.LSTM import RawLSTMRegressor
from .Models.RFR import RFRModel
from .Models.RawCNN import RawCNNRegressor
from .Models.Transformer import TransformerModel

from .Training.Trainer import RegressorTrainer
from ..Data_Processing.EMGValuesDataset import EMGSequenceDataset
from .Training.train_features import train_stacking_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import torch
import joblib

SAVE_DIR_DEFAULT = "/u/kamilah/FitnessCoach/Signal_Plotting/viz_biceps_4"

class FatiguePipeline:
    def __init__(self, save_dir, window_s=DEFAULT_WINDOW_S, step_s=DEFAULT_STEP_S):
        self.save_dir = save_dir
        self.window_s = window_s
        self.step_s = step_s
        self.processor = PreProcessor(fs=EMG_FS)
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1️⃣ Process raw Trial 5 Biceps data → extract + save
    # ------------------------------------------------------------
    def process_trial5_biceps(self):
        """
        Extracts features & labels from all subjects’ trial_5 biceps EMG.
        Saves numpy arrays (X_feat, y_feat, groups) for later use.
        """
        print("⚙️ Processing Trial 5 Right Biceps dataset...")
        all_feats, all_labels, all_groups = [], [], []

        for p in create_participant_list():
            for tr in p.trials:
                emg = pd.to_numeric(tr["emg_df"][ch], errors="coerce").fillna(0).to_numpy()
                label = tr["label_df"]["label"].astype(float).to_numpy()
                
                if not tr["name"].lower().startswith("trial_5"):
                    continue
                ch = tr["chosen_channel"]
                if "bicep" not in ch.lower():
                    continue

                emg = pd.to_numeric(tr["emg_df"][ch], errors="coerce").fillna(0).to_numpy()
                label = tr["label_df"]["label"].astype(float).to_numpy()
                label = (label - label.min()) / (label.max() - label.min() + 1e-8)
                print(f"  → Processing {p.subject_id}:{tr['name']} ({ch})  |  len(emg)={len(emg)}")
                emg = self.processor.full_process(emg, normalize_mode="zscore")

                feats, labels = self.extract_features(emg, label)
                if len(feats) == 0:
                    continue

                all_feats.append(feats)
                all_labels.append(labels)
                all_groups.extend([p.subject_id] * len(labels))

        df = pd.DataFrame(np.vstack(all_feats), columns=[
            "mean","std","rms","mav","wl","zc","ssc",
            "mean_freq","median_freq",
            "fractal_dim","sample_entropy","spectral_entropy","hurst_exp","lz_complexity"
        ]).replace([np.inf, -np.inf], np.nan).fillna(0)
        y = np.concatenate(all_labels)
        g = np.array(all_groups)

        np.save(os.path.join(self.save_dir, "X_feat.npy"), df.values)
        np.save(os.path.join(self.save_dir, "y_feat.npy"), y)
        np.save(os.path.join(self.save_dir, "groups.npy"), g)
        print(f"✅ Saved processed data in {self.save_dir}")
        print(f"   Shapes — X: {df.shape}, y: {y.shape}, groups: {len(np.unique(g))}")
    
    def process_raw_emg(self):
        print("⚙️ Processing raw EMG windows for raw CNN/LSTM...")
        all_raw, all_labels = [], []
        
        for p in create_participant_list():
            print(f"→ Processing subject {p.subject_id}...")
            for tr in p.trials:
                if not tr["name"].lower().startswith("trial_5"):
                    continue
                ch = tr["chosen_channel"]
                if "bicep" not in ch.lower():
                    continue

                emg = pd.to_numeric(tr["emg_df"][ch], errors="coerce").fillna(0).to_numpy()
                label_df = tr["label_df"]
                
                print(f"{p.subject_id}:{tr['name']}")
                print(f"  EMG length: {len(emg)}")
                print(f"  Label length: {len(label_df)}")

                # ✅ FIX: Interpolate labels to match EMG sampling rate
                from scipy.interpolate import interp1d
                
                # Get label times and values
                label_times = label_df["time"].astype(float).to_numpy()
                label_values = label_df["label"].astype(float).to_numpy()
                
                # Get EMG times (assuming it starts at same time as labels)
                emg_times = np.arange(len(emg)) / EMG_FS + label_times[0]
                
                # Interpolate labels to EMG timestamps
                interp_func = interp1d(
                    label_times, 
                    label_values, 
                    kind='linear',
                    bounds_error=False,
                    fill_value=(label_values[0], label_values[-1])  # extrapolate edges
                )
                label = interp_func(emg_times)
                
                print(f"  Upsampled label length: {len(label)}")
                
                # Now they should match
                assert len(emg) == len(label), f"Length mismatch: EMG={len(emg)}, Label={len(label)}"

                # --- Clean NaNs ---
                valid_mask = ~np.isnan(label)
                emg = emg[valid_mask]
                label = label[valid_mask]

                if len(label) == 0:
                    print(f"⚠️ Skipping {p.subject_id}:{tr['name']} — empty after cleaning")
                    continue

                # --- Normalize ---
                if label.max() > label.min():
                    label = (label - label.min()) / (label.max() - label.min())
                else:
                    label = np.zeros_like(label)

                emg = self.processor.full_process(emg, normalize_mode="zscore")

                win = int(self.window_s * EMG_FS)
                step = int(self.step_s * EMG_FS)
                
                print(f"  Window: {win} samples, Step: {step} samples")
                print(f"  Max possible windows: {(len(emg) - win) // step + 1}")
                
                windows_created = 0
                for i in range(0, len(emg) - win + 1, step):
                    seg = emg[i:i + win]
                    label_seg = label[i:i + win]
                    
                    if len(seg) < win or np.all(seg == 0) or np.isnan(seg).all():
                        continue
                    if np.isnan(label_seg).all():
                        continue
                    if self.processor.is_artifact(seg):
                        continue

                    all_raw.append(seg)
                    all_labels.append(np.nanmean(label_seg))
                    windows_created += 1
                
                print(f"  ✓ Created {windows_created} windows\n")

        X_raw = np.stack(all_raw)
        y_raw = np.array(all_labels)
        np.save(os.path.join(self.save_dir, "X_raw.npy"), X_raw)
        np.save(os.path.join(self.save_dir, "y_raw.npy"), y_raw)
        print(f"\n{'='*50}")
        print(f"NaN count in y: {np.isnan(y_raw).sum()}")
        print(f"Unique y values: {len(np.unique(y_raw))} unique values")
        print(f"X contains NaN: {np.isnan(X_raw).any()}")
        print(f"✅ Saved raw EMG arrays → {self.save_dir}")
        print(f"   X shape: {X_raw.shape}")
        print(f"   y shape: {y_raw.shape}")

    # ------------------------------------------------------------
    # 2️⃣ Load already processed Trial 5 Biceps data
    # ------------------------------------------------------------
    def load_trial5_biceps(self):
        """Loads preprocessed feature data if available."""
        X = np.load(os.path.join(self.save_dir, "X_feat.npy"))
        y = np.load(os.path.join(self.save_dir, "y_feat.npy"))
        g = np.load(os.path.join(self.save_dir, "groups.npy"))
        print(f"📦 Loaded preprocessed data: X={X.shape}, y={y.shape}, subjects={len(np.unique(g))}")
        return X, y, g

    # ------------------------------------------------------------
    # 3️⃣ Train models on the loaded dataset
    # ------------------------------------------------------------
    def train_models(self, prefix="", use_rf=False, use_cnn=False, use_rawcnn=False, use_lstm=False):
        """Trains selected models (RF, CNN, RawCNN, or LSTM)."""

        if use_rf:
            X, y, g = self.load_trial5_biceps()
            print("\n🌲 Training Random Forest + Gradient Boost stacking ensemble...")
            df = pd.DataFrame(X, columns=[
                "mean","std","rms","mav","wl","zc","ssc",
                "mean_freq","median_freq",
                "fractal_dim","sample_entropy","spectral_entropy","hurst_exp","lz_complexity"
            ])
            train_stacking_model(df, y, g, self.save_dir)

        if use_cnn:
            X, y, g = self.load_trial5_biceps()
            print("\n🧠 Training Feature-Based CNN regression model...")
            X_seq = np.expand_dims(X, axis=2)
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
            train_ds = EMGSequenceDataset(X_train, y_train)
            test_ds = EMGSequenceDataset(X_test, y_test)

            cnn = CNNRegressor(input_size=1)
            trainer = RegressorTrainer(cnn, train_ds, test_ds, lr=1e-3, batch_size=32)
            trainer.train(epochs=300)
            trainer.evaluate()
            torch.save(cnn.state_dict(), os.path.join(self.save_dir, f"cnn_feature_based_{prefix}.pt"))

        if use_rawcnn:
            print("\n💪 Training Raw-Signal CNN model...")
            # Assume raw EMG arrays saved separately as "X_raw.npy", "y_raw.npy"
            X_raw = np.load(os.path.join(self.save_dir, "X_raw.npy"))
            y_raw = np.load(os.path.join(self.save_dir, "y_raw.npy"))
            X_raw = np.expand_dims(X_raw, 1)
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

            train_ds = EMGSequenceDataset(X_train, y_train)
            test_ds = EMGSequenceDataset(X_test, y_test)

            rawcnn = RawCNNRegressor(input_channels=1)
            from torch.utils.data import Subset
            subset_idx = np.random.choice(len(train_ds), size=300, replace=False)  # try 300 windows
            train_small = Subset(train_ds, subset_idx)

            trainer = RegressorTrainer(rawcnn, train_small, test_ds, lr=1e-3, batch_size=16)
            trainer.train(epochs=100)
            # trainer = RegressorTrainer(rawcnn, train_ds, test_ds, lr=1e-3, batch_size=16)
            # trainer.train(epochs=5)
            trainer.evaluate()
            torch.save(rawcnn.state_dict(), os.path.join(self.save_dir, f"cnn_raw_signal_{prefix}.pt"))

        if use_lstm:
            print("\n📈 Training LSTM on raw EMG sequences...")
            X_raw = np.load(os.path.join(self.save_dir, "X_raw.npy"))
            y_raw = np.load(os.path.join(self.save_dir, "y_raw.npy"))
            X_raw = np.expand_dims(X_raw, 2)  # [batch, seq_len, 1]
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

            train_ds = EMGSequenceDataset(X_train, y_train)
            test_ds = EMGSequenceDataset(X_test, y_test)

            lstm = RawLSTMRegressor(input_size=1, hidden_size=64, num_layers=2)
            trainer = RegressorTrainer(lstm, train_ds, test_ds, lr=1e-3, batch_size=32)
            trainer.train(epochs=30)
            trainer.evaluate()
            torch.save(lstm.state_dict(), os.path.join(self.save_dir, f"lstm_raw_signal_{prefix}.pt"))

    def extract_features(self, emg_data, fatigue_labels):
        """Slice EMG into overlapping windows and extract time, frequency, and nonlinear features."""
        window_size = int(self.window_s * EMG_FS)
        step_size = int(self.step_s * EMG_FS)
        feats, labels = [], []

        n_windows = max(1, int((len(emg_data) - window_size) // step_size + 1))
        label_interp = np.interp(
            np.linspace(0, len(fatigue_labels) - 1, n_windows),
            np.arange(len(fatigue_labels)),
            fatigue_labels,
        )

        for i in range(0, len(emg_data) - window_size, step_size):
            seg = emg_data[i:i + window_size]
            if self.processor.is_artifact(seg):
                continue
            t = self.processor.extract_features(seg)
            f = self.processor.extract_frequency_features(seg)
            n = self.processor.extract_nonlinear_features(seg)
            feats.append(np.r_[t, f, n])
            labels.append(label_interp[int(i / step_size)])

        return np.array(feats), np.array(labels)