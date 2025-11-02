"""
Fatigue Model Package — Full pipeline from preprocessing → feature extraction → training.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
    # def process_trial5_biceps(self):
    #     """
    #     Extracts features & labels from all subjects’ trial_5 biceps EMG.
    #     Saves numpy arrays (X_feat, y_feat, groups) for later use.
    #     """
    #     print("⚙️ Processing Trial 5 Right Biceps dataset...")
    #     all_feats, all_labels, all_groups = [], [], []

    #     for p in create_participant_list():
    #         for tr in p.trials:
    #             emg = pd.to_numeric(tr["emg_df"][ch], errors="coerce").fillna(0).to_numpy()
    #             label = tr["label_df"]["label"].astype(float).to_numpy()
                
    #             if not tr["name"].lower().startswith("trial_5"):
    #                 continue
    #             ch = tr["chosen_channel"]
    #             if "bicep" not in ch.lower():
    #                 continue

    #             emg = pd.to_numeric(tr["emg_df"][ch], errors="coerce").fillna(0).to_numpy()
    #             label = tr["label_df"]["label"].astype(float).to_numpy()
    #             label = (label - label.min()) / (label.max() - label.min() + 1e-8)
    #             print(f"  → Processing {p.subject_id}:{tr['name']} ({ch})  |  len(emg)={len(emg)}")
    #             emg = self.processor.full_process(emg, normalize_mode="zscore")

    #             feats, labels = self.extract_features(emg, label)
    #             if len(feats) == 0:
    #                 continue

    #             all_feats.append(feats)
    #             all_labels.append(labels)
    #             all_groups.extend([p.subject_id] * len(labels))

    #     df = pd.DataFrame(np.vstack(all_feats), columns=[
    #         "mean","std","rms","mav","wl","zc","ssc",
    #         "mean_freq","median_freq",
    #         "fractal_dim","sample_entropy","spectral_entropy","hurst_exp","lz_complexity"
    #     ]).replace([np.inf, -np.inf], np.nan).fillna(0)
    #     y = np.concatenate(all_labels)
    #     g = np.array(all_groups)

    #     np.save(os.path.join(self.save_dir, "X_feat.npy"), df.values)
    #     np.save(os.path.join(self.save_dir, "y_feat.npy"), y)
    #     np.save(os.path.join(self.save_dir, "groups.npy"), g)
    #     print(f"✅ Saved processed data in {self.save_dir}")
    #     print(f"   Shapes — X: {df.shape}, y: {y.shape}, groups: {len(np.unique(g))}")
    
    def process_trial5_biceps(self):
        """
        Extracts time/frequency/nonlinear features from all subjects’ Trial 5 biceps EMG signals.
        Adds consistent interpolation, normalization, and summary visualizations.
        Saves numpy arrays (X_feat, y_feat, groups).
        """
        print("⚙️ Processing Trial 5 Biceps dataset...")
        all_feats, all_labels, all_groups = [], [], []
        all_subjects = []

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

                # Interpolate fatigue labels to match EMG sampling rate
                label_times = label_df["time"].astype(float).to_numpy()
                label_values = label_df["label"].astype(float).to_numpy()
                emg_times = np.arange(len(emg)) / EMG_FS + label_times[0]

                from scipy.interpolate import interp1d
                interp_func = interp1d(
                    label_times, label_values, kind="linear",
                    bounds_error=False,
                    fill_value=(label_values[0], label_values[-1])
                )
                label = interp_func(emg_times)

                # Clean invalid entries
                valid_mask = ~np.isnan(label)
                emg, label = emg[valid_mask], label[valid_mask]
                if len(label) == 0:
                    continue

                # Normalize fatigue labels
                label = (label - label.min()) / (label.max() - label.min() + 1e-8)
                emg = self.processor.full_process(emg, normalize_mode="zscore")

                # Extract features
                feats, labels = self.extract_features(emg, label)
                if len(feats) == 0:
                    continue

                all_feats.append(feats)
                all_labels.append(labels)
                all_groups.extend([p.subject_id] * len(labels))
                all_subjects.append(p.subject_id)

                print(f"  ✓ {p.subject_id}:{tr['name']} | {len(labels)} windows extracted")

        # Convert to arrays
        df = pd.DataFrame(np.vstack(all_feats), columns=[
            "mean","std","rms","mav","wl","zc","ssc",
            "mean_freq","median_freq",
            "fractal_dim","sample_entropy","spectral_entropy","hurst_exp","lz_complexity"
        ]).replace([np.inf, -np.inf], np.nan).fillna(0)

        y = np.concatenate(all_labels)
        g = np.array(all_groups)

        # Save arrays
        np.save(os.path.join(self.save_dir, "X_feat.npy"), df.values)
        np.save(os.path.join(self.save_dir, "y_feat.npy"), y)
        np.save(os.path.join(self.save_dir, "groups.npy"), g)

        print(f"\n✅ Saved processed feature data → {self.save_dir}")
        print(f"   X shape: {df.shape}, y shape: {y.shape}, subjects: {len(np.unique(g))}")

        # --- Dataset summary ---
        unique_subjects = list(set(all_subjects))
        subject_counts = pd.Series(all_subjects).value_counts(normalize=True) * 100
        mean_fatigue, std_fatigue = np.mean(y), np.std(y)
        print("\n📊 Dataset Summary:")
        print(f"Subjects processed: {len(unique_subjects)} → {unique_subjects}")
        print(f"Mean fatigue label: {mean_fatigue:.3f} ± {std_fatigue:.3f}")

        bins = [0, 0.33, 0.66, 1.0]
        labels_stage = ["Not Fatigued", "Onset", "Fatigued"]
        fatigue_stage = pd.cut(y, bins=bins, labels=labels_stage, include_lowest=True)
        fatigue_counts = pd.Series(fatigue_stage).value_counts(normalize=True).sort_index() * 100

        print("\n💪 Fatigue Stage Distribution (%):")
        print(fatigue_counts)

        # --- Visualization ---
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(style="whitegrid", palette="muted", font_scale=1.1)

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 3, 1)
        sns.histplot(y, bins=30, kde=True, color="teal")
        plt.title("Distribution of Fatigue Labels")
        plt.xlabel("Normalized Fatigue Level (0–1)")
        plt.ylabel("Count")

        plt.subplot(1, 3, 2)
        sns.barplot(x=subject_counts.index, y=subject_counts.values, color="purple")
        plt.title("Data Contribution per Subject")
        plt.xticks(rotation=30)
        plt.ylabel("Percentage of Windows")

        plt.subplot(1, 3, 3)
        sns.barplot(x=fatigue_counts.index, y=fatigue_counts.values, color="coral")
        plt.title("Fatigue Stage Breakdown")
        plt.ylabel("Percentage of Data")

        plt.suptitle("Trial 5 Biceps Feature Dataset Overview", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "trial5_feature_overview.png"))
        plt.show()
        print(f"📊 Saved visualization: trial5_feature_overview.png")

    def process_raw_emg(self):
        print("⚙️ Processing raw EMG windows for raw CNN/LSTM...")
        all_raw, all_labels, all_subjects = [], [], []
        count = 0
        number_participants = 4

        for p in create_participant_list():
            # count += 1
            # if count > number_participants:
            #     break

            print(f"→ Processing subject {p.subject_id}...")
            for tr in p.trials:
                if not tr["name"].lower().startswith("trial_5"):
                    continue
                ch = tr["chosen_channel"]
                if "bicep" not in ch.lower():
                    continue

                emg = pd.to_numeric(tr["emg_df"][ch], errors="coerce").fillna(0).to_numpy()
                label_df = tr["label_df"]

                # interpolate fatigue labels to EMG sampling rate
                label_times = label_df["time"].astype(float).to_numpy()
                label_values = label_df["label"].astype(float).to_numpy()
                emg_times = np.arange(len(emg)) / EMG_FS + label_times[0]
                interp_func = interp1d(label_times, label_values, kind="linear",
                                    bounds_error=False,
                                    fill_value=(label_values[0], label_values[-1]))
                label = interp_func(emg_times)

                # cleanup
                valid_mask = ~np.isnan(label)
                emg, label = emg[valid_mask], label[valid_mask]
                if len(label) == 0:
                    continue

                label = (label - label.min()) / (label.max() - label.min() + 1e-8)
                emg = self.processor.full_process(emg, normalize_mode="zscore")

                win = int(self.window_s * EMG_FS)
                step = int(self.step_s * EMG_FS)

                windows_created = 0
                for i in range(0, len(emg) - win + 1, step):
                    seg = emg[i:i + win]
                    label_seg = label[i:i + win]
                    if np.all(seg == 0) or np.isnan(seg).any() or np.isnan(label_seg).all():
                        continue
                    if self.processor.is_artifact(seg):
                        continue

                    all_raw.append(seg)
                    all_labels.append(np.nanmean(label_seg))
                    all_subjects.append(p.subject_id)
                    windows_created += 1

                print(f"  ✓ Created {windows_created} windows\n")

        # --- save arrays ---
        X_raw = np.stack(all_raw)
        y_raw = np.array(all_labels)
        np.save(os.path.join(self.save_dir, "X_raw.npy"), X_raw)
        np.save(os.path.join(self.save_dir, "y_raw.npy"), y_raw)
        print(f"\n✅ Saved raw EMG arrays → {self.save_dir}")
        print(f"   X shape: {X_raw.shape}")
        print(f"   y shape: {y_raw.shape}")

        # --- summary stats ---
        unique_subjects = list(set(all_subjects))
        subject_counts = pd.Series(all_subjects).value_counts(normalize=True) * 100
        print("\n📊 Dataset Summary:")
        print(f"Subjects processed: {len(unique_subjects)} → {unique_subjects}")
        print(f"Total windows: {len(y_raw)}")
        print(f"Mean fatigue label: {np.mean(y_raw):.3f} ± {np.std(y_raw):.3f}")

        # --- categorize fatigue ---
        bins = [0, 0.33, 0.66, 1.0]
        labels = ["Not Fatigued", "Onset", "Fatigued"]
        fatigue_stage = pd.cut(y_raw, bins=bins, labels=labels, include_lowest=True)
        fatigue_counts = pd.Series(fatigue_stage).value_counts(normalize=True).sort_index() * 100

        print("\n💪 Fatigue Stage Distribution (%):")
        print(fatigue_counts)

        # --- 📈 Visualization section ---
        sns.set(style="whitegrid", palette="muted", font_scale=1.1)
        plt.figure(figsize=(14, 6))

        # 1️⃣ Fatigue histogram
        plt.subplot(1, 3, 1)
        sns.histplot(y_raw, bins=30, kde=True, color="teal")
        plt.title("Distribution of Fatigue Labels")
        plt.xlabel("Normalized Fatigue Level (0–1)")
        plt.ylabel("Count")

        # 2️⃣ Subject contribution
        plt.subplot(1, 3, 2)
        sns.barplot(x=subject_counts.index, y=subject_counts.values, color="purple")
        plt.title("Data Contribution per Subject")
        plt.xticks(rotation=30)
        plt.ylabel("Percentage of Total Windows")

        # 3️⃣ Fatigue class distribution
        plt.subplot(1, 3, 3)
        sns.barplot(x=fatigue_counts.index, y=fatigue_counts.values, color="coral")
        plt.title("Fatigue Stage Breakdown")
        plt.ylabel("Percentage of Data")

        plt.suptitle("Raw EMG Dataset Overview", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "dataset_overview.png"))
        plt.show()

        print(f"📊 Saved visualization: dataset_overview.png")
        

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
            selected_features = [
                "std","rms","mav","wl","zc","ssc",
                "mean_freq",
                "fractal_dim","hurst_exp"
            ]
            # X, y, g = self.load_trial5_biceps()
            X, y, g = self.select_features(selected_features)
            print("\n🧠 Training Feature-Based CNN regression model...")
            X_seq = np.expand_dims(X, axis=2)
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
            train_ds = EMGSequenceDataset(X_train, y_train)
            test_ds = EMGSequenceDataset(X_test, y_test)

            cnn = CNNRegressor(input_size=1)
            trainer = RegressorTrainer(cnn, train_ds, test_ds, lr=1e-3, batch_size=32, model_name="CNN_Feature_Based_2")
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
            # from torch.utils.data import Subset
            # subset_idx = np.random.choice(len(train_ds), size=1000, replace=False)  # try 300 windows
            # train_small = Subset(train_ds, subset_idx)

            # trainer = RegressorTrainer(rawcnn, train_small, test_ds, lr=1e-3, batch_size=16)
            # # trainer.resume_training(epochs=48)
            # trainer.train(epochs=100)
            
            trainer = RegressorTrainer(rawcnn, train_ds, test_ds, lr=1e-3, batch_size=16, model_name="RawCNN_1")
            trainer.train(epochs=250)
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
            trainer = RegressorTrainer(lstm, train_ds, test_ds, lr=1e-3, batch_size=32, model_name="LSTM_2")
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
    

    def select_features(self, selected_features):
        """
        Selects a subset of features by name from the saved feature dataset.
        Also prints and returns a summary for verification.

        Parameters:
            selected_features (list[str]): List of feature names to include.
        Returns:
            X_selected (np.ndarray): Subset of X_feat using the selected columns.
            y_feat (np.ndarray): Fatigue labels.
            g (np.ndarray): Group labels (subjects).
        """
        # Load all features and labels
        X_feat = np.load("/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows/X_feat.npy")
        y_feat = np.load('/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows/y_feat.npy')
        g = np.load("/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows/groups.npy")

        feature_names = [
            "mean","std","rms","mav","wl","zc","ssc",
            "mean_freq","median_freq",
            "fractal_dim","sample_entropy","spectral_entropy","hurst_exp","lz_complexity"
        ]

        # Check if any provided feature name is invalid
        missing = [f for f in selected_features if f not in feature_names]
        if missing:
            raise ValueError(f"❌ Invalid feature(s): {missing}. Check your spelling or list.")

        # Select the requested columns
        selected_idx = [feature_names.index(f) for f in selected_features]
        X_selected = X_feat[:, selected_idx]

        print("\n✅ Selected Features:")
        for f in selected_features:
            print(f"  - {f}")
        print(f"→ Shape after selection: {X_selected.shape}")

        return X_selected, y_feat, g
    