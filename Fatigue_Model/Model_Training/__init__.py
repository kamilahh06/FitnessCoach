from .TrainingPipeline import FatiguePipeline

import numpy as np, pandas as pd, torch
from sklearn.metrics import mean_squared_error, r2_score

# save_dir = "/u/kamilah/FitnessCoach/Signal_Plotting/viz_biceps_7"  # or latest run

# # Load arrays
# X_feat = np.load(f"{save_dir}/X_feat.npy")
# y_feat = np.load(f"{save_dir}/y_feat.npy")

# print("📊 Dataset summary:")
# print(f"Samples: {len(y_feat)}, Features: {X_feat.shape[1]}")
# print(f"y range: {y_feat.min():.2f} → {y_feat.max():.2f}")

# # If you have predictions saved:
# try:
#     preds = np.load(f"{save_dir}/rf_preds.npy")
#     print("RF  RMSE:", mean_squared_error(y_feat, preds, squared=False))
#     print("RF  R²:", r2_score(y_feat, preds))
# except:
#     print("No saved preds found.")

# For CNN checkpoint metrics:
# print("\nCNN Results (from trainer logs if available)")
# # Paste printed train/test logs here manually if needed

pipeline = FatiguePipeline("/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows", window_s=8.0, step_s=0.5) #3, 1
# pipeline.process_trial5_biceps()   # re-extract features with new window size
# pipeline.process_raw_emg() # re-extract raw emg data with new window size

print("Starting.")
# pipeline.train_models(use_rf=True)            # Feature-based RFR
# pipeline.train_models(use_cnn=True)            # Feature-based CNN
pipeline.train_models(use_rawcnn=True, prefix="test3")           # Raw EMG CNN
# pipeline.train_models(use_lstm=True)             # Raw EMG LSTM