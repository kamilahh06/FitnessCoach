from .TrainingPipeline import FatiguePipeline

import numpy as np, pandas as pd, torch
from sklearn.metrics import mean_squared_error, r2_score
import os 

# save_dir = "/u/kamilah/FitnessCoach/Signal_Plotting/viz_biceps_7"  # or latest run

# # === Load processed arrays ===
# X = np.load("/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows/X_raw.npy")  # or X_feat.npy if feature-based
# y = np.load("/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows/y_raw.npy")  # fatigue levels (0–1 normalized)
# groups = np.load("/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows/groups.npy")  # subject IDs
# print(f"🧠 Total EMG Windows: {len(y)}")
# print(f"Subjects: {np.unique(groups)} ({len(np.unique(groups))} total)\n")

# # === Handle NaNs ===
# valid_mask = ~np.isnan(y)
# X = X[valid_mask]
# y = y[valid_mask]
# groups = groups[valid_mask]

# # === Subject distribution ===
# subject_counts = pd.Series(groups).value_counts(normalize=True) * 100
# subject_counts = subject_counts.rename_axis("Subject").reset_index(name="% of Data")

# # === Fatigue distribution (0–1 normalized labels) ===
# bins = [0.0, 0.33, 0.66, 1.0]
# labels = ["Not Fatigued", "Onset", "Fatigued"]
# fatigue_stage = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
# fatigue_counts = fatigue_stage.value_counts(normalize=True, dropna=True).sort_index() * 100

# # === Summary stats ===
# summary = pd.DataFrame({
#     "Metric": ["Total Windows", "Subjects", "Mean Fatigue", "Std Fatigue"],
#     "Value": [len(y), len(np.unique(groups)), round(np.mean(y), 3), round(np.std(y), 3)]
# })
# print("📊 Dataset Overview:")
# print(summary.to_string(index=False))

# print("\n👥 Subject Data Percentage:")
# print(subject_counts.to_string(index=False))

# print("\n💪 Fatigue Distribution (%):")
# print(fatigue_counts.to_string())

# # === Visualization ===
# plt.figure(figsize=(15,5))

# # Subject % bar chart
# plt.subplot(1,3,1)
# plt.bar(subject_counts["Subject"].astype(str), subject_counts["% of Data"])
# plt.title("Data Percentage per Subject")
# plt.xlabel("Subject"); plt.ylabel("% of Total Windows")

# # Fatigue pie chart
# plt.subplot(1,3,2)
# plt.pie(fatigue_counts, labels=fatigue_counts.index, autopct='%1.1f%%', startangle=90)
# plt.title("Fatigue Level Distribution")

# # Histogram of fatigue scores
# plt.subplot(1,3,3)
# plt.hist(y, bins=20, color="orange", alpha=0.7, edgecolor="black")
# plt.title("Continuous Fatigue Label Distribution")
# plt.xlabel("Normalized Fatigue Level"); plt.ylabel("Count")

# plt.tight_layout()
# plt.show()
# plt.save("/u/kamilah/FitnessCoach/Fatigue_Model/Model_Training/Windows/dataset_summary.png")






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
pipeline.process_raw_emg() # re-extract raw emg data with new window size

print("Starting.")
# pipeline.train_models(use_rf=True,  prefix="FIXED_DATA_SUBSET_10")            # Feature-based RFR
# pipeline.train_models(use_cnn=True,  prefix="FIXED_DATA_SUBSET_10")            # Feature-based CNN
# pipeline.train_models(use_rawcnn=True, prefix="FIXED_DATA_SUBSET_8")           # Raw EMG CNN
# pipeline.train_models(use_lstm=True)             # Raw EMG LSTM