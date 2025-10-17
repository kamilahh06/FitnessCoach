# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime as time
# from Fatigue_Model.Data_Processing.PreProcessor import PreProcessor
# from Fatigue_Model.pipeline import create_participant_list
# from Fatigue_Model.Model_Training.Models.TrainingPipeline import FatiguePipeline

# # === Setup ===
# fs = 1259
# processor = PreProcessor(fs=fs)
# pipeline = FatiguePipeline("/u/kamilah/FitnessCoach/Signal_Plotting/rf_biceps_8", window_s=8.0, step_s=0.5) #3, 1
# pipeline.process_trial5_biceps() 
# X, y, g = load_trial5_biceps()

# # === Build DataFrames ===
# cols_time  = ["mean","std","rms","mav","wl","zc","ssc"]
# cols_freq  = ["mean_freq","median_freq"]
# cols_nonlin= ["lz_complexity","sample_entropy","marginal_spec_entropy"]

# df_time  = pd.DataFrame(time_features, columns=cols_time)
# df_freq  = pd.DataFrame(freq_features, columns=cols_freq)
# df_nonlin= pd.DataFrame(nonlinear_features, columns=cols_nonlin)
# for df in [df_time, df_freq, df_nonlin]:
#     df["time_s"] = time_stamps
# df_time["artifact"] = artifact_mask

# # === Step 4: Visualization ===
# fig, axes = plt.subplots(8, 1, figsize=(14, 16), sharex=True)

# # (1) Raw vs Filtered Signal
# axes[0].plot(emg_df["time"], emg_df["emg"], color="gray", alpha=0.5, label="Raw EMG")
# axes[0].plot(emg_df["time"], emg_df["filtered"], color="blue", label="Filtered")
# axes[0].plot(emg_df["time"], emg_df["smoothed"], color="red", label="Smoothed", linewidth=0.8)
# axes[0].set_title("Filtered & Smoothed EMG Signal")
# axes[0].legend(); axes[0].grid(True)

# # (2) RMS & MAV
# axes[1].plot(df_time["time_s"], df_time["rms"], color="green", label="RMS")
# axes[1].plot(df_time["time_s"], df_time["mav"], color="orange", label="MAV")
# axes[1].fill_between(df_time["time_s"], 0, 1, where=df_time["artifact"],
#                      transform=axes[1].get_xaxis_transform(), color="gray", alpha=0.15)
# axes[1].set_title("Amplitude Features (RMS, MAV)")
# axes[1].legend(); axes[1].grid(True)

# # (3) Signal Dynamics (WL, ZC, SSC)
# axes[2].plot(df_time["time_s"], df_time["wl"], label="WL")
# axes[2].plot(df_time["time_s"], df_time["zc"], label="ZC")
# axes[2].plot(df_time["time_s"], df_time["ssc"], label="SSC")
# axes[2].set_title("Signal Dynamics")
# axes[2].legend(); axes[2].grid(True)

# # (4) Frequency Features
# axes[3].plot(df_freq["time_s"], df_freq["mean_freq"], label="Mean Freq", color="steelblue")
# axes[3].plot(df_freq["time_s"], df_freq["median_freq"], label="Median Freq", color="orange")
# axes[3].set_title("Frequency Features (Welch PSD)")
# axes[3].legend(); axes[3].grid(True)

# # (5) Nonlinear Features
# axes[4].plot(df_nonlin["time_s"], df_nonlin["sample_entropy"], label="Sample Entropy", color="red")
# axes[4].plot(df_nonlin["time_s"], df_nonlin["marginal_spec_entropy"], label="Spectral Entropy", color="green")
# axes[4].set_title("Nonlinear Features")
# axes[4].legend(); axes[4].grid(True)

# # (6) Fatigue correlation RMS vs Median Freq
# valid = ~df_time["artifact"]
# axes[5].scatter(df_time.loc[valid, "rms"], df_freq.loc[valid, "median_freq"],
#                 c=df_time.loc[valid, "time_s"], cmap="plasma", s=20)
# axes[5].set_title("RMS vs Median Frequency (Fatigue Trend)")
# axes[5].set_xlabel("RMS"); axes[5].set_ylabel("Median Frequency (Hz)"); axes[5].grid(True)

# # (7) Cumulative trends
# df_time["rms_cum"] = df_time["rms"].expanding().mean()
# df_freq["median_cum"] = df_freq["median_freq"].expanding().mean()
# axes[6].plot(df_time["time_s"], df_time["rms_cum"], label="Cumulative RMS", color="green")
# axes[6].plot(df_freq["time_s"], df_freq["median_cum"], label="Cumulative Median Freq", color="blue")
# axes[6].set_title("Cumulative Fatigue Trend")
# axes[6].legend(); axes[6].grid(True)

# # (8) Correlation summary
# corr_rms_mf = df_time["rms"].corr(df_freq["median_freq"])
# axes[7].text(0.1, 0.5, f"RMS vs Median Freq Correlation: {corr_rms_mf:.3f}",
#              fontsize=14, transform=axes[7].transAxes)
# axes[7].axis("off")

# plt.tight_layout()
# plt.savefig("/u/kamilah/FitnessCoach/Signal_Plotting/Visualizations/emg_feature_dashboard_v2.png", dpi=300)
# plt.show()
# print("✅ Saved dashboard → emg_feature_dashboard_v2.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Fatigue_Model.Data_Processing.PreProcessor import PreProcessor
from Fatigue_Model.Model_Training.TrainingPipeline import FatiguePipeline

# === Setup ===

print("test")
fs = 1259
processor = PreProcessor(fs=fs)
pipeline = FatiguePipeline("/u/kamilah/FitnessCoach/Signal_Plotting/rf_biceps_8", window_s=8.0, step_s=0.5)

# Process and load preprocessed data
pipeline.process_trial5_biceps()
X, y, g = pipeline.load_trial5_biceps()

# === Rebuild feature frames for visualization ===
cols = [
    "mean", "std", "rms", "mav", "wl", "zc", "ssc",
    "mean_freq", "median_freq",
    "fractal_dim", "sample_entropy", "spectral_entropy", "hurst_exp", "lz_complexity"
]
df = pd.DataFrame(X, columns=cols)
df["y"] = y
df["subject"] = g

subject_id = "subject_1"  # replace with actual ID
print(df['subject'])
df = df[df["subject"] == subject_id]

# === Time axis reconstruction (approximate)
time_stamps = np.arange(len(df)) * (pipeline.step_s)
df["time_s"] = time_stamps

# Split feature groups
df_time = df[["time_s", "mean", "std", "rms", "mav", "wl", "zc", "ssc"]].copy()
df_freq = df[["time_s", "mean_freq", "median_freq"]].copy()
df_nonlin = df[["time_s", "sample_entropy", "spectral_entropy", "lz_complexity"]].copy()

# === Visualization ===
fig, axes = plt.subplots(8, 1, figsize=(14, 16), sharex=True)

# (1) Simulated Raw vs Filtered Signal (if EMG not loaded)
axes[0].plot(df["time_s"], df["rms"], color="gray", alpha=0.5, label="Simulated EMG (RMS proxy)")
axes[0].set_title("Filtered & Smoothed EMG Signal")
axes[0].legend(); axes[0].grid(True)

# (2) RMS & MAV
axes[1].plot(df_time["time_s"], df_time["rms"], color="green", label="RMS")
axes[1].plot(df_time["time_s"], df_time["mav"], color="orange", label="MAV")
axes[1].set_title("Amplitude Features (RMS, MAV)")
axes[1].legend(); axes[1].grid(True)

# (3) Signal Dynamics (WL, ZC, SSC)
axes[2].plot(df_time["time_s"], df_time["wl"], label="WL")
axes[2].plot(df_time["time_s"], df_time["zc"], label="ZC")
axes[2].plot(df_time["time_s"], df_time["ssc"], label="SSC")
axes[2].set_title("Signal Dynamics")
axes[2].legend(); axes[2].grid(True)

# (4) Frequency Features
axes[3].plot(df_freq["time_s"], df_freq["mean_freq"], label="Mean Freq", color="steelblue")
axes[3].plot(df_freq["time_s"], df_freq["median_freq"], label="Median Freq", color="orange")
axes[3].set_title("Frequency Features (Welch PSD)")
axes[3].legend(); axes[3].grid(True)

# (5) Nonlinear Features
axes[4].plot(df_nonlin["time_s"], df_nonlin["sample_entropy"], label="Sample Entropy", color="red")
axes[4].plot(df_nonlin["time_s"], df_nonlin["spectral_entropy"], label="Spectral Entropy", color="green")
axes[4].set_title("Nonlinear Features")
axes[4].legend(); axes[4].grid(True)

# (6) Fatigue correlation RMS vs Median Freq
axes[5].scatter(df["rms"], df["median_freq"], c=df["time_s"], cmap="plasma", s=20)
axes[5].set_title("RMS vs Median Frequency (Fatigue Trend)")
axes[5].set_xlabel("RMS"); axes[5].set_ylabel("Median Frequency (Hz)"); axes[5].grid(True)

# (7) Cumulative trends
df["rms_cum"] = df["rms"].expanding().mean()
df["median_cum"] = df["median_freq"].expanding().mean()
axes[6].plot(df["time_s"], df["rms_cum"], label="Cumulative RMS", color="green")
axes[6].plot(df["time_s"], df["median_cum"], label="Cumulative Median Freq", color="blue")
axes[6].set_title("Cumulative Fatigue Trend")
axes[6].legend(); axes[6].grid(True)

# (8) Correlation summary
corr_rms_mf = df["rms"].corr(df["median_freq"])
axes[7].text(0.1, 0.5, f"RMS vs Median Freq Correlation: {corr_rms_mf:.3f}",
             fontsize=14, transform=axes[7].transAxes)
axes[7].axis("off")

plt.tight_layout()
plt.savefig("/u/kamilah/FitnessCoach/Signal_Plotting/Visualizations/emg_feature_dashboard_v3.png", dpi=300)
plt.show()
print("✅ Saved dashboard → emg_feature_dashboard_v2.png")

import seaborn as sns

# === Correlation of Each Feature with the Fatigue Label (y) ===
label_corr = df[cols + ["y"]].corr()["y"].drop("y")  # drop self-correlation

plt.figure(figsize=(10, 6))
sns.barplot(x=label_corr.values, y=label_corr.index, palette="viridis")
plt.title("Correlation of Each Feature with Fatigue Label (y)", fontsize=14)
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Feature")
plt.grid(True, axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/u/kamilah/FitnessCoach/Signal_Plotting/Visualizations/feature_label_correlation_3.png", dpi=300)
plt.show()

print("✅ Saved feature vs. label correlation plot → feature_label_correlation.png")