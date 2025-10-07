import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from Fatigue_Model.Data_Processing.PreProcessor import PreProcessor

# Load data
emg_path = "/u/kamilah/FitnessCoach/Fatigue_Model/Datasets/Participant_Data/1/emg_data.csv"
emg = pd.read_csv(emg_path)

# Preprocess data
processor = PreProcessor(fs=1000)
filtered = processor.full_process(emg)
emg = emg.iloc[:len(filtered)].copy()
emg["filtered"] = filtered
emg["smoothed"] = pd.Series(filtered).rolling(window=50).mean()

# Feature extraction
window = 2000   # 2 seconds
step = 1000     # 1 second
fs = 1000

time_stamps = []
time_features = []
freq_features = []
nonlinear_features = []
artifact_mask = []  # track which windows were rejected

for start in range(0, len(emg) - window, step):
    segment = emg["filtered"].iloc[start:start + window].dropna().values
    t = emg["time_ms"].iloc[start + window // 2] / 1000

    if len(segment) < window * 0.9 or processor.is_artifact(segment):
        artifact_mask.append(True)
        time_stamps.append(t)
        time_features.append([np.nan] * 7)
        freq_features.append([np.nan, np.nan])
        nonlinear_features.append([np.nan] * 3)
        continue

    # Get features
    artifact_mask.append(False)
    time_stamps.append(t)
    time_features.append(processor.extract_features(segment))
    freq_features.append(processor.extract_frequency_features(segment))
    nonlinear_features.append(processor.extract_nonlinear_features(segment))

# Create dataframe
cols_time = ["mean", "std", "rms", "mav", "wl", "zc", "ssc"]
cols_freq = ["mean_freq", "median_freq"]
cols_nonlinear = ["lz_complexity", "sample_entropy", "marginal_spec_entropy"]

df_time = pd.DataFrame(time_features, columns=cols_time)
df_freq = pd.DataFrame(freq_features, columns=cols_freq)
df_nonlinear = pd.DataFrame(nonlinear_features, columns=cols_nonlinear)
df_time["time_s"] = df_freq["time_s"] = df_nonlinear["time_s"] = time_stamps
df_time["artifact"] = artifact_mask

# Plotting
fig, axes = plt.subplots(9, 1, figsize=(14, 18), sharex=True)

# (1) Filtered & Smoothed
axes[0].plot(emg["time_ms"]/1000, emg["filtered"], color="blue", label="Filtered EMG")
axes[0].plot(emg["time_ms"]/1000, emg["smoothed"], color="red", label="Smoothed EMG", linewidth=0.8)
axes[0].set_title("Filtered and Smoothed EMG")
axes[0].legend(); axes[0].grid(True)

# (2) RMS & MAV
axes[1].plot(df_time["time_s"], df_time["rms"], color="green", label="RMS")
axes[1].plot(df_time["time_s"], df_time["mav"], color="orange", label="MAV")
axes[1].fill_between(df_time["time_s"], 0, 1, where=df_time["artifact"], transform=axes[1].get_xaxis_transform(),
                     color="gray", alpha=0.15, label="Rejected")
axes[1].set_title("RMS & MAV")
axes[1].legend(); axes[1].grid(True)

# (3) Signal Dynamics
axes[2].plot(df_time["time_s"], df_time["wl"], color="blue", label="WL")
axes[2].plot(df_time["time_s"], df_time["zc"], color="brown", label="Zero Crossings")
axes[2].plot(df_time["time_s"], df_time["ssc"], color="green", label="SSC")
axes[2].fill_between(df_time["time_s"], 0, 1, where=df_time["artifact"], transform=axes[2].get_xaxis_transform(),
                     color="gray", alpha=0.15)
axes[2].set_title("Signal Dynamics")
axes[2].legend(); axes[2].grid(True)

# (4) Frequency Features (Welch-based)
axes[3].plot(df_freq["time_s"], df_freq["mean_freq"], color="steelblue", label="Mean Freq")
axes[3].plot(df_freq["time_s"], df_freq["median_freq"], color="darkorange", label="Median Freq")
axes[3].fill_between(df_freq["time_s"], 0, 1, where=df_time["artifact"], transform=axes[3].get_xaxis_transform(),
                     color="gray", alpha=0.15)
axes[3].set_title("Frequency Features (Welch PSD)")
axes[3].set_ylabel("Hz")
axes[3].legend(); axes[3].grid(True)

# (5) Activation & Energy Indicators
iemg_vals, var_vals, tkeo_vals, times = [], [], [], []
for start in range(0, len(emg) - window, step):
    segment = emg["filtered"].iloc[start:start+window].values
    t = emg["time_ms"].iloc[start + window // 2] / 1000
    if processor.is_artifact(segment):
        iemg_vals.append(np.nan)
        var_vals.append(np.nan)
        tkeo_vals.append(np.nan)
    else:
        iemg, var = processor.extract_additional_time_features(segment)
        tkeo = np.mean(segment[1:-1]**2 - segment[:-2]*segment[2:])
        iemg_vals.append(iemg); var_vals.append(var); tkeo_vals.append(t)
    times.append(t)

axes[4].plot(times, tkeo_vals, color="blue", label="TKEO")
axes[4].plot(times, iemg_vals, color="orange", label="iEMG")
axes[4].plot(times, var_vals, color="green", label="Variance")
axes[4].fill_between(times, 0, 1, where=df_time["artifact"], transform=axes[4].get_xaxis_transform(),
                     color="gray", alpha=0.15)
axes[4].set_title("Activation & Energy Indicators")
axes[4].legend(); axes[4].grid(True)

# (6) Nonlinear Features
axes[5].plot(df_nonlinear["time_s"], df_nonlinear["sample_entropy"], color="darkorange", label="Sample Entropy")
axes[5].plot(df_nonlinear["time_s"], df_nonlinear["marginal_spec_entropy"], color="green", label="Spectral Entropy")
axes[5].fill_between(df_nonlinear["time_s"], 0, 1, where=df_time["artifact"], transform=axes[5].get_xaxis_transform(),
                     color="gray", alpha=0.15)
axes[5].set_title("Nonlinear Features (Complexity & Entropy)")
axes[5].legend(); axes[5].grid(True)

# (7) RMS vs Median Frequency (Fatigue Correlation)
valid = ~df_time["artifact"]
axes[6].scatter(df_time.loc[valid, "rms"], df_freq.loc[valid, "median_freq"],
                c=df_time.loc[valid, "time_s"], cmap="plasma", s=20)
axes[6].set_title("Fatigue Correlation: RMS vs Median Frequency")
axes[6].set_xlabel("RMS (Amplitude)")
axes[6].set_ylabel("Median Frequency (Hz)")
axes[6].grid(True)

# Add after df_time and df_freq creation
df_time["rms_cum"] = df_time["rms"].expanding().mean()
df_freq["median_cum"] = df_freq["median_freq"].expanding().mean()

axes[7].plot(df_time["time_s"], df_time["rms_cum"], color="green", label="Cumulative RMS")
axes[7].plot(df_freq["time_s"], df_freq["median_cum"], color="blue", label="Cumulative Median Freq")
axes[7].set_title("Cumulative Fatigue Trend")
axes[7].legend(); axes[7].grid(True)

corr_rms_mf = df_time["rms"].corr(df_freq["median_freq"])
axes[8].text(0.1, 0.5, f"RMS vs Median Freq Correlation: {corr_rms_mf:.3f}",
             fontsize=14, transform=axes[8].transAxes)
axes[8].axis("off")

plt.tight_layout()
plt.savefig("/u/kamilah/FitnessCoach/Signal_Plotting/Visualizations/emg_feature_dashboard_final_4.png", dpi=300)
plt.show()
print("✅ Saved feature dashboard to emg_feature_dashboard_final.png")

plt.figure(figsize=(10, 5))
plt.hexbin(df_time["rms"], df_freq["median_freq"], C=df_time["time_s"], gridsize=25, cmap="inferno")
plt.colorbar(label="Time (s)")
plt.title("RMS vs Median Frequency Density Over Time")
plt.xlabel("RMS"); plt.ylabel("Median Frequency (Hz)")
plt.tight_layout()
plt.savefig("/u/kamilah/FitnessCoach/Signal_Plotting/Visualizations/fatigue_hexbin.png", dpi=300)
plt.show()