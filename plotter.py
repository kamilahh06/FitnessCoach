import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

file = pd.read_csv("FileName")
#print(file.head())

def median_frequency(window,hz):
    f, Pxx = welch(window, fs=hz, nperseg=len(window))
    cumulative = np.cumsum(Pxx)
    cutoff = cumulative[-1] / 2
    mdf = f[np.where(cumulative >= cutoff)[0][0]]
    return mdf

window_size = 50
step_size = 10

#raw emg plot creation
plt.figure(figsize=(10,5))
plt.plot(file["time_ms"]/1000, file["emg_value"], linewidth=0.8)

plt.title("Raw EMG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("ADC Value")
plt.show()
#smoothing raw emg
file["smooth"] = file["emg_value"].rolling(window=50).mean()
plt.plot(file["time_ms"]/1000, file["smooth"], color="red")
plt.legend(["Raw", "Smoothed"])
plt.show()


#difference btwn timestamps for mdf calculation
intervals = np.diff(file["time_ms"])
avg_interval = np.mean(intervals)
hz = 1000/avg_interval

#extract features
rms_vals = []
rms_times= []
mdf_vals = []
mdf_times = []
mav_vals= []
features= []

trial_end = file["time_ms"].max()/1000
fatigue_start = .8*trial_end


for start in range(0, len(file)-window_size, step_size):
    window = file["emg_value"].iloc[start:start+window_size]
    rms = np.sqrt(np.mean(window**2))
    mdf = median_frequency(window, hz)
    t = file["time_ms"].iloc[start+window_size//2]/1000

    label = 1 if t >= fatigue_start else 0
    
    rms_vals.append(rms)
    rms_times.append(t)

    mdf_vals.append(mdf)
    mdf_times.append(t)

    mav = np.mean(np.abs(window))
    mav_vals.append(mav)

    features.append([t, rms, mdf, mav, label])

feat_file = pd.DataFrame(features, columns=["time_s","RMS","MDF", "MAV", "label"])


#mdf vs rmf plot
fig, ax1 = plt.subplots(figsize=(10,4))

ax1.plot(rms_times, rms_vals, color="red", label="RMS")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("RMS", color="red")

ax2 = ax1.twinx()
ax2.plot(mdf_times, mdf_vals, color="blue", label="MDF")
ax2.set_ylabel("Median Frequency (Hz)", color="blue")

plt.title("EMG Features: RMS and MDF over time")
plt.show()

#fatigued vs not fatigued plot
plt.figure(figsize=(10,4))
plt.scatter(feat_file["time_s"], feat_file["RMS"], c=feat_file["label"], cmap="coolwarm", s=10)
plt.xlabel("Time (s)")
plt.ylabel("RMS")
plt.title("RMS with fatigue labels (red = fatigued)")
plt.show()


