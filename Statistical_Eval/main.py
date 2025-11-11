import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

DATA_DIR = "data/"
RESULTS_DIR = "results/"

os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)

participants = [f"p{str(i).zfill(2)}" for i in range(1, 13)]
blocks = ["block1", "block2"]  # block1 = with AI coach, block2 = without (or vice versa)

all_data = []

for pid in participants:
    for block in blocks:
        # Load self-report
        sr_file = f"{DATA_DIR}/self_report/{pid}_{block}_selfreport.csv"
        sr = pd.read_csv(sr_file)
        sr['data_type'] = 'self_report'
        
        # Load EMG
        emg_file = f"{DATA_DIR}/emg/{pid}_{block}_emg.csv"
        emg = pd.read_csv(emg_file)
        emg['data_type'] = 'emg'

        # Compute EMG RMS
        emg['emg_rms'] = emg.iloc[:,1].rolling(200).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)

        # Predictions (only for AI coach block)
        pred_file = f"{DATA_DIR}/predictions/{pid}_{block}_predictions.csv"
        if os.path.exists(pred_file):
            pred = pd.read_csv(pred_file)
            pred['data_type'] = 'prediction'
            merged = sr.merge(pred, on="server_timestamp", how="left").merge(emg, on="timestamp", how="left")
        else:
            pred = None
            merged = sr.merge(emg, left_on="client_timestamp", right_on="timestamp", how="left")

        merged['participant'] = pid
        merged['block'] = block
        all_data.append(merged)

df = pd.concat(all_data, ignore_index=True)

# ---------------------------------------------------
# TABLE: Mean Fatigue by Block
# ---------------------------------------------------
fatigue_summary = df.groupby(['participant', 'block'])['response'].mean().unstack()
fatigue_summary.to_csv(f"{RESULTS_DIR}/tables/mean_fatigue_by_block.csv")
print("\nMean Fatigue Saved → results/tables/mean_fatigue_by_block.csv")

# ---------------------------------------------------
# FIGURE 1: Fatigue Comparison (paired plot)
# ---------------------------------------------------
plt.figure(figsize=(8,4))
for pid in participants:
    try:
        x = ["AI Coach", "No Coach"]
        y = fatigue_summary.loc[pid].values
        plt.plot(x, y, marker="o", alpha=0.7)
    except:
        pass

plt.title("Self-Reported Fatigue: With AI vs Without")
plt.ylabel("Mean Fatigue Level")
plt.savefig(f"{RESULTS_DIR}/figures/fatigue_comparison.png")
print("Figure Saved → results/figures/fatigue_comparison.png")

# ---------------------------------------------------
# FIGURE 2: EMG RMS Change Over Time (example)
# ---------------------------------------------------
example = df[df.participant == "p01"]
plt.figure(figsize=(10,4))
plt.plot(example['emg_rms'], label="EMG RMS")
plt.title("EMG RMS Over Time (Participant 01)")
plt.ylabel("RMS")
plt.xlabel("Time index")
plt.savefig(f"{RESULTS_DIR}/figures/p01_emg_rms.png")
print("Figure Saved → results/figures/p01_emg_rms.png")

print("\n✅ Analysis complete.")

from scipy.stats import ttest_rel, wilcoxon
import seaborn as sns

# Convert to a clean comparison table
comparison = fatigue_summary.copy()
comparison.columns = ["with_ai_coach", "no_coach"]

# Drop participants missing either block
comparison = comparison.dropna()

# ---------------------------------------------------
# PAIRED T-TEST
# ---------------------------------------------------
t_stat, p_val = ttest_rel(comparison["with_ai_coach"], comparison["no_coach"])
print("\n--- Paired t-test (Fatigue With AI vs Without) ---")
print(f"t = {t_stat:.3f}, p = {p_val:.5f}")

# ---------------------------------------------------
# WILCOXON SIGNED-RANK TEST
# ---------------------------------------------------
w_stat, w_p_val = wilcoxon(comparison["with_ai_coach"], comparison["no_coach"])
print("\n--- Wilcoxon Signed-Rank Test ---")
print(f"W = {w_stat:.3f}, p = {w_p_val:.5f}")

# Save stats table
comparison.to_csv(f"{RESULTS_DIR}/tables/fatigue_block_comparison.csv")

# ---------------------------------------------------
# VIOLIN / BOXPLOT FOR GROUP COMPARISON
# ---------------------------------------------------
plt.figure(figsize=(6,5))
stacked = comparison.melt(var_name="condition", value_name="fatigue")
sns.violinplot(data=stacked, x="condition", y="fatigue", inner="box")
plt.title("Fatigue Distribution With vs Without AI Coach (1–7 Scale)")
plt.ylabel("Fatigue (1 = low, 7 = high)")
plt.savefig(f"{RESULTS_DIR}/figures/fatigue_violinplot.png")
print("Figure Saved → results/figures/fatigue_violinplot.png")