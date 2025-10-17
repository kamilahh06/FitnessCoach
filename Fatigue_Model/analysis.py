import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ---- internal imports ----
from .Data_Processing.pipeline import create_participant_list
from .Data_Processing.PreProcessor import PreProcessor
from .Data_Processing.config import EMG_DIR, LABEL_DIR, EMG_FS, PREFERRED_MUSCLE_KEYWORDS, TRIAL_MUSCLE_HINTS, DEFAULT_WINDOW_S, DEFAULT_STEP_S



# ==========================================================
# 1️⃣  MAIN ANALYSIS FUNCTION
# ==========================================================
def run_statistical_analysis(df_time, df_freq, df_nonlin, fatigue_labels, save_dir):
    """
    Perform statistical + visual analysis for fatigue features.
    Works on time, frequency, and nonlinear feature sets.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.concat([df_time, df_freq, df_nonlin], axis=1)
    df["fatigue_label"] = fatigue_labels

    print(f"\n📊 Running statistical analysis on {len(df)} valid windows...")

    # skip empty
    if len(df) == 0:
        print("⚠️ No data provided to statistical analysis.")
        return

    if df["fatigue_label"].nunique() < 2:
        print("⚠️ Only one label value detected; skipping hypothesis tests.")
        return

    # ------------------------------------------------------
    #  ANOVA + Correlations + Effect Sizes
    # ------------------------------------------------------
    def cohens_d(a, b):
        if len(a) < 2 or len(b) < 2:
            return np.nan
        return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a)+np.var(b))/2 + 1e-8)

    cols_time = ["mean", "std", "rms", "mav", "wl", "zc", "ssc"]
    cols_freq = ["mean_freq", "median_freq"]
    cols_nonlin = ["fractal_dim", "sample_entropy", "spectral_entropy"]
    all_features = cols_time + cols_freq + cols_nonlin

    early = df[df["fatigue_label"] <= 0.5]
    late  = df[df["fatigue_label"] >= 1.5]

    rows = []
    for f in all_features:
        if f not in df.columns:
            continue
        groups = [g[f].dropna() for _, g in df.groupby(pd.cut(df["fatigue_label"], bins=3))]
        p = stats.f_oneway(*groups).pvalue if all(len(g) > 1 for g in groups) else np.nan
        d = cohens_d(early[f], late[f])
        r = df["fatigue_label"].corr(df[f])
        rows.append([f, early[f].mean(), late[f].mean(), d, r, p])

    stats_df = pd.DataFrame(rows, columns=["Feature", "Mean (Early)", "Mean (Late)",
                                           "Cohen_d", "Corr(fatigue)", "pval"])
    stats_df["Signif(p<.05)"] = stats_df["pval"] < 0.05
    stats_df.sort_values("pval", inplace=True)

    print("\n🧪 FEATURE SIGNIFICANCE RESULTS:\n")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(os.path.join(save_dir, "feature_significance_table.csv"), index=False)

    # ------------------------------------------------------
    #  Visualization Section
    # ------------------------------------------------------
    sns.set(style="whitegrid")

    # Effect Sizes
    plt.figure(figsize=(10, 5))
    sns.barplot(data=stats_df, x="Feature", y="Cohen_d", hue="Signif(p<.05)")
    plt.title("Effect Sizes (Early vs Late)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "effect_sizes.png"), dpi=200)
    plt.close()

    # Correlation Heatmap
    corr = df[[c for c in all_features if c in df.columns] + ["fatigue_label"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "corr_heatmap.png"), dpi=200)
    plt.close()

    # Boxplots for top features
    top = stats_df.nsmallest(3, "pval")["Feature"].tolist()
    if top:
        fig, axes = plt.subplots(1, len(top), figsize=(5 * len(top), 5))
        for i, feat in enumerate(top):
            sns.boxplot(x=pd.cut(df["fatigue_label"], bins=5), y=df[feat], ax=axes[i])
            axes[i].set_title(f"{feat} by Fatigue Quintiles")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "boxplots.png"), dpi=200)
        plt.close()

    print(f"✅ Finished statistical analysis. Plots saved to: {save_dir}\n")


# ==========================================================
# 2️⃣  MAIN EXECUTION PIPELINE
# ==========================================================
def main_stats():
    print("🚀 Starting fatigue analysis pipeline (trial_5, right biceps)...")

    participants = create_participant_list()
    processor = PreProcessor(fs=EMG_FS)

    all_time_feats, all_freq_feats, all_nonlin_feats, all_fatigue_labels = [], [], [], []
    window_size = int(DEFAULT_WINDOW_S * EMG_FS)
    step_size = int(DEFAULT_STEP_S * EMG_FS)

    for p in participants:
        print(f"\n🧍 Processing {p.subject_id}")
        for trial in p.trials:
            # ✅ Use only trial_5
            if trial["name"].lower() != "trial_5":
                continue

            emg_df = trial["emg_df"]
            label_df = trial["label_df"]
            label_df["label"] = (label_df["label"] - label_df["label"].min()) / (
                label_df["label"].max() - label_df["label"].min() + 1e-8
            )

            # ✅ Use only R BICEPS BRACHII channel
            target_cols = [c for c in emg_df.columns if "biceps" in c.lower() and "r" in c.lower()]
            if not target_cols:
                print(f"⚠️ No right biceps channel found for {p.subject_id}")
                continue
            emg_col = target_cols[0]
            emg_data = pd.to_numeric(emg_df[emg_col], errors="coerce").fillna(0).values

            fatigue_labels = label_df["label"].values
            if len(np.unique(fatigue_labels)) < 2:
                print(f"⚠️ Skipping {trial['name']} (no variation in fatigue labels)")
                continue

            n_windows = max(1, int((len(emg_data) - window_size) // step_size + 1))
            if n_windows < 2:
                print(f"⚠️ Skipping {trial['name']} — too short for windowing (len={len(emg_data)})")
                continue

            fatigue_labels_interp = np.interp(
                np.linspace(0, len(fatigue_labels) - 1, n_windows),
                np.arange(len(fatigue_labels)),
                fatigue_labels
            )

            emg_data = processor.full_process(emg_data)

            for i in range(0, len(emg_data) - window_size, step_size):
                seg = emg_data[i:i + window_size]
                if processor.is_artifact(seg):
                    continue

                t_feats = processor.extract_features(seg)
                f_feats = processor.extract_frequency_features(seg)
                nl_feats = processor.extract_nonlinear_features(seg)

                all_time_feats.append(t_feats)
                all_freq_feats.append(f_feats)
                all_nonlin_feats.append(nl_feats)
                all_fatigue_labels.append(fatigue_labels_interp[int(i / step_size)])

    # Build feature DataFrames
    cols_time = ["mean", "std", "rms", "mav", "wl", "zc", "ssc"]
    cols_freq = ["mean_freq", "median_freq"]
    cols_nonlin = ["fractal_dim", "sample_entropy", "spectral_entropy", "hurst_exp", "lz_complexity"]

    df_time = pd.DataFrame(all_time_feats, columns=cols_time)
    df_freq = pd.DataFrame(all_freq_feats, columns=cols_freq)
    df_nonlin = pd.DataFrame(all_nonlin_feats, columns=cols_nonlin)
    fatigue_labels = np.array(all_fatigue_labels)

    save_dir = "/u/kamilah/FitnessCoach/Signal_Plotting/viz_biceps_only"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n✅ Extracted {len(df_time)} windows from trial_5 (R Biceps).")
    print("Final label distribution:", np.unique(fatigue_labels, return_counts=True))

    run_statistical_analysis(df_time, df_freq, df_nonlin, fatigue_labels, save_dir)


# ==========================================================
# 3️⃣  SCRIPT ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main_stats()