import os
import sys
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(save_dir, exist_ok=True)
import pandas as pd
from .Data_Processing import Participant, EMGWindow, EMGDataset
from .Data_Processing.PreProcessor import PreProcessor
from .Data_Processing.Participant import Participant
from .Data_Processing.EMGWindow import EMGWindow
from .Data_Processing.EMGDataset import TrainingDataset
from sklearn.model_selection import train_test_split


def create_participant_list(file_path):
    """
    Loop through folders labelled "Participant Data"
    Inside, there is 11 folders labelled 1- 11
    Each folder has a self_reported_fatigue.csv, emg_data.csv, and metadata.csv
    Create a Participant object for each folder and return a list of Participants
    
    Args:
        file_path: path to the main directory containing participant folders
        metadata.csv has columns ['participant_id', 'start_time', 'fatigue_time', 'speed_drop_time', 'stop_time']
            participant_id is an int
            start_time, speed_drop_time, stop_time are in "MM/DD/YYYY HH:MM:SS" format
    """
    print("Getting started")
    participant_list = []
    for folder_name in os.listdir(file_path):
        print("folder name: " + folder_name)
        if True:
            folder_path = os.path.join(file_path, folder_name)
            for participant_folder in os.listdir(folder_path):
                participant_path = os.path.join(folder_path, participant_folder)
                # print("participant path: " + participant_path)
                # print(os.path.isdir(participant_path))
                if os.path.isdir(participant_path):
                    self_reported_fatigue = load_csv(os.path.join(participant_path, 'self_reported_fatigue.csv'))
                    # print(self_reported_fatigue)
                    emg_data = load_csv(os.path.join(participant_path, 'emg_data.csv'))
                    # print(emg_data)
                    metadata_df = load_csv(os.path.join(participant_path, 'metadata.csv'))
                    # print(metadata_df.columns)
                    metadata = {
                        'participant_id': metadata_df['participant_id'].iloc[0],
                        'start_time': metadata_df['start_time'].iloc[0],
                        'speed_drop_time': metadata_df['speed_drop_time'].iloc[0],
                        'stop_time': metadata_df['stop_time'].iloc[0]
                    }
                    # print(metadata)
                    participant = create_participant(self_reported_fatigue, emg_data, metadata)
                    participant_list.append(participant)
    return participant_list

def create_participant(self_reported_fatigue, emg_data, metadata):
    """
    Create a Participant object from the given data files.
    
    Args:
        self_reported_fatigue: dataframe with columns ['time', 'fatigue_level']
        emg_data: dataframe with columns ['time_ms', 'emg_value']
        metadata: dictionary with keys ['participant_id', 'start_time', 'fatigue_time', 'speed_drop_time', 'stop_time']
    
    Returns:
        Participant object
    """
    print("making participant: ")
    participant_id = metadata['participant_id']
    start_time = metadata['start_time']
    speed_drop_time = metadata['speed_drop_time']
    stop_time = metadata['stop_time']
    
    participant = Participant(participant_id, emg_data, self_reported_fatigue, start_time, speed_drop_time, stop_time)
    return participant


def iterate_windows(participant, window_size=20000, step_size=10000):
    """
    Iterate through the participant's emg data in windows of given size and step.
    Create an EMGValue for each window.
    
    Args:
        participant: Participant object
        window_size: size of each window in ms
        step_size: step size between windows in ms
    
    Returns:
        list of EMGWindow objects
    """
    data = participant.data
    start_time = participant.start_time
    end_time = participant.stop_time
    windows = []
    
    for window_start in range(start_time, end_time - window_size + 1, step_size):
        window_end = window_start + window_size
        window_data = data[(data['time_ms'] >= window_start) & (data['time_ms'] < window_end)]
        if not window_data.empty:
            emg_window = EMGWindow(participant, window_start, window_end)
            windows.append(emg_window)
            
    return windows


def load_csv(file_path):
    """
    Load a CSV file and return a pandas DataFrame.
    
    Args:
        file_path: path to the CSV file
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(file_path)

def get_windows(file_path="Fatigue_Model/Datasets/Participant_Data"):
    """
    Load all participants and extract all EMG windows.
    Args:
        file_path: path to the main directory containing participant folders
    """
    participants = create_participant_list(file_path)
    
    all_windows = []
    for participant in participants:
        windows = iterate_windows(participant)
        all_windows.extend(windows)
    
    return all_windows

def run_statistical_analysis(df_time, df_freq, df_nonlinear, fatigue_labels, save_dir):
    """
    Perform full statistical analysis to identify fatigue-sensitive EMG features.
    Includes:
        - ANOVA across fatigue levels
        - Cohen's d (early vs late fatigue)
        - Correlations with time and fatigue label
        - Ranked significance table and visualizations
    """

    # Merge features
    df = pd.concat([df_time, df_freq, df_nonlinear], axis=1)
    df["fatigue_label"] = fatigue_labels
    df = df.dropna()
    print(f"\n📊 Running statistical analysis on {len(df)} valid windows...")

    if len(df) < 5:
        print("⚠️ Not enough valid data for testing.")
        return

    # Helper functions
    def cohens_d(x1, x2):
        """
        Compute Cohen's d effect size between two groups.
        Args:
            x1, x2: arrays of values for two groups
        """
        if len(x1) < 2 or len(x2) < 2:
            return np.nan
        return (np.mean(x1) - np.mean(x2)) / np.sqrt((np.var(x1) + np.var(x2)) / 2)

    def safe_anova(feature):
        """
        Perform one-way ANOVA safely, returning None if groups are insufficient.
        Args:
            feature: column name in df to test
        """
        groups = [g[feature].dropna() for _, g in df.groupby("fatigue_label")]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            return stats.f_oneway(*groups)
        return None

    # Compute statistics for each feature
    numeric_cols = [c for c in df.columns if df[c].dtype != "object" and c != "fatigue_label"]
    results = []
    early = df[df["fatigue_label"] <= 3]
    late = df[df["fatigue_label"] >= 5]

    for f in numeric_cols:
        corr = df["fatigue_label"].corr(df[f])
        anova = safe_anova(f)
        p_val = anova.pvalue if anova else np.nan
        d_val = cohens_d(early[f], late[f])
        mean_early, mean_late = early[f].mean(), late[f].mean()
        results.append([f, mean_early, mean_late, d_val, corr, p_val])

    stats_df = pd.DataFrame(results, columns=[
        "Feature", "Mean (Early Fatigue)", "Mean (Late Fatigue)",
        "Cohen's d", "Correlation (Fatigue)", "p-value"
    ])
    stats_df["Significant (p<0.05)"] = stats_df["p-value"] < 0.05
    stats_df.sort_values("p-value", inplace=True)

    # Print table
    print("\n🧪 FEATURE SIGNIFICANCE RESULTS:\n")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(os.path.join(save_dir, "feature_significance_table.csv"), index=False)

    # (1) Effect Size Ranking
    plt.figure(figsize=(10, 5))
    sns.barplot(data=stats_df, x="Feature", y="Cohen's d",
                hue="Significant (p<0.05)", palette="coolwarm")
    plt.title("Effect Sizes (Cohen’s d) for Fatigue vs Non-Fatigue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_effect_sizes.png"), dpi=300)
    plt.close()

    # (2) Correlation Heatmap
    corr_matrix = df[numeric_cols + ["fatigue_label"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature Correlation Heatmap (Fatigue & Time)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_correlation_heatmap.png"), dpi=300)
    plt.close()

    # (3) Feature Distributions by Fatigue
    top_feats = stats_df.nsmallest(3, "p-value")["Feature"].tolist()
    if top_feats:
        fig, axes = plt.subplots(1, len(top_feats), figsize=(14, 5))
        for i, feat in enumerate(top_feats):
            sns.boxplot(x="fatigue_label", y=feat, data=df, ax=axes[i], palette="coolwarm")
            axes[i].set_title(f"{feat} by Fatigue Level")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_boxplots.png"), dpi=300)
        plt.close()

    print(f"\n✅ Statistical analysis complete. Plots & tables saved to {save_dir}")
    print("Top features (most significant):")
    print(stats_df.head(5).to_string(index=False))


def main_stats():
    """
    Full pipeline to run statistical analysis on EMG features vs fatigue labels.
    """
    print("🚀 Starting fatigue analysis pipeline...")
    participants = create_participant_list("/u/kamilah/FitnessCoach/Fatigue_Model/Datasets")

    all_time_feats, all_freq_feats, all_nonlin_feats, all_fatigue_labels = [], [], [], []

    processor = PreProcessor(fs=1000)
    window_size = 10000   # 5 sec windows
    step_size = 1000     # 1 sec overlap

    for p_idx, participant in enumerate(participants, 1):
        print(f"\n🧍 Processing Participant {p_idx} ({participant.participant_id})")

        # Create EMG windows for this participant
        windows = iterate_windows(participant, window_size, step_size)

        for win in windows:
            data = participant.data[
                (participant.data["time_ms"] >= win.window_start) &
                (participant.data["time_ms"] < win.window_end)
            ]["emg_value"].values

            # Extract feature groups
            time_feats = processor.extract_features(data)
            freq_feats = processor.extract_frequency_features(data)
            nonlin_feats = processor.extract_nonlinear_features(data)

            all_time_feats.append(time_feats)
            all_freq_feats.append(freq_feats)
            all_nonlin_feats.append(nonlin_feats)
            all_fatigue_labels.append(win.label)

    # Make dataframes
    cols_time = ["mean", "std", "rms", "mav", "wl", "zc", "ssc"]
    cols_freq = ["mean_freq", "median_freq"]
    cols_nonlin = ["fractal_dim", "sample_entropy", "spectral_entropy"]

    df_time = pd.DataFrame(all_time_feats, columns=cols_time)
    df_freq = pd.DataFrame(all_freq_feats, columns=cols_freq)
    df_nonlinear = pd.DataFrame(all_nonlin_feats, columns=cols_nonlin)
    fatigue_labels = np.array(all_fatigue_labels)

    # Run analysis
    save_dir = "/u/kamilah/FitnessCoach/Signal_Plotting/Visualization_5"
    run_statistical_analysis(df_time, df_freq, df_nonlinear, fatigue_labels, save_dir)

    print("✅ Finished running statistical analysis.")

def main():
    # Load all windows from participants
    all_windows = get_windows("/u/kamilah/FitnessCoach/Fatigue_Model/Datasets")
    
    # Create full dataset
    # full_dataset = EMGDataset(all_windows)

    # # Split train/test
    # train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    
    # # Define model
    # input_size = 7   # you extract 7 features per window
    # # num_classes = 3  # Not Fatigue / Early Fatigue / Fatigue
    # num_classes = 7 # 7-point fatigue scale
    # model = LSTM_Model(input_size=input_size, num_classes=num_classes)

    # # Initialize trainer
    # trainer = Trainer(model, train_dataset, test_dataset, lr=1e-3, batch_size=32)

    # # Train
    # trainer.train(epochs=10)

    # # Evaluate
    # trainer.evaluate()

    # # Save PyTorch model
    # suffix = "_v1" # Update version as needed
    # torch.save(model.state_dict(), f"Fatigue_Model/Models/fatigue_model{suffix}.pth")

if __name__ == "__main__":
    # main()
    main_stats()