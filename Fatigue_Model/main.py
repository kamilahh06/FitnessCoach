import os
import torch
from Data_Processing import Participant, EMGWindow, EMGDataset
from Fatigue_Model.Model_Training import LSTM_Model
from Model_Training import Trainer
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
            start_time, fatigue_time, speed_drop_time, stop_time are in "MM/DD/YYYY HH:MM:SS" format
    """
    participant_list = []
    for folder_name in os.listdir(file_path):
        if folder_name.startswith("Participant Data"):
            folder_path = os.path.join(file_path, folder_name)
            for participant_folder in os.listdir(folder_path):
                participant_path = os.path.join(folder_path, participant_folder)
                if os.path.isdir(participant_path):
                    self_reported_fatigue = load_csv(os.path.join(participant_path, 'self_reported_fatigue.csv'))
                    emg_data = load_csv(os.path.join(participant_path, 'emg_data.csv'))
                    metadata_df = load_csv(os.path.join(participant_path, 'metadata.csv'))
                    metadata = {
                        'participant_id': metadata_df['participant_id'].iloc[0],
                        'start_time': metadata_df['start_time'].iloc[0],
                        'fatigue_time': metadata_df['fatigue_time'].iloc[0],
                        'speed_drop_time': metadata_df['speed_drop_time'].iloc[0],
                        'stop_time': metadata_df['stop_time'].iloc[0]
                    }
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
    participant_id = metadata['participant_id']
    start_time = metadata['start_time']
    fatigue_time = metadata['fatigue_time']
    speed_drop_time = metadata['speed_drop_time']
    stop_time = metadata['stop_time']
    
    participant = Participant(participant_id, emg_data, self_reported_fatigue, start_time, fatigue_time, speed_drop_time, stop_time)
    return participant


def iterate_windows(participant, window_size=2000, step_size=1000):
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
    participants = create_participant_list(file_path)
    
    all_windows = []
    for participant in participants:
        windows = iterate_windows(participant)
        all_windows.extend(windows)
    
    return all_windows


def main():
    # Load all windows from participants
    all_windows = get_windows("Fatigue_Model/Datasets/Participant_Data")

    # Create full dataset
    full_dataset = EMGDataset(all_windows)

    # Split train/test
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    
    # Define model
    input_size = 7   # you extract 7 features per window
    # num_classes = 3  # Not Fatigue / Early Fatigue / Fatigue
    num_classes = 7 # 7-point fatigue scale
    model = LSTM_Model(input_size=input_size, num_classes=num_classes)

    # Initialize trainer
    trainer = Trainer(model, train_dataset, test_dataset, lr=1e-3, batch_size=32)

    # Train
    trainer.train(epochs=10)

    # Evaluate
    trainer.evaluate()

    # Save PyTorch model
    suffix = "_v1" # Update version as needed
    torch.save(model.state_dict(), f"Fatigue_Model/Models/fatigue_model{suffix}.pth")

if __name__ == "__main__":
    main()