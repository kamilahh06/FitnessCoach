def create_participant_list(file_path):
    """
    Loop through folders labelled "Participant Data"
    Inside, there is 11 folders labelled 1- 11
    Each folder has a self_reported_fatigue.csv, emg_data.csv, and metadata.csv
    Create a Participant object for each folder and return a list of Participants
    
    Args:
        file_path: path to the main directory containing participant folders
    """

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

def iterate_windows(participant, window_size=2000, step_size=1000):
    """
    Iterate through the participant's emg data in windows of given size and step.
    Create an EMGValue for each window.
    
    Args:
        participant: Participant object
        window_size: size of each window in ms
        step_size: step size between windows in ms
    
    Returns:
        EMGWindow objects
    """

def load_csv(file_path):
    """
    Load a CSV file and return a pandas DataFrame.
    
    Args:
        file_path: path to the CSV file
    
    Returns:
        pd.DataFrame
    """
    
