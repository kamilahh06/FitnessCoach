import pandas as pd
import datetime

class Participant:    
    """
    Class to process and label participant data based on fatigue levels and performance metrics.
    Attributes:
        participant_id (int): Unique identifier for the participant.
        fatigue_time (float): Self-reported fatigue time
        speed_drop_time (float): Timestamp when speed drop-off occurs.
        stop_time (float): Timestamp when the participant stops.
        data (pd.DataFrame): DataFrame containing the participant's emg data.
            time_ms, emg_value
    """
    def __init__(self, participant_id, data, reported_fatigue, start_time, fatigue_time, speed_drop_time, stop_time):
        self.partipant_id = participant_id
        self.data = data
        self.reported_fatigue = reported_fatigue
        self.start_time = self.convert_to_ms(start_time)  # in ms
        self.fatigue_time = self.convert_to_ms(fatigue_time)  # in ms
        self.speed_dropoff_time = self.convert_to_ms(speed_drop_time)  # in ms
        self.stop_time = self.convert_to_ms(stop_time)  # in ms
        self.crop_emg_values()
      
    def convert_to_ms(self, timestamps):
        """
        Convert a list of timestamp strings to milliseconds since the start of the first timestamp.
        Args:
            timestamps (list of str): List of timestamp strings in "MM/DD/YYYY HH:MM:SS" format ie "9/23/2025 10:35:26"
        """
        fmt = "%m/%d/%Y %H:%M:%S"
        datetimes = [datetime.strptime(ts, fmt) for ts in timestamps]
        start_time = datetimes[0]
        ms_since_start = [(dt - start_time).total_seconds() * 1000 for dt in datetimes]
        return ms_since_start  

    def _process_ratings(self, fatigue_ratings):
        """
        Convert rating timestamps into ms since start and store with ratings.
        Args:
            fatigue_ratings (pd.DataFrame): columns = ['timestamp', 'rating']
        Returns:
            pd.DataFrame: ['time_ms', 'rating']
        """
        fmt = "%m/%d/%Y %H:%M:%S"
        start_time = datetime.strptime(fatigue_ratings['timestamp'].iloc[0], fmt)
        fatigue_ratings['time_ms'] = fatigue_ratings['timestamp'].apply(
            lambda ts: (datetime.strptime(ts, fmt) - start_time).total_seconds() * 1000
        )
        return fatigue_ratings[['time_ms', 'rating']]

    def crop_emg_values(self, start_buffer=5000, end_buffer=5000):
        """
        Crop emg data to only include values between start_time + start_buffer and stop_time - end_buffer.
        Args:
            start_buffer (int): Buffer in ms to add after start_time.
            end_buffer (int): Buffer in ms to subtract before stop_time.
        """
        start_crop = self.start_time + start_buffer
        stop_crop = self.stop_time - end_buffer
        self.data = self.data[(self.data['time_ms'] >= start_crop) & (self.data['time_ms'] <= stop_crop)].reset_index(drop=True)

       