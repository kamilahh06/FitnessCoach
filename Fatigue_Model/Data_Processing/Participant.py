import pandas as pd
from datetime import datetime
class Participant:
    def __init__(self, participant_id, data, reported_fatigue, start_time, speed_drop_time, stop_time):
        """
        Initialize a Participant object with their data and metadata.

        Args:
            participant_id (int): The ID of the participant.
            data (pd.DataFrame): The EMG data for the participant.
            reported_fatigue (pd.DataFrame): Self-reported fatigue data.
            start_time (str): The start time in "MM/DD/YYYY HH:MM:SS" format.
            speed_drop_time (str): The speed drop time in "MM/DD/YYYY HH:MM:SS" format or 'NA'.
            stop_time (str): The stop time in "MM/DD/YYYY HH:MM:SS" format.
        """
        print("Initializing Participant:", participant_id)
        print("Data head:\n", data.head())
        print("Reported fatigue head:\n", reported_fatigue.head())
        print("Start time:", start_time)
        print("Speed drop time:", speed_drop_time)
        print("Stop time:", stop_time)

        self.participant_id = participant_id
        self.data = data
        self.reported_fatigue = reported_fatigue
        # Convert reported_fatigue['server_timestamp'] to milliseconds
        self.reported_fatigue['time_ms'] = self.reported_fatigue['server_timestamp'].apply(self.convert_to_ms)
        # then subtract start_time_ms
        self.reported_fatigue['time_ms'] -= self.convert_to_ms(start_time)
        print("Converted reported fatigue times (ms):\n", self.reported_fatigue['time_ms'].head())
        # Convert start_time to milliseconds
        self.start_time_ms = self.convert_to_ms(start_time)  # in ms

        # Calculate other times relative to start_time
        stop_ms = self.convert_to_ms(stop_time)
        speed_drop_ms = self.convert_to_ms(speed_drop_time)

        self.stop_time = stop_ms - self.start_time_ms if stop_ms is not None else None
        self.speed_dropoff_time = (
            speed_drop_ms - self.start_time_ms if speed_drop_ms is not None else None
        )

        # Set start_time to 0 ms (all other times are now relative to it)
        self.start_time = 0
        print("Converted start time (ms):", self.start_time)
        print("Converted stop time (ms):", self.stop_time)
        print("Converted speed dropoff time (ms):", self.speed_dropoff_time)

        # Crop EMG values based on the calculated times
        self.crop_emg_values()

    def convert_to_ms(self, timestamp):
        """
        Convert a timestamp string to milliseconds since the epoch.

        Args:
            timestamp (str): A timestamp string in "MM/DD/YYYY HH:MM:SS" format.

        Returns:
            int: The timestamp in milliseconds since the epoch.
        """
        if str(timestamp).strip().upper() in ['NA', 'N/A', 'NONE']:
            return None

        timestamp = str(timestamp).strip()  # Remove leading/trailing spaces
        fmt = "%m/%d/%Y %H:%M:%S"

        try:
            dt = pd.to_datetime(timestamp, format=fmt)
        except ValueError:
            # fallback: auto-detect format (useful if inconsistent input)
            dt = pd.to_datetime(timestamp, errors="coerce")
            if pd.isna(dt):
                raise ValueError(f"Could not parse timestamp: {timestamp}")

        return int(dt.timestamp() * 1000)

    def crop_emg_values(self):
        """
        Crop the EMG data to only include values within the start and stop times.
        """
        self.data = self.data[
            (self.data['time_ms'] >= self.start_time) & (self.data['time_ms'] <= self.stop_time)
        ]
        print("Cropped EMG data:\n", self.data.head())