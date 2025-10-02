from collections import Counter
from PreProcessor import PreProcessor


class EMGWindow:
    """
    Class to represent a window of EMG data and its associated features and label.
    Attributes:
        participant (Participant): The participant object containing EMG data and fatigue info.
        curr_self_report_fatigue (int): The most relevant self-reported fatigue score within the window.
        features (tuple): Extracted features from the EMG data in the window.
        after_self_report_fatigue (bool): Whether the window is after the self-reported fatigue time.
        after_fatigue_onset (bool): Whether the window is after the fatigue onset time.
        label (int): The assigned label for the window based on fatigue levels and performance metrics.
    """
    def __init__(self, participant, window_start, window_end):
        self.processor = PreProcessor()
        self.participant = participant
        self.curr_self_report_fatigue = self.get_self_report_fatigue(participant)
        self.features = self.processor.extract_features(self.processor.filter(participant.data))
        self.after_self_report_fatigue = window_end >= self.curr_self_report_fatigue
        self.after_fatigue_onset = window_end >= participant.fatigue_time[-1] if participant.fatigue_time else False
        self.label = self.assign_label(self.after_self_report_fatigue, self.after_fatigue_onset)

    def get_self_report_fatigue(participant):
        """
        Get the most relevant self-reported fatigue score within the window.
        If no fatigue score is reported in the window, return infinity.
        """
        # Find the self-reported fatigue (participant.reported_fatigue) time between window start and window end
        reported_fatigue = participant.reported_fatigue
        if not reported_fatigue:
            return float('inf')
        
        # Filter fatigue scores within the window
        fatigue_scores_in_window = [
            ts for ts in reported_fatigue 
            if participant.start_time <= ts <= participant.stop_time
        ]
        
        if not fatigue_scores_in_window:
            return float('inf')
        
        # Count occurrences of each fatigue score
        fatigue_counts = Counter(fatigue_scores_in_window)
        
        # Return the fatigue score with the highest count
        most_relevant_fatigue = max(fatigue_counts, key=fatigue_counts.get)
        return most_relevant_fatigue

def assign_label(self, participant, window_start, window_end):
        ratings = participant.reported_fatigue
        in_window = ratings[
            (ratings['time_ms'] >= window_start) & (ratings['time_ms'] <= window_end)
        ]

        if in_window.empty:
            raw_rating = None
        else:
            raw_rating = in_window['rating'].mean()  # average if multiple ratings in window

        # Fallback if no self-report
        if raw_rating is None:
            if window_end < participant.speed_drop_time:
                return 1  # default low fatigue
            elif window_end < participant.stop_time:
                return 4  # mid fatigue
            else:
                return 7  # max fatigue

        # --- Normalize per participant ---
        # Scale to [1,7] based on min/max ratings they gave
        min_rating = participant.reported_fatigue['rating'].min()
        max_rating = participant.reported_fatigue['rating'].max()
        normalized = 1 + (raw_rating - min_rating) * (6 / (max_rating - min_rating + 1e-8))

        # Clamp based on physiology
        if window_end < participant.speed_drop_time:
            normalized = min(normalized, 3)  # pre-fatigue
        elif window_end > participant.stop_time:
            normalized = 7  # forced max

        return int(round(normalized))
    
    