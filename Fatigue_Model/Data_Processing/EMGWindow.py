from collections import Counter
# from PreProcessor import PreProcessor
from .PreProcessor import PreProcessor
from .Participant import Participant
import numpy as np

class EMGWindow:
    """
    Represents a window of EMG data for a participant, along with extracted features and assigned fatigue label.
    """
    def __init__(self, participant, window_start, window_end):
        self.processor = PreProcessor()
        self.participant = participant
        self.window_start = window_start
        self.window_end = window_end

        self.curr_self_report_fatigue = self.get_self_report_fatigue()
        self.features = self.processor.extract_features(self.processor.full_process(participant.data))
        self.label = self.assign_label(participant, window_start, window_end)

    def get_self_report_fatigue(self):
        """
        Find the most relevant self-reported fatigue score within the window.
        """
        reported_fatigue = self.participant.reported_fatigue
        # print("Reported Fatigue ------- ", reported_fatigue)

        if reported_fatigue.empty:
            return float('inf')

        # Filter fatigue scores within the window
        fatigue_scores_in_window = reported_fatigue[
            (reported_fatigue['time_ms'] >= self.window_start) & (reported_fatigue['time_ms'] <= self.window_end)
        ]

        if fatigue_scores_in_window.empty:
            return float('inf')

        # Find the most frequent fatigue score in the window
        fatigue_counts = Counter(fatigue_scores_in_window['response'])
        most_relevant_fatigue = max(fatigue_counts, key=fatigue_counts.get)
        return most_relevant_fatigue

    def assign_label(self, participant, window_start, window_end):
        ratings = participant.reported_fatigue.sort_values('time_ms')
        # Get all ratings before or within this window
        prev_ratings = ratings[ratings['time_ms'] <= window_end]

        if prev_ratings.empty:
            print(f"⚠️ No fatigue scores yet at window start ({window_start}-{window_end}). Assigning default label.")
            return 0

        # Most recent (last) fatigue value before the window end
        raw_rating = prev_ratings.iloc[-1]['response']

        # Optionally normalize if needed
        # min_rating = ratings['response'].min()
        # max_rating = ratings['response'].max()
        # normalized = 1 + (raw_rating - min_rating) * (6 / (max_rating - min_rating + 1e-8))
        # return int(round(normalized))
        return raw_rating