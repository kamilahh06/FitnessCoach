import os, numpy as np, pandas as pd
from .config import EMG_DIR, LABEL_DIR, EMG_FS, DEFAULT_WINDOW_S, DEFAULT_STEP_S
from .data_utils import list_subjects
# from .Data_Processing import Participant, EMGWindow
from .PreProcessor import PreProcessor
from .Participant import Participant
from .EMGWindow import EMGWindow
# from .Data_Processing.EMGDataset import TrainingDataset


def create_participant_list():
    root = EMG_DIR
    subs = list_subjects(root)
    return [Participant(s) for s in subs]

def iterate_windows(trial_dict, window_s=DEFAULT_WINDOW_S, step_s=DEFAULT_STEP_S, normalize_label_to_7=False):
    # trial duration from EMG time column
    df = trial_dict["emg_df"]; tcol = trial_dict["time_col"]
    tmin, tmax = float(df[tcol].min()), float(df[tcol].max())
    win, step = window_s, step_s

    windows = []
    t = tmin
    while t + win <= tmax:
        windows.append(EMGWindow(trial_dict, t, t + win, fs=EMG_FS, normalize_label_to_7=normalize_label_to_7))
        t += step
    return windows

def get_all_windows(normalize_label_to_7=False, subjects=None, use_trials=None, window_s=2.0, step_s=1.0):
    all_windows = []
    parts = create_participant_list()
    for p in parts:
        if subjects and p.subject_id not in subjects: continue
        for tr in p.trials:
            if use_trials and tr["name"].lower() not in [u.lower() for u in use_trials]: 
                continue
            all_windows.extend(iterate_windows(tr, window_s, step_s, normalize_label_to_7))
    return all_windows