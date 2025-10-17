import os, re
import numpy as np
import pandas as pd
from .data_utils import list_trials, safe_read_csv
from .config import (
    EMG_DIR, LABEL_DIR, EMG_FS,
    TRIALS_TO_USE, EXACT_BICEPS_COLUMN,
    PREFERRED_MUSCLE_KEYWORDS, TRIAL_MUSCLE_HINTS
)

class Participant:
    """
    One subject with (filtered) trials. We **only** keep trial_5 and
    we **only** keep a single EMG column (right biceps).
    """
    def __init__(self, subject_id: str):
        self.subject_id = subject_id               # e.g. "subject_1"
        self.trials = []                           # list of dicts (name, emg_df, label_df, time_col, chosen_channel)
        self._load_all_trials()

    # ------------- Loading -------------
    def _load_all_trials(self):
        emg_folder   = os.path.join(EMG_DIR,   self.subject_id)
        label_folder = os.path.join(LABEL_DIR, self.subject_id)
        if not (os.path.isdir(emg_folder) and os.path.isdir(label_folder)):
            raise FileNotFoundError(f"Missing folders for {self.subject_id}")

        for trial_file in list_trials(emg_folder):
            trial_name = trial_file.split(".")[0]  # "trial_5"
            if trial_name.lower() not in [t.lower() for t in TRIALS_TO_USE]:
                continue  # ❗ keep only trial_5

            emg_path = os.path.join(emg_folder, trial_file)

            # match label file (trial_5 vs Trial_5 etc.)
            cand = [f for f in os.listdir(label_folder) if f.split(".")[0].lower() == trial_name.lower()]
            if not cand:
                num = re.findall(r"\d+", trial_name)[-1]
                cand = [f for f in os.listdir(label_folder)
                        if re.findall(r"\d+", f) and re.findall(r"\d+", f)[-1] == num]
            if not cand:
                continue
            label_path = os.path.join(label_folder, cand[0])

            emg_df   = safe_read_csv(emg_path)
            label_df = safe_read_csv(label_path)

            # normalize columns
            emg_df.columns   = [str(c).strip() for c in emg_df.columns]
            label_df.columns = [str(c).strip().lower() for c in label_df.columns]

            # time column (often first; contains "[s]" or "time")
            time_col = next((c for c in emg_df.columns
                             if "[s]" in c or c.lower() in ("time","t","seconds")), emg_df.columns[0])

            # choose the single EMG channel we will use
            emg_cols = [c for c in emg_df.columns if c != time_col]
            chosen = self._choose_biceps_channel(emg_cols)

            # squash EMG df to [time_col, chosen] only to avoid accidental multi-channel use
            emg_df = emg_df[[time_col, chosen]].copy()

            # ensure label df has 'time' + 'label'
            if "time" not in label_df.columns:
                label_df.rename(columns={label_df.columns[0]: "time"}, inplace=True)
            if "label" not in label_df.columns:
                label_df.rename(columns={label_df.columns[1]: "label"}, inplace=True)

            self.trials.append(dict(
                name=trial_name,
                emg_df=emg_df,
                label_df=label_df,
                time_col=time_col,
                channels=[chosen],
                chosen_channel=chosen,
                exercise="trial_5"
            ))

    def _choose_biceps_channel(self, emg_cols):
        # 1) exact header wins
        for c in emg_cols:
            if c.strip().lower() == EXACT_BICEPS_COLUMN.strip().lower():
                return c
        # 2) hint / keywords
        hint = TRIAL_MUSCLE_HINTS.get("trial_5", None)
        if hint:
            for c in emg_cols:
                if hint.lower() in c.lower():
                    return c
        for kw in PREFERRED_MUSCLE_KEYWORDS:
            for c in emg_cols:
                if kw.lower() in c.lower():
                    return c
        # 3) last resort: first channel
        return emg_cols[0]