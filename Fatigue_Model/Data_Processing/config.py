# --- Dataset roots ---
DATA_ROOT = "/u/kamilah/FitnessCoach/sEMG_Training_Data"
EMG_DIR   = f"{DATA_ROOT}/sEMG_data"
LABEL_DIR = f"{DATA_ROOT}/self_perceived_fatigue_index"

# --- Sampling ---
EMG_FS = 1259
LABEL_FS = 50
DEFAULT_WINDOW_S = 3.0
DEFAULT_STEP_S   = 1

# --- What we will use (hard requirement for this project) ---
TRIALS_TO_USE = ["trial_5"]                      # only this trial
EXACT_BICEPS_COLUMN = "R BICEPS BRACHII: EMG 1 [V]"  # preferred exact header

# Fallbacks if the exact header is a little different on some files
PREFERRED_MUSCLE_KEYWORDS = ["R BICEPS", "BICEPS", "BICEP", "BRACHII"]

# (kept for completeness; not needed once EXACT_BICEPS_COLUMN is found)
TRIAL_MUSCLE_HINTS = {
    "trial_5": "BICEPS",
}