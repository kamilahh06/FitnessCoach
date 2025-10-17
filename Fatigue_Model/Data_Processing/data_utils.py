import os, re
import pandas as pd
import csv
import numpy as np


def list_subjects(folder):
    names = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    names = sorted([n for n in names if re.match(r"subject_\d+", n, re.I)])
    return names

def list_trials(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    # keep trial files only (ignore MVC)
    files = [f for f in files if re.match(r"(trial|Trial|TRIAL)_\d+\.csv", f)]
    return sorted(files, key=lambda x: int(re.findall(r"\d+", x)[-1]))

# def safe_read_csv(path):
#     # Handles weird delimiters / encodings gracefully
#     try:
#         return pd.read_csv(path)
#     except Exception:
#         return pd.read_csv(path, engine="python")

def safe_read_csv(path):
    # expand max field size before reading
    import sys
    csv.field_size_limit(sys.maxsize)

    try:
        return pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    except Exception:
        # fallback to python engine & tolerant mode
        try:
            return pd.read_csv(
                path,
                engine="python",
                sep=None,        # auto-detects delimiter
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception as e:
            print(f"⚠️  Failed to read {path}: {e}")
            return pd.DataFrame()