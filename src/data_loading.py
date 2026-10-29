#claude generated
import numpy as np
import pandas as pd

import config


def load_series(subject, series, data_dir=None):
    """Return X (32, N) raw EEG and Y (6, N) binary event labels for one recording."""
    data_dir = data_dir or config.TRAIN_DIR
    data_path = data_dir / f"subj{subject}_series{series}_data.csv"
    events_path = data_dir / f"subj{subject}_series{series}_events.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Check config.TRAIN_DIR.")
    X = pd.read_csv(data_path).drop(columns="id").to_numpy(dtype=np.float64).T
    Y = pd.read_csv(events_path)[config.EVENTS].to_numpy(dtype=np.int8).T
    assert X.shape[1] == Y.shape[1], "data and events lengths differ"
    return X, Y