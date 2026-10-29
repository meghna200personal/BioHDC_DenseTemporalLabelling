#claude generated
import numpy as np
from scipy.signal import butter, lfilter

import config

_B, _A = butter(5, np.array(config.BAND) / (config.FS / 2), btype="bandpass")


def bandpass(X):
    """Causal 7-30 Hz bandpass on (channels, time). Uses only past samples."""
    return lfilter(_B, _A, X, axis=1)


def bandpass_macs():
    """Multiply-accumulates per channel per sample for the filter above."""
    return len(_B) + len(_A) - 1