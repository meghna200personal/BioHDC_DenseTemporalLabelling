#claude generated but from config matches the one used in logreg_baseline_v1.py
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\MEGHNA\Documents\GitHub\ResearchProject_1\BioHDC_DenseTemporalLabelling")
DATA_DIR = PROJECT_DIR / "data" / "grasp-and-lift-eeg-detection"
TRAIN_DIR = DATA_DIR / "train" / "train"      # the unzip created a nested train folder
TEST_DIR = DATA_DIR / "test" / "test"        # the unzip created a nested test folder
OUTPUT_DIR = PROJECT_DIR / "outputs"

EVENTS = ["HandStart", "FirstDigitTouch", "BothStartLoadPhase",
          "LiftOff", "Replace", "BothReleased"]
FS = 500                       # sampling rate in Hz
N_CHANNELS = 32
BAND = (7, 30)                 # bandpass edges in Hz

SUBJECTS = list(range(1, 13))
TRAIN_SERIES = [1, 2, 3, 4, 5, 6]   # same split for every model
VAL_SERIES = [7, 8]                 # held out, labels available