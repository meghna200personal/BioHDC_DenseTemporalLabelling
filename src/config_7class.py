# src/config_7class.py
from pathlib import Path
import config

PROJECT_DIR = config.PROJECT_DIR
DATA_DIR = config.DATA_DIR
TRAIN_DIR = config.TRAIN_DIR
TEST_DIR = config.TEST_DIR

# Folder for non-destructive 7-class outputs
OUTPUT_DIR_7CLASS = PROJECT_DIR / "outputs" / "7_class_outputs"

EVENTS_6 = config.EVENTS
EVENTS_7 = config.EVENTS + ["Background"]

FS = config.FS
N_CHANNELS = config.N_CHANNELS
BAND = config.BAND

SUBJECTS = config.SUBJECTS
TRAIN_SERIES = config.TRAIN_SERIES
VAL_SERIES = config.VAL_SERIES