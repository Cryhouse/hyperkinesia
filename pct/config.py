from pathlib import Path
import os
from os.path import join as j
current = Path(os.getcwd())
parent = current.parent.parent.parent.absolute()
example_data = j(parent, "example_data")

RAW_DATA_DIR = j(example_data, "raw_data")
LABELS_PATH = j(example_data, "labels/labels.csv")
TEST_PATIENTS = j(example_data, "labels/test_pats.txt")
TRAIN_PATIENTS = j(example_data, "labels/train_pats.txt")
VIDEO_DIR = j(example_data, "videos")