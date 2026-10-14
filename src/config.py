"""Central place for paths and settings used across the project.

Change the values here instead of hard-coding them in the scripts and notebooks.
"""

from pathlib import Path

# Project folders. PROJECT_ROOT is the folder that holds this repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Where train.py writes its outputs and predict.py reads them from.
MODEL_PATH = MODELS_DIR / "model.joblib"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

# Name of the column you want to predict.
TARGET = "target"

# Fraction of the data held out for testing, and the seed that makes the split
# reproducible. Keep the seed fixed so everyone on the team gets the same split.
TEST_SIZE = 0.2
RANDOM_SEED = 42
