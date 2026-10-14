"""Loading and splitting the data.

Replace `load_data` with your own capstone dataset. Everything downstream
(features, train, predict) works with whatever DataFrame you return here.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    X_TEST_PATH,
    Y_TEST_PATH,
)

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Return the full dataset as one DataFrame, target column included.

    TODO: replace this with your own data. Two common patterns:

        return pd.read_csv(DATA_DIR / "my_dataset.csv")
        return pd.read_sql(query, engine)

    Until then it loads a small built-in scikit-learn dataset so the pipeline
    runs end to end out of the box.
    """
    from sklearn.datasets import load_diabetes

    logger.info("Loading the example dataset (replace me with your own data)")
    dataset = load_diabetes(as_frame=True)
    df = dataset.frame
    return df.rename(columns={"target": TARGET})


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train and test sets, features and target apart."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    logger.info("Splitting into train and test (test_size=%s)", TEST_SIZE)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


def save_test_data(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Write the test split to data/ so predict.py can score against it later."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    logger.info("Saved test data to %s and %s", X_TEST_PATH, Y_TEST_PATH)
