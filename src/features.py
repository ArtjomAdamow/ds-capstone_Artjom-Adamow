"""Feature engineering.

These are small, dataset-agnostic helpers plus one `engineer_features` entry
point. Train and predict both call `engineer_features`, so whatever you add
there is applied consistently to both splits. That is the whole point: a
transformation applied to train but forgotten in predict is one of the most
common sources of silently wrong results.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop the given columns, ignoring any that are not present."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def fill_missing_numeric(df: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    """Fill missing values column by column with the given constants.

    Pass statistics computed on the *training* set only (a mean computed over
    the test set leaks information into your evaluation).
    """
    df = df.copy()
    for column, value in values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature transformation the model expects.

    TODO: this is where your capstone's feature engineering goes, for example:

        df = drop_columns(df, ["id", "scraped_at"])
        df = fill_missing_numeric(df, {"age": age_mean_from_train})
        df["price_per_m2"] = df["price"] / df["living_area"]

    The example below just passes the data through unchanged, because the
    built-in demo dataset is already clean and numeric.
    """
    logger.info("Engineering features on %s rows", len(df))
    return df.copy()
