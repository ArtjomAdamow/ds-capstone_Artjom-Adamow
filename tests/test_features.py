"""Example tests. Copy the pattern for your own feature functions.

Run them all with:  uv run pytest   (or: make test)
"""

import numpy as np
import pandas as pd

from src.features import drop_columns, engineer_features, fill_missing_numeric


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "age": [30.0, np.nan, 50.0], "score": [1, 2, 3]})


def test_drop_columns_removes_the_listed_columns():
    result = drop_columns(sample_df(), ["id"])
    assert "id" not in result.columns
    assert "age" in result.columns


def test_drop_columns_ignores_missing_columns():
    result = drop_columns(sample_df(), ["does_not_exist"])
    assert list(result.columns) == ["id", "age", "score"]


def test_fill_missing_numeric_fills_only_the_given_column():
    result = fill_missing_numeric(sample_df(), {"age": 40.0})
    assert result["age"].tolist() == [30.0, 40.0, 50.0]


def test_engineer_features_does_not_mutate_the_input():
    df = sample_df()
    engineer_features(df)
    assert df["age"].isna().sum() == 1
