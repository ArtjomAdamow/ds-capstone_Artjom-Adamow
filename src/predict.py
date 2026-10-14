"""Load a saved model and predict on a CSV.

Run with:  uv run python -m src.predict   (or: make predict)

Defaults come from src/config.py, so the plain command scores the test split
that train.py wrote. Point it somewhere else with the optional arguments:

    uv run python -m src.predict --model models/model.joblib \
        --features data/new_data.csv --target data/new_target.csv
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.config import MODEL_PATH, X_TEST_PATH, Y_TEST_PATH
from src.features import engineer_features

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="saved model file")
    parser.add_argument("--features", type=Path, default=X_TEST_PATH, help="CSV of features")
    parser.add_argument(
        "--target",
        type=Path,
        default=Y_TEST_PATH,
        help="optional CSV of true values; when given, metrics are reported",
    )
    parser.add_argument("--output", type=Path, help="optional CSV to write the predictions to")
    return parser.parse_args()


def run_prediction(
    model_path: Path, features_path: Path, target_path: Path | None = None
) -> pd.Series:
    """Predict on one CSV of features and report metrics when truth is given."""
    logger.info("Loading the model from %s", model_path)
    model = joblib.load(model_path)

    X = pd.read_csv(features_path)
    X = engineer_features(X)
    predictions = pd.Series(model.predict(X), name="prediction")

    if target_path is not None and target_path.exists():
        y_true = pd.read_csv(target_path).squeeze("columns")
        rmse = mean_squared_error(y_true, predictions) ** 0.5
        logger.info("RMSE = %.3f, R2 = %.3f", rmse, r2_score(y_true, predictions))

    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    preds = run_prediction(args.model, args.features, args.target)
    if args.output:
        preds.to_csv(args.output, index=False)
        logger.info("Wrote predictions to %s", args.output)
    else:
        print(preds.head())
