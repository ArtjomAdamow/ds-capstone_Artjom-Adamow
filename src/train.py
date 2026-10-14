"""Train a model and save it to models/.

Run with:  uv run python -m src.train   (or: make train)
"""

import logging

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_PATH, MODELS_DIR
from src.data import load_data, save_test_data, split_data
from src.features import engineer_features

logger = logging.getLogger(__name__)


def build_model() -> Pipeline:
    """Return the untrained model.

    TODO: swap in your own estimator, for example a RandomForestRegressor or a
    classifier. Keeping scaling and the estimator inside one Pipeline means the
    saved model carries its own preprocessing, so predict.py cannot forget it.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def evaluate(y_true: pd.Series, y_pred, split_name: str) -> dict[str, float]:
    """Compute and log the metrics for one split."""
    metrics = {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }
    logger.info("%s: RMSE = %.3f, R2 = %.3f", split_name, metrics["rmse"], metrics["r2"])
    return metrics


def run_training() -> Pipeline:
    """Load the data, train the model, report metrics, save the model."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Save the raw test split first, so predict.py can run the exact same
    # feature engineering on it later, just like it would on fresh data.
    save_test_data(X_test, y_test)

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    logger.info("Training the model")
    model = build_model()
    model.fit(X_train, y_train)

    evaluate(y_train, model.predict(X_train), "train")
    evaluate(y_test, model.predict(X_test), "test")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Saved the model to %s", MODEL_PATH)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training()
