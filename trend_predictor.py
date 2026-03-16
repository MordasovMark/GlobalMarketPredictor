from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Dict, Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


LOGGER_NAME: Final[str] = "trend_predictor"
DEFAULT_INPUT_CSV: Final[Path] = Path("data") / "merged_dataset.csv"
DEFAULT_MODEL_PATH: Final[Path] = Path("models") / "trend_rf_model.joblib"

FEATURE_COLUMNS: Final[list[str]] = [
    "sentiment_score",
    "sentiment_5d_ma",
    "close_5d_ma",
    "daily_return_pct",
]
TARGET_COLUMN: Final[str] = "Target"


logger = logging.getLogger(LOGGER_NAME)


def _configure_logging() -> None:
    """
    Configure basic logging for the trend predictor module.

    This sets up a simple stream handler with an informative log format.
    It is safe to call multiple times; repeated calls will not add
    duplicate handlers.
    """
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def build_and_evaluate_model(
    input_csv: str = str(DEFAULT_INPUT_CSV),
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> Dict[str, Any]:
    """
    Build and evaluate a Random Forest model for market trend prediction.

    This function performs the following steps:

    1. Loads the merged dataset containing stock and sentiment features.
    2. Constructs a binary `Target` variable indicating whether the next
       day's closing price is strictly higher than today's closing price.
    3. Engineers additional features:
       - `sentiment_5d_ma`: 5-day rolling mean of `sentiment_score`.
       - `close_5d_ma`: 5-day rolling mean of `Close`.
       - `daily_return_pct`: daily percentage return based on `Close`.
    4. Drops rows with NaN values produced by shifting and rolling
       operations.
    5. Splits the data chronologically into training (80%) and testing
       (20%) sets (no shuffling).
    6. Trains a `RandomForestClassifier` and evaluates it on the test set.
    7. Prints the accuracy score and full classification report.
    8. Saves the trained model to disk using `joblib`.

    Parameters
    ----------
    input_csv : str, optional
        Path to the merged dataset CSV file. Defaults to
        `"data/merged_dataset.csv"`.
    model_path : str, optional
        Path where the trained model will be saved. Defaults to
        `"models/trend_rf_model.joblib"`.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing at least the following keys:
        - `accuracy`: float accuracy score on the test set.
        - `classification_report`: string with the classification report.

    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist.
    ValueError
        If required columns are missing or if there is insufficient data
        after feature engineering.
    RuntimeError
        If an unexpected error occurs while reading data, training the
        model, evaluating, or saving the model.
    """
    _configure_logging()

    input_path = Path(input_csv)
    model_output_path = Path(model_path)

    logger.info("Starting model build and evaluation using %s", input_path)

    if not input_path.is_file():
        msg = f"Input CSV file not found at {input_path}."
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to read input CSV file at {input_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if df.empty:
        msg = f"Input CSV file at {input_path} is empty."
        logger.error(msg)
        raise ValueError(msg)

    # Determine which column represents the closing price. Depending on how
    # the CSV was generated (e.g., from yfinance with multi-level headers),
    # the close price may not be labeled exactly as 'Close'.
    close_column_candidates = [
        "Close",
        "close",
        "Adj Close",
        "adj_close",
        "Adj_Close",
        "Price",
        "price",
        "Unnamed: 1",  # common when the header rows are collapsed
    ]
    close_col = None
    for candidate in close_column_candidates:
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        msg = (
            "Could not identify a close-price column. Expected one of "
            f"{close_column_candidates!r}, but found columns: {list(df.columns)!r}."
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Using '%s' column as the Close price.", close_col)

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Feature engineering
    try:
        df[TARGET_COLUMN] = (df[close_col].shift(-1) > df[close_col]).astype(int)

        if "sentiment_score" not in df.columns:
            msg = "Input CSV must contain a 'sentiment_score' column."
            logger.error(msg)
            raise ValueError(msg)

        df["sentiment_5d_ma"] = df["sentiment_score"].rolling(
            window=5,
            min_periods=5,
        ).mean()
        df["close_5d_ma"] = df[close_col].rolling(
            window=5,
            min_periods=5,
        ).mean()
        df["daily_return_pct"] = df[close_col].pct_change()

        df = df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS).copy()
    except Exception as exc:  # noqa: BLE001
        msg = "Failed during feature engineering for model training."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if df.empty:
        msg = (
            "No data available after feature engineering and NaN removal. "
            "Cannot train model."
        )
        logger.error(msg)
        raise ValueError(msg)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        msg = f"Missing required feature columns: {missing_features!r}."
        logger.error(msg)
        raise ValueError(msg)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if len(df) < 10:
        msg = (
            "Insufficient data for a reliable train/test split; "
            f"need at least 10 rows, found {len(df)}."
        )
        logger.error(msg)
        raise ValueError(msg)

    split_index = int(len(df) * 0.8)
    if split_index == 0 or split_index == len(df):
        msg = "Train/test split resulted in an empty train or test set."
        logger.error(msg)
        raise ValueError(msg)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    logger.info(
        "Training data size: %d rows; Test data size: %d rows.",
        len(X_train),
        len(X_test),
    )

    try:
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        model.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001
        msg = "Failed while training the RandomForestClassifier."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
    except Exception as exc:  # noqa: BLE001
        msg = "Failed while generating predictions or evaluation metrics."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    logger.info("Model accuracy on test set: %.4f", acc)
    logger.info("Classification report:\n%s", report)

    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    try:
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output_path)
        logger.info("Trained model saved to %s.", model_output_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to save trained model to {model_output_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    return {
        "accuracy": acc,
        "classification_report": report,
    }


if __name__ == "__main__":
    try:
        results = build_and_evaluate_model()
        print(
            "Model training and evaluation pipeline completed successfully. "
            f"Model saved to {DEFAULT_MODEL_PATH}. "
            f"Test accuracy: {results['accuracy']:.4f}"
        )
    except Exception as exc:  # noqa: BLE001
        _configure_logging()
        logger.exception("Model training and evaluation failed: %s", exc)
        raise

