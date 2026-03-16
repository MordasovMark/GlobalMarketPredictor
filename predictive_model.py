from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Final, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

from config import HORIZONS


LOGGER_NAME: Final[str] = "predictive_model"
DEFAULT_INPUT_CSV: Final[Path] = Path("data") / "merged_dataset.csv"
DEFAULT_MODEL_PATH: Final[Path] = Path("models") / "multi_horizon_model.joblib"
DEFAULT_REGRESSOR_PATH: Final[Path] = Path("models") / "return_regressor.joblib"

TARGET_COLUMNS: Final[List[str]] = [
    f"target_{days}d" for days in HORIZONS.values()
]
TARGET_RETURN_COLUMN: Final[str] = "target_return"
ASSET_TYPE_COLUMN: Final[str] = "Asset_Type"
SECTOR_COLUMN: Final[str] = "Sector"
GEO_RISK_COLUMN: Final[str] = "geopolitical_risk_score"

BASE_FEATURE_COLUMNS: Final[List[str]] = [
    "sentiment_score",
    "sentiment_5d_ma",
    "close_5d_ma",
    "close_20d_ma",
    "daily_return_pct",
    "rsi_14",
]
# Same 4 features the API sends for inference (regressor only uses these).
REGRESSOR_FEATURE_COLUMNS: Final[List[str]] = [
    "sentiment_score",
    "sentiment_5d_ma",
    "close_5d_ma",
    "daily_return_pct",
]


logger = logging.getLogger(LOGGER_NAME)


def _configure_logging() -> None:
    """
    Configure basic logging for the predictive model module.

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


def _detect_close_column(df: pd.DataFrame) -> str:
    """
    Detect the column representing the closing price in a DataFrame.

    This mirrors the logic used elsewhere in the project and supports
    different header conventions, including yfinance-style CSVs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing price information.

    Returns
    -------
    str
        Name of the detected price column.

    Raises
    ------
    ValueError
        If no suitable close-price column can be identified.
    """
    candidates = [
        "Close",
        "close",
        "Adj Close",
        "adj_close",
        "Adj_Close",
        "Price",
        "price",
        "Unnamed: 1",
    ]
    for col in candidates:
        if col in df.columns:
            logger.info("Using '%s' column as the Close price.", col)
            return col

    msg = (
        "Could not identify a close-price column. Expected one of "
        f"{candidates!r}, but found columns: {list(df.columns)!r}."
    )
    logger.error(msg)
    raise ValueError(msg)


def build_and_evaluate_multi_horizon_model(
    input_csv: str = str(DEFAULT_INPUT_CSV),
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> Dict[str, Any]:
    """
    Build and evaluate a multi-horizon, multi-ticker Random Forest model.

    This function trains a multi-output classifier that forecasts market
    direction across multiple horizons simultaneously, as defined in
    :data:`config.HORIZONS`. For the standard configuration these are:

    - ``target_1d``: Next-day direction.
    - ``target_7d``: Direction 7 days ahead.
    - ``target_30d``: Direction 30 days ahead.

    Feature set:

    - ``sentiment_score``: Daily sentiment derived from news.
    - ``sentiment_5d_ma``: 5-day moving average of sentiment_score.
    - ``close_5d_ma``: 5-day moving average of the closing price.
    - ``close_20d_ma``: 20-day moving average of the closing price
      (monthly-style trend).
    - ``daily_return_pct``: Daily percentage return based on the close.
    - ``rsi_14``: 14-period Relative Strength Index (RSI) computed from
      closing prices.
    - One-hot encoded ``Asset_Type`` (Stock/Commodity/Index, etc.).
    - Label-encoded ticker identifier (``ticker_code``) when a ticker
      column is available.

    Parameters
    ----------
    input_csv : str, optional
        Path to the merged dataset CSV file. Defaults to
        `"data/merged_dataset.csv"`.
    model_path : str, optional
        Path where the trained multi-output model will be saved. Defaults
        to `"models/trend_multi_rf_model.joblib"`.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing accuracy scores and classification
        reports for each horizon.

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

    logger.info(
        "Starting multi-horizon model build and evaluation using %s", input_path
    )

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

    close_col = _detect_close_column(df)

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    if "sentiment_score" not in df.columns:
        msg = "Input CSV must contain a 'sentiment_score' column."
        logger.error(msg)
        raise ValueError(msg)

    if ASSET_TYPE_COLUMN not in df.columns:
        df[ASSET_TYPE_COLUMN] = "Equity"
        logger.info("Added missing '%s' column with default 'Equity'.", ASSET_TYPE_COLUMN)

    # Fill missing sentiment with neutral (0) so long price history is retained for training.
    NEUTRAL_SENTIMENT: float = 0.0
    n_missing = df["sentiment_score"].isna().sum()
    df["sentiment_score"] = df["sentiment_score"].fillna(NEUTRAL_SENTIMENT)
    if n_missing > 0:
        logger.info("Filled %d missing sentiment values with neutral %.1f for full price history.", n_missing, NEUTRAL_SENTIMENT)

    # Sector/category: prefer explicit Sector column; otherwise derive from Asset_Type.
    if SECTOR_COLUMN not in df.columns:
        df[SECTOR_COLUMN] = df[ASSET_TYPE_COLUMN].astype(str)

    # Geopolitical risk: default to 0 if not present in the merged dataset.
    if GEO_RISK_COLUMN not in df.columns:
        df[GEO_RISK_COLUMN] = 0.0

    # Feature engineering
    try:
        df["sentiment_5d_ma"] = df["sentiment_score"].rolling(
            window=5,
            min_periods=1,
        ).mean()
        df["close_5d_ma"] = df[close_col].rolling(
            window=5,
            min_periods=5,
        ).mean()
        df["daily_return_pct"] = df[close_col].pct_change()

        # 20-day moving average for longer-term trend
        df["close_20d_ma"] = df[close_col].rolling(
            window=20,
            min_periods=20,
        ).mean()

        # RSI (Relative Strength Index) with a 14-period lookback
        period = 14
        delta = df[close_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # If targets are not already present (e.g., legacy datasets), create
        # them based on the configured forecast horizons.
        for label, days in HORIZONS.items():
            col_name = f"target_{days}d"
            if col_name not in df.columns:
                df[col_name] = (df[close_col].shift(-days) > df[close_col]).astype(
                    float
                )

        # Next-day percentage return for the regressor (e.g. 2.5 means +2.5%).
        df[TARGET_RETURN_COLUMN] = (
            (df[close_col].shift(-1) - df[close_col]) / df[close_col] * 100
        )

        # One-hot encode the asset type (Stock/Commodity/Index, etc.)
        asset_dummies = pd.get_dummies(
            df[ASSET_TYPE_COLUMN].astype(str),
            prefix="Asset_Type",
        )
        df = pd.concat([df, asset_dummies], axis=1)

        # One-hot encode sector/category.
        sector_dummies = pd.get_dummies(
            df[SECTOR_COLUMN].astype(str),
            prefix="Sector",
        )
        df = pd.concat([df, sector_dummies], axis=1)

        # Interaction features: geopolitical risk × sector dummy.
        interaction_cols: List[str] = []
        for col in sector_dummies.columns:
            safe_name = col.replace("Sector_", "", 1)
            feat = f"Risk_x_{safe_name}"
            df[feat] = df[GEO_RISK_COLUMN].astype(float).fillna(0.0) * df[col].astype(float).fillna(0.0)
            interaction_cols.append(feat)

        # Label-encode the ticker name as an additional categorical feature,
        # when a ticker column is available.
        ticker_col = None
        for candidate in ("ticker", "Ticker"):
            if candidate in df.columns:
                ticker_col = candidate
                break

        if ticker_col is not None:
            encoder = LabelEncoder()
            df["ticker_code"] = encoder.fit_transform(df[ticker_col].astype(str))
        else:
            logger.warning(
                "No ticker column found in input data; 'ticker_code' feature "
                "will be omitted."
            )
    except Exception as exc:  # noqa: BLE001
        msg = "Failed during feature engineering for multi-horizon model."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if df.empty:
        msg = (
            "No data available after feature engineering and NaN removal. "
            "Cannot train multi-horizon model."
        )
        logger.error(msg)
        raise ValueError(msg)

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            msg = f"Missing required target column: {col!r}."
            logger.error(msg)
            raise ValueError(msg)

     # Reconstruct the full feature set including encoded asset type and
    # ticker columns.
    asset_feature_cols = [c for c in df.columns if c.startswith("Asset_Type_")]
    sector_feature_cols = [c for c in df.columns if c.startswith("Sector_")]
    interaction_feature_cols = [c for c in df.columns if c.startswith("Risk_x_")]
    ticker_feature_cols = ["ticker_code"] if "ticker_code" in df.columns else []
    feature_columns = (
        BASE_FEATURE_COLUMNS
        + [GEO_RISK_COLUMN]
        + asset_feature_cols
        + sector_feature_cols
        + interaction_feature_cols
        + ticker_feature_cols
    )

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        msg = f"Missing required feature columns: {missing_features!r}."
        logger.error(msg)
        raise ValueError(msg)

    # Fill missing feature values with 0 so we can train on long history even
    # when some signals (news/risk) are sparse.
    df[feature_columns] = df[feature_columns].fillna(0.0)

    # Drop rows where targets are missing (from shifting).
    df = df.dropna(subset=TARGET_COLUMNS + [TARGET_RETURN_COLUMN]).copy()

    X = df[feature_columns]
    y = df[TARGET_COLUMNS].astype(int)
    y_return = df[TARGET_RETURN_COLUMN]

    if len(df) < 60:
        msg = (
            "Insufficient data for a reliable multi-horizon train/test split; "
            f"need at least 60 rows, found {len(df)}."
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
    y_train_return = y_return.iloc[:split_index]
    y_test_return = y_return.iloc[split_index:]

    logger.info(
        "Training data size: %d rows; Test data size: %d rows.",
        len(X_train),
        len(X_test),
    )

    try:
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        model = MultiOutputClassifier(base_estimator)
        model.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001
        msg = "Failed while training the multi-output RandomForestClassifier."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    try:
        y_pred = model.predict(X_test)
    except Exception as exc:  # noqa: BLE001
        msg = "Failed while generating predictions for the multi-horizon model."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    # Evaluate each horizon separately
    accuracies: Dict[str, float] = {}
    reports: Dict[str, str] = {}

    for idx, target_name in enumerate(TARGET_COLUMNS):
        y_true_h = y_test.iloc[:, idx]
        y_pred_h = y_pred[:, idx]

        acc = accuracy_score(y_true_h, y_pred_h)
        report = classification_report(y_true_h, y_pred_h)

        accuracies[target_name] = acc
        reports[target_name] = report

        logger.info("Accuracy for %s: %.4f", target_name, acc)
        logger.info("Classification report for %s:\n%s", target_name, report)

        print(f"\n=== {target_name} ===")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)

    try:
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output_path)
        logger.info("Trained multi-horizon model saved to %s.", model_output_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to save trained multi-horizon model to {model_output_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    # Train regressors for next-day percentage return.
    # - API-compatible regressor: the original 4-feature version.
    # - Event regressor: extended feature set including sector & risk interactions.
    regressor_path = model_output_path.parent / "return_regressor.joblib"
    regressor_event_path = model_output_path.parent / "return_regressor_event.joblib"
    try:
        # API-compatible
        X_train_reg = df.iloc[:split_index][REGRESSOR_FEATURE_COLUMNS].fillna(0.0)
        X_test_reg = df.iloc[split_index:][REGRESSOR_FEATURE_COLUMNS].fillna(0.0)
        regressor = RandomForestRegressor(n_estimators=200, random_state=42)
        regressor.fit(X_train_reg, y_train_return)
        y_pred_return = regressor.predict(X_test_reg)
        mae = mean_absolute_error(y_test_return, y_pred_return)
        logger.info("Return regressor (API) MAE (%%): %.4f", mae)
        joblib.dump(regressor, regressor_path)
        logger.info("Return regressor (API) saved to %s.", regressor_path)

        # Event-driven
        X_train_reg_evt = df.iloc[:split_index][feature_columns].fillna(0.0)
        X_test_reg_evt = df.iloc[split_index:][feature_columns].fillna(0.0)
        regressor_evt = RandomForestRegressor(n_estimators=300, random_state=42)
        regressor_evt.fit(X_train_reg_evt, y_train_return)
        y_pred_return_evt = regressor_evt.predict(X_test_reg_evt)
        mae_evt = mean_absolute_error(y_test_return, y_pred_return_evt)
        logger.info("Return regressor (event) MAE (%%): %.4f", mae_evt)
        joblib.dump(regressor_evt, regressor_event_path)
        logger.info("Return regressor (event) saved to %s.", regressor_event_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not train or save return regressor: %s", exc)

    return {
        "accuracies": accuracies,
        "classification_reports": reports,
    }


if __name__ == "__main__":
    try:
        results = build_and_evaluate_multi_horizon_model()
        print("\nMulti-horizon model training and evaluation completed successfully.")
        for target, acc in results["accuracies"].items():
            print(f"{target}: accuracy = {acc:.4f}")
    except Exception as exc:  # noqa: BLE001
        _configure_logging()
        logger.exception("Multi-horizon model training and evaluation failed: %s", exc)
        raise

