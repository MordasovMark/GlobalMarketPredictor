"""
Memory-safe regression pipeline: predict 85th percentile (Bull Case) % change for 1D, 5D, 10D.
Quantile regression (quantile=0.85) for aggressive, meaningful dashboard forecasts. Saves to ai_regression_models.pkl.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "V", "WMT",
    "UNH", "JNJ", "PG", "HD", "MA", "BAC", "DIS", "PFE", "XOM", "SPY",
]
YEARS = 10
MODEL_PATH = Path("ai_regression_models.pkl")
RANDOM_STATE = 42

# Feature params
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOL_WINDOW = 20

# HistGradientBoostingRegressor params (quantile regression: 85th percentile = Bull Case)
MAX_ITER = 200
QUANTILE = 0.85

FEATURE_COLS = ["SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "Volatility", "Daily_Return"]


def _fetch_ticker(ticker: str, years: int) -> pd.DataFrame:
    """Download daily OHLCV for one ticker."""
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=years * 365)
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if data is None or data.empty or len(data) < 50:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0).str.strip()
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="first")]
    return data


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _macd_and_signal(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd, signal


def build_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature columns and regression targets for one ticker. No look-ahead in features."""
    if "Close" not in df.columns or len(df) < 30:
        return pd.DataFrame()
    close = df["Close"].astype(float)
    ret = close.pct_change()

    # Features
    df = df.copy()
    df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    df["RSI"] = _rsi(close, RSI_PERIOD)
    macd, signal = _macd_and_signal(close)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["Volatility"] = ret.rolling(window=VOL_WINDOW, min_periods=1).std()
    df["Daily_Return"] = ret

    # Regression targets (% change)
    df["Target_1D"] = (close.shift(-1) - close) / close * 100
    df["Target_5D"] = (close.shift(-5) - close) / close * 100
    df["Target_10D"] = (close.shift(-10) - close) / close * 100

    return df[FEATURE_COLS + ["Target_1D", "Target_5D", "Target_10D"]]


def main() -> None:
    print("=" * 60)
    print("Regression pipeline: 1D / 5D / 10D % change (memory-safe)")
    print("=" * 60)

    # Initialize lists for memory-safe collection
    X_list = []
    y_1d_list = []
    y_5d_list = []
    y_10d_list = []

    # Process each ticker one by one
    for ticker in TICKERS:
        raw = _fetch_ticker(ticker, YEARS)
        if raw.empty:
            print(f"  Skip {ticker} (no data)")
            continue
        df = build_features_and_targets(raw)
        if df.empty:
            print(f"  Skip {ticker} (insufficient data after features)")
            continue
        df = df.dropna()
        if len(df) < 20:
            print(f"  Skip {ticker} (too few rows after dropna)")
            continue
        X_list.append(df[FEATURE_COLS].values)
        y_1d_list.append(df["Target_1D"].values)
        y_5d_list.append(df["Target_5D"].values)
        y_10d_list.append(df["Target_10D"].values)
        print(f"  {ticker}: {len(df)} rows")

    if not X_list:
        raise RuntimeError("No data collected for any ticker.")

    # Concatenate into master arrays
    X = np.vstack(X_list)
    y_1d = np.concatenate(y_1d_list)
    y_5d = np.concatenate(y_5d_list)
    y_10d = np.concatenate(y_10d_list)

    print(f"\nMaster arrays: X {X.shape}, y_1d {y_1d.shape}, y_5d {y_5d.shape}, y_10d {y_10d.shape}")

    # Train 3 separate HistGradientBoostingRegressors (quantile=0.85 = Bull Case, realistic upside)
    model_1d = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=QUANTILE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model_5d = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=QUANTILE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model_10d = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=QUANTILE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    print("\nTraining model_1d...")
    model_1d.fit(X, y_1d)
    print("Training model_5d...")
    model_5d.fit(X, y_5d)
    print("Training model_10d...")
    model_10d.fit(X, y_10d)

    # Store in dict and save
    models = {
        "1d": model_1d,
        "5d": model_5d,
        "10d": model_10d,
        "feature_names": FEATURE_COLS,
    }
    joblib.dump(models, MODEL_PATH)

    print(f"\nSaved 3 regression models + feature_names to {MODEL_PATH}")
    print("Success: ai_regression_models.pkl is ready for inference.")
    print("=" * 60)


if __name__ == "__main__":
    main()
