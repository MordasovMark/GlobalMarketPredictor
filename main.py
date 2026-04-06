from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import feedparser
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import Pipeline, pipeline
from api import app as ai_api_app


LOGGER_NAME = "api"
MODEL_PATH = Path("models") / "trained_ai_model.pkl"
REGRESSOR_PATH = Path("models") / "return_regressor.joblib"
FEATURE_COLUMNS: List[str] = [
    "sentiment_score",
    "sentiment_5d_ma",
    "close_5d_ma",
    "daily_return_pct",
]
NEWS_LOOKBACK_DAYS = 7
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
GEO_RISK_KEYWORDS: List[str] = [
    "war",
    "iran",
    "missiles",
    "escalation",
    "idf",
    "strike",
    "conflict",
    "gaza",
    "lebanon",
]


logger = logging.getLogger(LOGGER_NAME)


def _configure_logging() -> None:
    """
    Configure basic logging for the FastAPI application.

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


_configure_logging()

app = FastAPI(title="Global Market Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://globalmarketpredictor.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure `/api/analyze` (implemented in `api.py`) is reachable when running
# `uvicorn main:app` locally.
app.mount("", ai_api_app)


class MarketDataInput(BaseModel):
    """
    Input schema for market trend prediction.

    This model defines the features required by the trained Random Forest
    classifier to predict whether the market trend for the next day is
    expected to move up or down.
    """

    sentiment_score: float = Field(
        ...,
        description="Current day's sentiment score derived from news.",
    )
    sentiment_5d_ma: float = Field(
        ...,
        description="5-day moving average of sentiment_score.",
    )
    close_5d_ma: float = Field(
        ...,
        description="5-day moving average of the closing price.",
    )
    daily_return_pct: float = Field(
        ...,
        description="Percentage return of today's close relative to yesterday.",
    )


model: Optional[Any] = None
return_regressor: Optional[Any] = None
sentiment_pipe: Optional[Pipeline] = None


def _load_model(path: Path = MODEL_PATH) -> Optional[Any]:
    """
    Load the trained Random Forest model from disk.

    Parameters
    ----------
    path : Path, optional
        Filesystem path to the serialized model file. Defaults to
        `models/trained_ai_model.pkl`.

    Returns
    -------
    Optional[Any]
        The loaded model instance if available; otherwise, ``None``.
    """
    if not path.is_file():
        logger.warning("Model file not found at %s. Predictions will be unavailable.", path)
        return None

    try:
        loaded_model = joblib.load(path)
        logger.info("Model successfully loaded from %s.", path)
        return loaded_model
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load model from %s: %s", path, exc)
        return None


def _load_regressor(path: Path = REGRESSOR_PATH) -> Optional[Any]:
    if not path.is_file():
        logger.warning("Return regressor not found at %s. predicted_return_pct will be omitted.", path)
        return None
    try:
        reg = joblib.load(path)
        logger.info("Return regressor loaded from %s.", path)
        return reg
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load return regressor from %s: %s", path, exc)
        return None


# Load the brain: trained Random Forest from models/trained_ai_model.pkl on startup
model = _load_model()
return_regressor = _load_regressor()


def _get_sentiment_pipeline() -> Pipeline:
    global sentiment_pipe
    if sentiment_pipe is None:
        sentiment_pipe = pipeline(task="sentiment-analysis", model=SENTIMENT_MODEL_NAME)
        logger.info("Loaded sentiment pipeline: %s", SENTIMENT_MODEL_NAME)
    return sentiment_pipe


def _geopolitical_risk_score(text: str) -> float:
    if not text:
        return 0.0
    haystack = str(text).lower()
    matches = sum(1 for kw in GEO_RISK_KEYWORDS if kw.lower() in haystack)
    if matches <= 0:
        return 0.0
    return min(1.0, matches / float(len(GEO_RISK_KEYWORDS)))


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _macd_hist_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line).fillna(0.0)


def _fetch_trained_model_features(ticker: str) -> Dict[str, float]:
    """Fetch latest daily data for ticker + ^VIX and ^TNX; compute rsi, sma_20, macd for live prediction."""
    period = "4mo"
    hist = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
    vix = yf.download("^VIX", period=period, interval="1d", progress=False, threads=False)
    tnx = yf.download("^TNX", period=period, interval="1d", progress=False, threads=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}.")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0).str.strip()
    close = hist["Close"].astype(float) if "Close" in hist.columns else hist.iloc[:, 0].astype(float)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="first")]

    rsi = _rsi_series(close, 14)
    sma_20 = close.rolling(window=20, min_periods=1).mean()
    macd_hist = _macd_hist_series(close, 12, 26, 9)

    vix_close = 20.0
    if vix is not None and not vix.empty and "Close" in vix.columns:
        vix_aligned = vix["Close"].reindex(close.index, method="ffill")
        if vix_aligned.notna().any():
            vix_close = float(vix_aligned.iloc[-1])
    tnx_close = 4.0
    if tnx is not None and not tnx.empty and "Close" in tnx.columns:
        tnx_aligned = tnx["Close"].reindex(close.index, method="ffill")
        if tnx_aligned.notna().any():
            tnx_close = float(tnx_aligned.iloc[-1])

    latest_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
    daily_return_pct = ((latest_close - prev_close) / prev_close * 100.0) if prev_close else 0.0

    return {
        "rsi": float(rsi.iloc[-1]),
        "sma_20": float(sma_20.iloc[-1]),
        "macd": float(macd_hist.iloc[-1]),
        "vix": vix_close,
        "tnx": tnx_close,
        "latest_close": latest_close,
        "daily_return_pct": daily_return_pct,
    }


def _fetch_recent_prices(ticker: str) -> Dict[str, float]:
    hist = yf.download(ticker, period="2mo", interval="1d", progress=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}.")

    close_series = None
    for c in ("Close", "Adj Close"):
        if c in hist.columns:
            close_series = hist[c].astype(float)
            break
    if close_series is None or close_series.dropna().empty:
        raise RuntimeError(f"Price history for {ticker} missing Close/Adj Close.")

    close_5d_ma = float(close_series.rolling(window=5, min_periods=1).mean().iloc[-1])
    daily_return_pct = float(close_series.pct_change().fillna(0.0).iloc[-1])
    latest_close = float(close_series.iloc[-1])
    return {
        "latest_close": latest_close,
        "close_5d_ma": close_5d_ma,
        "daily_return_pct": daily_return_pct,
    }


def _fetch_yahoo_rss_news(ticker: str) -> pd.DataFrame:
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    try:
        parsed = feedparser.parse(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse Yahoo RSS for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["title", "published", "summary", "link"])

    if getattr(parsed, "bozo", False):
        logger.warning(
            "Yahoo RSS malformed for %s: %s", ticker, getattr(parsed, "bozo_exception", None)
        )

    entries = getattr(parsed, "entries", []) or []
    records: list[dict[str, str]] = []
    for e in entries:
        records.append(
            {
                "title": str(getattr(e, "title", "") or ""),
                "published": str(
                    getattr(e, "published", "") or getattr(e, "updated", "") or ""
                ),
                "summary": str(
                    getattr(e, "summary", "") or getattr(e, "description", "") or ""
                ),
                "link": str(getattr(e, "link", "") or ""),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    pub = df["published"]
    if pub.dt.tz is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)).replace(
            tzinfo=None
        )
        pub = pub.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        cutoff = datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)
    return df.loc[pub.notna() & (pub >= cutoff)].copy()


def _score_sentiment_and_risk(news_df: pd.DataFrame) -> Dict[str, float]:
    if news_df is None or news_df.empty:
        return {"avg_sentiment": 0.0, "sentiment_5d_ma": 0.0, "avg_geopolitical_risk": 0.0}

    full_text = (news_df["title"].astype(str) + " " + news_df["summary"].astype(str)).astype(str)
    risk_scores = full_text.apply(_geopolitical_risk_score).astype(float)
    avg_risk = float(risk_scores.mean()) if len(risk_scores) else 0.0

    titles = news_df["title"].astype(str).fillna("").tolist()
    try:
        pipe = _get_sentiment_pipeline()
        results = pipe(titles)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentiment inference failed; using neutral. Error: %s", exc)
        return {"avg_sentiment": 0.0, "sentiment_5d_ma": 0.0, "avg_geopolitical_risk": avg_risk}

    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    numeric_scores: List[float] = []
    for r in results:
        label = str(r.get("label", "")).lower()
        conf = float(r.get("score", 0.0))
        numeric_scores.append(label_map.get(label, 0.0) * conf)

    avg_sentiment = float(np.mean(numeric_scores)) if numeric_scores else 0.0

    tmp = news_df.copy()
    tmp["sentiment_score"] = numeric_scores
    tmp["date"] = pd.to_datetime(tmp["published"], errors="coerce").dt.normalize()
    daily = (
        tmp.dropna(subset=["date"])
        .groupby("date", as_index=False)["sentiment_score"]
        .mean()
        .sort_values("date")
    )
    if daily.empty:
        sent_5d = avg_sentiment
    else:
        sent_5d = float(daily["sentiment_score"].rolling(window=5, min_periods=1).mean().iloc[-1])

    return {"avg_sentiment": avg_sentiment, "sentiment_5d_ma": sent_5d, "avg_geopolitical_risk": avg_risk}


def _build_model_input_row(
    base_features: Dict[str, float],
    sector: Optional[str],
    model_obj: Any,
) -> pd.DataFrame:
    """
    Create a 1-row DataFrame aligned to model_obj.feature_names_in_ if present.
    Falls back to FEATURE_COLUMNS ordering otherwise.
    """
    if hasattr(model_obj, "feature_names_in_"):
        cols = list(getattr(model_obj, "feature_names_in_"))
        row = {c: 0.0 for c in cols}
        for k, v in base_features.items():
            if k in row:
                row[k] = float(v)
        if sector:
            for c in (f"Sector_{sector}", f"Sector_{sector.replace(' ', '_')}"):
                if c in row:
                    row[c] = 1.0
        return pd.DataFrame([row], columns=cols).fillna(0.0)
    return pd.DataFrame([[base_features.get(c, 0.0) for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS).fillna(0.0)


@app.get("/")
def read_root() -> Dict[str, str]:
    """
    Health check endpoint for the Global Market Predictor API.

    Returns
    -------
    Dict[str, str]
        A JSON object indicating that the API is running.
    """
    return {"status": "Global Market Predictor API is running"}


@app.post("/predict_trend")
def predict_trend(input_data: MarketDataInput) -> Dict[str, Any]:
    """
    Predict the next day's market trend based on engineered features.

    This endpoint accepts the four key features used by the trained
    Random Forest model, formats them for the model, and returns a
    prediction along with the associated probability.

    Parameters
    ----------
    input_data : MarketDataInput
        The input feature set for a single prediction.

    Returns
    -------
    Dict[str, Any]
        A JSON-compatible dictionary containing:

        - ``prediction``: "Up" or "Down"
        - ``confidence``: float probability score associated with the
          predicted class.

    Raises
    ------
    HTTPException
        If the model is not loaded or if prediction fails.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available. Please train the model first.",
        )

    try:
        data_row = [
            input_data.sentiment_score,
            input_data.sentiment_5d_ma,
            input_data.close_5d_ma,
            input_data.daily_return_pct,
        ]

        # Use a DataFrame to preserve column ordering and compatibility
        # with models trained on named features.
        input_df = pd.DataFrame([data_row], columns=FEATURE_COLUMNS)
        # Fallback: fill any NaN (e.g. from limited news) with neutral so we still return a forecast.
        input_df = input_df.fillna(0.0)

        # Ensure the model supports probability estimates.
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Loaded model does not support probability predictions.")

        probabilities = model.predict_proba(input_df)
        preds = model.predict(input_df)

        predicted_class = int(preds[0])
        # Assuming binary classification with classes [0, 1]
        prob_up = float(probabilities[0][1])
        prob_down = float(probabilities[0][0])

        if predicted_class == 1:
            prediction_label = "Up"
            confidence = prob_up
        else:
            prediction_label = "Down"
            confidence = prob_down

        result: Dict[str, Any] = {
            "prediction": prediction_label,
            "confidence": confidence,
        }

        if return_regressor is not None:
            try:
                pred_return = return_regressor.predict(input_df)
                result["predicted_return_pct"] = round(float(np.asarray(pred_return).flat[0]), 2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Regressor prediction failed: %s", exc)

        return result
    except HTTPException:
        # Re-raise HTTPExceptions unchanged.
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the prediction.",
        ) from exc


@app.get("/predict_ticker/{ticker}")
def predict_ticker(ticker: str) -> Dict[str, Any]:
    """
    Live prediction for a ticker using the trained Random Forest (trained_ai_model.pkl).

    - Fetches latest daily data for the ticker plus macro proxies (^VIX, ^TNX).
    - Computes technical indicators: rsi, sma_20, macd.
    - Builds a single-row DataFrame in the model's feature order and runs predict_proba.
    - Returns: ticker, features, prediction (Bullish/Bearish), confidence (Probability %).
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available. Please train the model first.",
        )

    symbol = (ticker or "").strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Ticker is required.")

    try:
        # Use RSI/SMA20/MACD/VIX/TNX features if the loaded model expects them (trained_ai_model.pkl)
        feature_names = getattr(model, "feature_names_in_", None) or []
        use_trained_model_features = set(feature_names) >= {"rsi", "sma_20", "macd", "vix", "tnx"}

        if use_trained_model_features:
            price_feats = _fetch_trained_model_features(symbol)
            base_features = {
                "rsi": price_feats["rsi"],
                "sma_20": price_feats["sma_20"],
                "macd": price_feats["macd"],
                "vix": price_feats["vix"],
                "tnx": price_feats["tnx"],
            }
            latest_close = price_feats["latest_close"]
            daily_return_pct = price_feats["daily_return_pct"]
            news_df = _fetch_yahoo_rss_news(symbol)
            scored = _score_sentiment_and_risk(news_df)
        else:
            price_feats = _fetch_recent_prices(symbol)
            news_df = _fetch_yahoo_rss_news(symbol)
            scored = _score_sentiment_and_risk(news_df)
            latest_close = float(price_feats["latest_close"])
            daily_return_pct = float(price_feats["daily_return_pct"])
            base_features = {
                "sentiment_score": float(scored["avg_sentiment"]),
                "sentiment_5d_ma": float(scored["sentiment_5d_ma"]),
                "close_5d_ma": float(price_feats["close_5d_ma"]),
                "daily_return_pct": daily_return_pct,
                "geopolitical_risk_score": float(scored["avg_geopolitical_risk"]),
            }

        sector: Optional[str] = None
        try:
            sector = yf.Ticker(symbol).info.get("sector")
        except Exception:  # noqa: BLE001
            sector = None

        clf_input = _build_model_input_row(base_features, sector=sector, model_obj=model)
        # Ensure single-row DataFrame has features in the exact order the model expects
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            clf_input = clf_input[cols] if all(c in clf_input.columns for c in cols) else clf_input

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Loaded model does not support probability predictions.")
        probabilities = model.predict_proba(clf_input)
        preds = model.predict(clf_input)

        predicted_class = int(preds[0])
        prob_up = float(probabilities[0][1])
        prob_down = float(probabilities[0][0])
        if predicted_class == 1:
            prediction_label = "Bullish"
            confidence_pct = prob_up * 100.0
        else:
            prediction_label = "Bearish"
            confidence_pct = prob_down * 100.0

        if use_trained_model_features:
            result = {
                "ticker": symbol,
                "features": {
                    "rsi": round(base_features["rsi"], 4),
                    "sma_20": round(base_features["sma_20"], 4),
                    "macd": round(base_features["macd"], 4),
                    "vix": round(base_features["vix"], 4),
                    "tnx": round(base_features["tnx"], 4),
                },
                "prediction": prediction_label,
                "confidence": prob_up if predicted_class == 1 else prob_down,
                "confidence_pct": round(confidence_pct, 2),
            }
        else:
            result = {
                "ticker": symbol,
                "sector": sector,
                "latest_close": latest_close,
                "daily_return_pct": daily_return_pct,
                "avg_sentiment": float(scored["avg_sentiment"]),
                "sentiment_5d_ma": float(scored["sentiment_5d_ma"]),
                "geopolitical_risk_score": float(scored["avg_geopolitical_risk"]),
                "prediction": prediction_label,
                "confidence": prob_up if predicted_class == 1 else prob_down,
                "news_items_used": int(len(news_df)) if news_df is not None else 0,
            }

        if return_regressor is not None and not use_trained_model_features:
            try:
                reg_input = _build_model_input_row(
                    base_features,
                    sector=sector,
                    model_obj=return_regressor,
                )
                pred_return = return_regressor.predict(reg_input)
                result["predicted_return_pct"] = round(float(np.asarray(pred_return).flat[0]), 2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Regressor prediction failed for %s: %s", symbol, exc)

        return result
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict_ticker failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail="Failed to generate ticker forecast.") from exc


if __name__ == "__main__":
    import uvicorn

    _configure_logging()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

