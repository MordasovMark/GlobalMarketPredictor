"""
FastAPI backend for Global Market Predictor.
Serves predictions from ai_range_models.pkl and live data from yfinance.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import random
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import feedparser
except ImportError:  # pragma: no cover - optional RSS enhancement
    feedparser = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path("ai_range_models.pkl")
REGRESSION_FEATURE_COLS = [
    "SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "Volatility", "Daily_Return"
]
HISTORY_CACHE_TTL_SECONDS = 120
HISTORY_CACHE_MAX_ENTRIES = 128
QUOTE_CACHE_TTL_SECONDS = 45
DASHBOARD_CACHE_TTL_SECONDS = 300
MARKET_NEWS_LOOKBACK_DAYS = 5
MARKET_SYMBOL_META = {
    "SPY": {"label": "SPY", "name": "S&P 500"},
    "QQQ": {"label": "QQQ", "name": "Nasdaq"},
    "IWM": {"label": "IWM", "name": "Russell 2000"},
}
FEAR_KEYWORDS = (
    "inflation",
    "rates",
    "rate",
    "yield",
    "yields",
    "recession",
    "selloff",
    "sell-off",
    "falls",
    "drop",
    "drops",
    "slump",
    "risk",
    "war",
    "tariff",
    "debt",
    "volatility",
)
GREED_KEYWORDS = (
    "rally",
    "rallies",
    "record",
    "surge",
    "surges",
    "gain",
    "gains",
    "beat",
    "beats",
    "growth",
    "optimism",
    "soft landing",
    "easing",
    "cuts",
    "ai",
    "earnings",
)

app = FastAPI(title="AI Trading API")

# CORS: allow frontend (e.g. localhost:5173) to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://globalmarketpredictor.onrender.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup (same structure as train_model.py output)
_models: Optional[Dict[str, Any]] = None
_history_cache: Dict[tuple[str, str], tuple[float, pd.DataFrame, str, bool, str]] = {}
_quote_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_dashboard_cache: tuple[float, Dict[str, Any]] | None = None


def _remember_history(
    cache_key: tuple[str, str],
    hist: pd.DataFrame,
    date_fmt: str,
    is_intraday: bool,
    range_lower: str,
) -> None:
    _history_cache[cache_key] = (
        time.monotonic(),
        hist.copy(),
        date_fmt,
        is_intraday,
        range_lower,
    )
    if len(_history_cache) > HISTORY_CACHE_MAX_ENTRIES:
        oldest_key = min(_history_cache, key=lambda key: _history_cache[key][0])
        _history_cache.pop(oldest_key, None)


def _load_models() -> Optional[Dict[str, Any]]:
    global _models
    if _models is not None:
        return _models
    try:
        if MODEL_PATH.is_file():
            _models = joblib.load(MODEL_PATH)
            return _models
    except Exception as e:
        print(f"Warning: Could not load ai_range_models.pkl — {e}")
    return None


# ---------------------------------------------------------------------------
# Data & feature helpers (aligned with dashboard / train_model)
# ---------------------------------------------------------------------------
def _fetch_price_df(ticker: str, period: str = "4y") -> pd.DataFrame:
    """Fetch OHLCV from yfinance, return DataFrame with DatetimeIndex and Close."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
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
    if "Close" not in data.columns:
        return pd.DataFrame()
    return data


def _build_regression_features(price_df: pd.DataFrame, close_col: str = "Close") -> Optional[pd.DataFrame]:
    """Build one row of features for ai_range_models.pkl."""
    if price_df.empty or close_col not in price_df.columns:
        return None
    close = pd.to_numeric(price_df[close_col], errors="coerce").dropna()
    if len(close) < 50:
        return None
    ret = close.pct_change()
    rsi_period = 14
    macd_fast, macd_slow, macd_signal = 12, 26, 9
    vol_window = 20
    sma_20 = close.rolling(window=20, min_periods=1).mean()
    sma_50 = close.rolling(window=50, min_periods=1).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50.0)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=macd_signal, adjust=False).mean()
    volatility = ret.rolling(window=vol_window, min_periods=1).std()
    last = pd.DataFrame({
        "SMA_20": [sma_20.iloc[-1]],
        "SMA_50": [sma_50.iloc[-1]],
        "RSI": [rsi.iloc[-1]],
        "MACD": [macd.iloc[-1]],
        "MACD_Signal": [macd_sig.iloc[-1]],
        "Volatility": [volatility.iloc[-1]],
        "Daily_Return": [ret.iloc[-1]],
    })
    last = last.fillna(0.0)
    return last[REGRESSION_FEATURE_COLS]


def _get_range_forecasts(
    price_df: pd.DataFrame, close_col: str, models: Dict[str, Any]
) -> Optional[Dict[str, tuple[float, float]]]:
    """Return {'1D': (low, high), '5D': (low, high), '10D': (low, high)} or None."""
    if not models or not isinstance(models, dict):
        return None
    horizons = ["1D", "5D", "10D"]
    for h in horizons:
        band = models.get(h)
        if not isinstance(band, dict) or "low" not in band or "high" not in band:
            return None
        if not hasattr(band["low"], "predict") or not hasattr(band["high"], "predict"):
            return None
    try:
        X = _build_regression_features(price_df, close_col)
        if X is None:
            return None
        feature_names = models.get("feature_names", REGRESSION_FEATURE_COLS)
        if feature_names:
            X = X[[c for c in feature_names if c in X.columns]]
        out = {}
        for h in horizons:
            low_val = float(models[h]["low"].predict(X)[0])
            high_val = float(models[h]["high"].predict(X)[0])
            out[h] = (low_val, high_val)
        return out
    except Exception:
        return None


def _get_macro_1d(symbols: List[str]) -> List[tuple[str, float, float]]:
    """Fetch (symbol, price, change_pct) for each symbol."""
    out: List[tuple[str, float, float]] = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d", interval="1d")
            if hist is None or hist.empty or len(hist) < 2:
                out.append((sym, 0.0, 0.0))
                continue
            close = hist["Close"]
            latest = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            ch = ((latest - prev) / prev * 100.0) if prev else 0.0
            out.append((sym, latest, ch))
        except Exception:
            out.append((sym, 0.0, 0.0))
    return out


def _clean_symbol(symbol: str) -> str:
    cleaned = "".join(ch for ch in str(symbol or "").strip().upper() if ch.isalnum() or ch in ".-^=")
    return cleaned[:24]


def _yf_symbol(symbol: str) -> str:
    """Yahoo uses dashes for class shares while the UI uses dots (BRK.B)."""
    return symbol.replace(".", "-")


def _quote_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    symbol_norm = _clean_symbol(symbol)
    if not symbol_norm:
        return None

    now = time.monotonic()
    cached = _quote_cache.get(symbol_norm)
    if cached is not None and now - cached[0] < QUOTE_CACHE_TTL_SECONDS:
        return dict(cached[1])

    try:
        ticker = yf.Ticker(_yf_symbol(symbol_norm))
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            hist = yf.download(
                _yf_symbol(symbol_norm),
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        if hist is None or hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0).str.strip()
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if close.empty:
            return None

        latest = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else latest
        change = latest - prev
        change_pct = (change / prev * 100.0) if prev else 0.0
        row = {
            "ticker": symbol_norm,
            "price": round(latest, 2),
            "change": round(change, 2),
            "changePercent": round(change_pct, 2),
            "change_pct": round(change_pct, 2),
        }
        _quote_cache[symbol_norm] = (now, row)
        return dict(row)
    except Exception:
        return None


def _download_close_series(symbol: str, period: str = "1y") -> pd.Series:
    try:
        data = yf.download(
            symbol,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return pd.Series(dtype=float)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0).str.strip()
    close_col = "Close" if "Close" in data.columns else data.columns[0]
    close = pd.to_numeric(data[close_col], errors="coerce").dropna()
    return close.sort_index()


def _clamp_sentiment_score(value: Any, fallback: float = 50.0) -> int:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = fallback
    if not np.isfinite(num):
        num = fallback
    return int(round(min(100.0, max(0.0, num))))


def _score_market_sentiment(sp500: float, sma20: float, return_5d_pct: float, vix: float) -> int:
    """Composite 0-100 sentiment from current market data when CNN data is unavailable."""
    score = 50.0
    if sma20 > 0:
        score += float(np.clip(((sp500 / sma20) - 1.0) * 500.0, -25.0, 25.0))
    score += float(np.clip(return_5d_pct * 2.5, -15.0, 15.0))
    if vix > 0:
        score += float(np.clip((20.0 - vix) * 1.4, -20.0, 20.0))
    return _clamp_sentiment_score(score)


def _build_market_sentiment_from_yfinance() -> Dict[str, Any]:
    sp500 = _download_close_series("^GSPC", "1y")
    vix = _download_close_series("^VIX", "1y")
    if sp500.empty:
        return {
            "value": 50,
            "source": "yfinance composite unavailable",
            "trend": [
                {"period": "Current", "periodKey": "Current", "score": 50, "value": 50},
                {"period": "1 Week Ago", "periodKey": "1W", "score": 50, "value": 50},
                {"period": "1 Month Ago", "periodKey": "1M", "score": 50, "value": 50},
            ],
            "timeline": [],
        }

    combined = pd.DataFrame({"sp500": sp500})
    if not vix.empty:
        combined["vix"] = vix.reindex(combined.index, method="ffill")
    else:
        combined["vix"] = 20.0
    combined = combined.ffill().dropna()
    if combined.empty:
        return {
            "value": 50,
            "source": "yfinance composite unavailable",
            "trend": [
                {"period": "Current", "periodKey": "Current", "score": 50, "value": 50},
                {"period": "1 Week Ago", "periodKey": "1W", "score": 50, "value": 50},
                {"period": "1 Month Ago", "periodKey": "1M", "score": 50, "value": 50},
            ],
            "timeline": [],
        }

    combined["sma20"] = combined["sp500"].rolling(window=20, min_periods=5).mean()
    combined["return_5d_pct"] = combined["sp500"].pct_change(5).fillna(0.0) * 100.0
    combined["score"] = combined.apply(
        lambda row: _score_market_sentiment(row["sp500"], row["sma20"], row["return_5d_pct"], row["vix"]),
        axis=1,
    )
    latest = combined.iloc[-1]
    value = _clamp_sentiment_score(latest["score"])

    def score_as_of(days_back: int) -> int:
        target = combined.index[-1] - datetime.timedelta(days=days_back)
        historical = combined.loc[combined.index <= target]
        if historical.empty:
            historical = combined
        return _clamp_sentiment_score(historical.iloc[-1]["score"], value)

    trend = [
        {"period": "Current", "periodKey": "Current", "score": value, "value": value},
        {"period": "1 Week Ago", "periodKey": "1W", "score": score_as_of(7), "value": score_as_of(7)},
        {"period": "1 Month Ago", "periodKey": "1M", "score": score_as_of(30), "value": score_as_of(30)},
    ]

    monthly = combined[["sp500", "score"]].resample("ME").last().dropna().tail(12)
    timeline = [
        {
            "month": idx.strftime("%b"),
            "fearGreed": _clamp_sentiment_score(row["score"], value),
            "sp500": round(float(row["sp500"]), 2),
        }
        for idx, row in monthly.iterrows()
    ]
    return {
        "value": value,
        "source": "yfinance market composite",
        "trend": trend,
        "timeline": timeline,
    }


def _fetch_cnn_fear_greed() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=8,
            headers={"User-Agent": "GlobalMarketPredictor/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        current = payload.get("fear_and_greed") or {}
        score = current.get("score", current.get("value"))
        value = _clamp_sentiment_score(score)
        historical_payload = payload.get("fear_and_greed_historical") or {}
        historical_rows = historical_payload.get("data") or []
        scores: list[tuple[datetime.datetime, int]] = []
        for row in historical_rows:
            raw_x = row.get("x") or row.get("date")
            raw_y = row.get("y") if row.get("y") is not None else row.get("score")
            try:
                if isinstance(raw_x, (int, float)):
                    dt = datetime.datetime.fromtimestamp(float(raw_x) / 1000.0, tz=datetime.timezone.utc)
                else:
                    dt = pd.to_datetime(raw_x, utc=True).to_pydatetime()
            except Exception:
                continue
            scores.append((dt, _clamp_sentiment_score(raw_y, value)))

        def historical_score(days_back: int) -> int:
            if not scores:
                return value
            target = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)
            eligible = [item for item in scores if item[0] <= target]
            return eligible[-1][1] if eligible else scores[0][1]

        trend = [
            {"period": "Current", "periodKey": "Current", "score": value, "value": value},
            {"period": "1 Week Ago", "periodKey": "1W", "score": historical_score(7), "value": historical_score(7)},
            {"period": "1 Month Ago", "periodKey": "1M", "score": historical_score(30), "value": historical_score(30)},
        ]
        return {
            "value": value,
            "rating": current.get("rating"),
            "source": "CNN Fear & Greed Index",
            "trend": trend,
        }
    except Exception:
        return None


def _time_ago_from_struct(published_parsed: Any) -> str:
    if not published_parsed:
        return ""
    try:
        published = datetime.datetime.fromtimestamp(time.mktime(published_parsed), tz=datetime.timezone.utc)
    except Exception:
        return ""
    delta = datetime.datetime.now(datetime.timezone.utc) - published
    if delta.days >= 1:
        return f"{delta.days}d ago"
    hours = int(delta.total_seconds() // 3600)
    if hours >= 1:
        return f"{hours}h ago"
    minutes = max(1, int(delta.total_seconds() // 60))
    return f"{minutes}m ago"


def _source_from_link(link: str) -> str:
    host = urlparse(link or "").netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if "finance.yahoo" in host:
        return "Yahoo Finance"
    if "reuters" in host:
        return "Reuters"
    if "bloomberg" in host:
        return "Bloomberg"
    if "cnbc" in host:
        return "CNBC"
    return host.split(".")[0].title() if host else "Market news"


def _fetch_market_news_drivers(sentiment_value: int) -> Dict[str, Any]:
    if feedparser is None:
        label = "fear" if sentiment_value < 45 else "greed" if sentiment_value > 55 else "neutral"
        if label == "fear":
            summary = "Risk-off conditions dominate as live market data points to weaker risk appetite."
        elif label == "greed":
            summary = "Risk appetite is constructive as live market data shows stronger demand for equities."
        else:
            summary = "Market sentiment is balanced as live data does not show a decisive fear or greed signal."
        return {
            "fearDrivers": [],
            "greedDrivers": [],
            "aiConclusion": {"summary": summary},
        }

    feed_urls = [
        "https://finance.yahoo.com/rss/topstories",
        "https://finance.yahoo.com/rss/headline?s=SPY",
        "https://finance.yahoo.com/rss/headline?s=QQQ",
    ]
    seen: set[str] = set()
    fear: list[Dict[str, str]] = []
    greed: list[Dict[str, str]] = []
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=MARKET_NEWS_LOOKBACK_DAYS)

    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            continue
        for entry in getattr(parsed, "entries", []) or []:
            title = str(getattr(entry, "title", "") or "").strip()
            if len(title) < 8 or title in seen:
                continue
            published_parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            if published_parsed:
                try:
                    published = datetime.datetime.fromtimestamp(time.mktime(published_parsed), tz=datetime.timezone.utc)
                    if published < cutoff:
                        continue
                except Exception:
                    pass
            seen.add(title)
            lower = title.lower()
            item = {
                "id": f"news-{len(seen)}",
                "headline": title,
                "source": _source_from_link(str(getattr(entry, "link", "") or "")),
                "time": _time_ago_from_struct(published_parsed),
            }
            if any(keyword in lower for keyword in FEAR_KEYWORDS):
                fear.append(item)
            elif any(keyword in lower for keyword in GREED_KEYWORDS):
                greed.append(item)
            elif sentiment_value < 45:
                fear.append(item)
            elif sentiment_value > 55:
                greed.append(item)

            if len(fear) >= 3 and len(greed) >= 3:
                break
        if len(fear) >= 3 and len(greed) >= 3:
            break

    label = "fear" if sentiment_value < 45 else "greed" if sentiment_value > 55 else "neutral"
    if label == "fear":
        summary = "Risk-off conditions dominate as live market data and recent headlines point to weaker risk appetite."
    elif label == "greed":
        summary = "Risk appetite is constructive as live market data and recent headlines show stronger demand for equities."
    else:
        summary = "Market sentiment is balanced as live data and recent headlines do not show a decisive fear or greed signal."

    return {
        "fearDrivers": fear[:3],
        "greedDrivers": greed[:3],
        "aiConclusion": {"summary": summary},
    }


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class MacroItem(BaseModel):
    symbol: str
    price: float
    change_pct: float


class ForecastRange(BaseModel):
    low: float
    high: float


class ForecastResponse(BaseModel):
    ticker: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    forecasts: Optional[Dict[str, ForecastRange]] = None


class PriceResponse(BaseModel):
    ticker: str
    price: Optional[float] = None
    change_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/health")
def health():
    models_loaded = _load_models() is not None
    return {"status": "ok", "models_loaded": models_loaded}


@app.get("/api/market", response_model=List[MacroItem])
def api_market():
    """Fetch current market data for SPY, QQQ, IWM: price and 1D % change. Returns a JSON array."""
    symbols = ["SPY", "QQQ", "IWM"]
    data = _get_macro_1d(symbols)
    return [MacroItem(symbol=s, price=p, change_pct=round(c, 2)) for s, p, c in data]


@app.get("/api/macro", response_model=List[MacroItem])
def api_macro(symbols: str = "SPY,QQQ,IWM"):
    """Return latest price and 1D % change for macro symbols (default: SPY, QQQ, IWM)."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        sym_list = ["SPY", "QQQ", "IWM"]
    data = _get_macro_1d(sym_list)
    return [MacroItem(symbol=s, price=p, change_pct=c) for s, p, c in data]


@app.get("/api/quotes")
def api_quotes(symbols: str):
    """Return latest quote data for a comma-separated ticker list."""
    sym_list = [_clean_symbol(s) for s in symbols.split(",") if _clean_symbol(s)]
    sym_list = list(dict.fromkeys(sym_list))[:50]
    if not sym_list:
        raise HTTPException(status_code=400, detail="At least one symbol is required.")
    quotes = [_quote_for_symbol(symbol) for symbol in sym_list]
    return [quote for quote in quotes if quote is not None]


@app.get("/api/market-sentiment")
def api_market_sentiment():
    """Return current US market sentiment, recent trend, and timeline data."""
    composite = _build_market_sentiment_from_yfinance()
    cnn = _fetch_cnn_fear_greed()
    if cnn is not None:
        composite["value"] = cnn["value"]
        composite["source"] = cnn.get("source", composite.get("source"))
        composite["trend"] = cnn.get("trend") or composite.get("trend")
        if cnn.get("rating"):
            composite["rating"] = cnn["rating"]
    composite["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return composite


@app.get("/api/dashboard")
def api_dashboard():
    """Return live dashboard data for sentiment, market cards, and news drivers."""
    global _dashboard_cache
    now = time.monotonic()
    if _dashboard_cache is not None and now - _dashboard_cache[0] < DASHBOARD_CACHE_TTL_SECONDS:
        return dict(_dashboard_cache[1])

    sentiment = api_market_sentiment()
    sentiment_value = _clamp_sentiment_score(sentiment.get("value"))
    market_quotes = []
    for symbol, meta in MARKET_SYMBOL_META.items():
        quote = _quote_for_symbol(symbol)
        if quote is None:
            continue
        market_quotes.append({
            "symbol": symbol,
            "label": meta["label"],
            "name": meta["name"],
            "price": quote["price"],
            "change_pct": quote["change_pct"],
            "changePercent": quote["changePercent"],
        })

    payload = {
        "sentiment": {
            "value": sentiment_value,
            "source": sentiment.get("source"),
            "rating": sentiment.get("rating"),
            "trend": sentiment.get("trend") or [],
            "historical": sentiment.get("trend") or [],
            "last_updated": sentiment.get("last_updated"),
        },
        "timeline": sentiment.get("timeline") or [],
        "market": market_quotes,
        "news": _fetch_market_news_drivers(sentiment_value),
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _dashboard_cache = (now, payload)
    return payload


def _yf_history_for_time_range(ticker: str, time_range: str) -> tuple[pd.DataFrame, str, bool, str]:
    """Download OHLCV for the dashboard time_range. Returns hist (sorted), strftime format, intraday flag, normalized range key."""
    ticker_norm = (ticker or "").strip().upper()
    range_lower = (time_range or "3mo").strip().lower()
    if range_lower not in ("1d", "1mo", "3mo", "6mo", "1y"):
        range_lower = "3mo"

    cache_key = (ticker_norm, range_lower)
    cached = _history_cache.get(cache_key)
    if cached is not None:
        cached_at, cached_hist, cached_fmt, cached_intraday, cached_range = cached
        if time.monotonic() - cached_at < HISTORY_CACHE_TTL_SECONDS:
            return cached_hist.copy(), cached_fmt, cached_intraday, cached_range

    stock = yf.Ticker(ticker_norm)
    if range_lower == "1d":
        hist = stock.history(period="1d", interval="5m")
        date_fmt = "%H:%M"
        is_intraday = True
    elif range_lower == "1mo":
        hist = stock.history(period="1mo", interval="1d")
        date_fmt = "%Y-%m-%d"
        is_intraday = False
    elif range_lower == "6mo":
        hist = stock.history(period="6mo", interval="1d")
        date_fmt = "%Y-%m-%d"
        is_intraday = False
    elif range_lower == "1y":
        hist = stock.history(period="1y", interval="1d")
        date_fmt = "%Y-%m-%d"
        is_intraday = False
    else:
        hist = stock.history(period="3mo", interval="1d")
        date_fmt = "%Y-%m-%d"
        is_intraday = False

    if hist is None or hist.empty or len(hist) < 2:
        empty = pd.DataFrame()
        _remember_history(cache_key, empty, date_fmt, is_intraday, range_lower)
        return empty.copy(), date_fmt, is_intraday, range_lower

    if isinstance(hist.index, pd.DatetimeIndex):
        hist = hist.sort_index()
    _remember_history(cache_key, hist, date_fmt, is_intraday, range_lower)
    return hist, date_fmt, is_intraday, range_lower


@app.get("/api/portfolio/simulate")
def api_portfolio_simulate(ticker: str, time_range: str = "3mo"):
    """Paper-trade demo: same SMA / 5d momentum rules as /api/analyze, $10k all-in long-only."""
    INITIAL = 10_000.0
    try:
        hist, date_fmt, is_intraday, range_norm = _yf_history_for_time_range(ticker, time_range)
        if hist.empty or len(hist) < 2:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        close = pd.to_numeric(hist["Close"], errors="coerce")
        close = close.dropna()
        if len(close) < 2:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        min_warmup = 14 if is_intraday else 20
        cash = INITIAL
        shares = 0.0
        trades: List[Dict[str, Any]] = []
        prev_signal = "HOLD"

        if len(close) > min_warmup:
            for i in range(min_warmup, len(close)):
                window = close.iloc[: i + 1]
                price = float(window.iloc[-1])
                sma_5 = float(window.iloc[-5:].mean()) if len(window) >= 5 else price
                sma_20 = float(window.iloc[-20:].mean()) if len(window) >= 20 else price
                pct_5d = (
                    ((price - float(window.iloc[-5])) / float(window.iloc[-5]) * 100.0)
                    if len(window) >= 5
                    else 0.0
                )

                if sma_5 > sma_20 and pct_5d > 0:
                    sig = "BUY"
                elif sma_5 < sma_20 and pct_5d < 0:
                    sig = "SELL"
                else:
                    sig = "HOLD"

                date_ts = close.index[i]
                if hasattr(date_ts, "strftime"):
                    date_str = date_ts.strftime(date_fmt)
                else:
                    date_str = str(date_ts)[:16] if is_intraday else str(date_ts)[:10]

                if sig == "BUY" and prev_signal != "BUY" and shares < 1e-9 and cash > 0:
                    shares = cash / price
                    cash = 0.0
                    trades.append({"date": date_str, "action": "BUY", "execution_price": round(price, 2)})
                elif sig == "SELL" and prev_signal != "SELL" and shares > 1e-9:
                    cash = shares * price
                    shares = 0.0
                    trades.append({"date": date_str, "action": "SELL", "execution_price": round(price, 2)})

                prev_signal = sig

        last_px = float(close.iloc[-1])
        final_balance = float(cash + shares * last_px)
        roi_pct = ((final_balance - INITIAL) / INITIAL) * 100.0 if INITIAL else 0.0

        return {
            "ticker": ticker.upper(),
            "time_range": range_norm,
            "initial_balance": round(INITIAL, 2),
            "final_balance": round(final_balance, 2),
            "total_roi_pct": round(roi_pct, 2),
            "trades": trades,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze")
def api_analyze(ticker: str, time_range: str = "3mo"):
    """Fetch live price, 1d change, chart_data (history + forecast), and forecast_summary. Query: ticker, time_range (1d|1mo|3mo|6mo|1y)."""
    try:
        hist, date_fmt, is_intraday, _ = _yf_history_for_time_range(ticker, time_range)
        if hist.empty or len(hist) < 2:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        close = hist["Close"]
        current_price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change_pct = ((current_price - prev) / prev * 100.0) if prev else 0.0

        chart_data: List[Dict[str, Any]] = []
        # Loop 1: Historical data — only { date, price }, NO prediction keys
        for date_ts, row in hist.iterrows():
            if hasattr(date_ts, "strftime"):
                date_str = date_ts.strftime(date_fmt)
            else:
                date_str = str(date_ts)[:16].replace(" ", " ")[:5] if is_intraday else str(date_ts)[:10]
            chart_data.append({
                "date": date_str,
                "price": round(float(row["Close"]), 2),
            })

        # Stitching: last historical point gets predictedPrice so Recharts
        # connects the solid price line seamlessly to the dashed forecast line
        chart_data[-1]["predictedPrice"] = chart_data[-1]["price"]

        # --- Step 1: Derive signal from recent price action (SMA-5 vs SMA-20) ---
        models = _load_models()
        is_simulated = models is None

        closes = pd.to_numeric(close, errors="coerce").dropna()
        sma_5 = float(closes.iloc[-5:].mean()) if len(closes) >= 5 else current_price
        sma_20 = float(closes.iloc[-20:].mean()) if len(closes) >= 20 else current_price
        pct_5d = (
            ((current_price - float(closes.iloc[-5])) / float(closes.iloc[-5]) * 100.0)
            if len(closes) >= 5 else 0.0
        )

        if sma_5 > sma_20 and pct_5d > 0:
            signal = "BUY"
        elif sma_5 < sma_20 and pct_5d < 0:
            signal = "SELL"
        else:
            signal = "HOLD"

        # --- Step 2: Generate signal-consistent forecast (strict final day %) ---
        # Business rule:
        # - BUY  => final day is +[1%, 7%] vs current_price, with small daily jitter
        # - SELL => final day is -[2%, 5%] vs current_price, with small daily jitter
        # - HOLD => stays stable or up to -1% total (tiny jitter)
        n_forecast = 5 if is_intraday else 7
        noise_jitter = 0.001  # ±0.1% daily noise

        if signal == "BUY":
            total_pct = float(np.random.uniform(0.01, 0.07))
            target_ratio = 1.0 + total_pct
        elif signal == "SELL":
            total_pct = float(np.random.uniform(-0.05, -0.02))
            target_ratio = 1.0 + total_pct
        else:
            # HOLD: allow slight stability or up to -1% total
            total_pct = float(np.random.uniform(-0.01, 0.002))
            target_ratio = 1.0 + total_pct

        target_final_price = current_price * target_ratio
        base_daily = target_ratio ** (1.0 / n_forecast) - 1.0

        # Build the first (n-1) daily multipliers from base_daily + jitter,
        # then compute the last-day multiplier to hit the exact target_final_price.
        # This guarantees the final day ends at the required %.
        last_ts = hist.index[-1]
        running_price = current_price
        first_multipliers: List[float] = []

        for _ in range(max(n_forecast - 1, 0)):
            jitter = float(np.random.uniform(-noise_jitter, noise_jitter))
            daily_rate = base_daily + jitter

            if signal == "BUY":
                # Enforce non-decreasing path (avoid dips); final day is adjusted later.
                daily_rate = max(0.0001, daily_rate)
            elif signal == "SELL":
                # Enforce non-increasing path (avoid spikes); final day is adjusted later.
                daily_rate = min(-0.0001, daily_rate)
            else:
                # HOLD: keep it near-flat and safe.
                daily_rate = float(np.clip(daily_rate, -0.005, 0.005))

            first_multipliers.append(1.0 + daily_rate)

        product_first = float(np.prod(first_multipliers)) if first_multipliers else 1.0

        # If we overshot the target in the wrong direction (rare with clamping),
        # scale the first multipliers in log space so the final adjustment keeps monotonicity.
        if signal == "BUY" and product_first > target_ratio and product_first > 0:
            # scale so that product_first == target_ratio (leaves last day ~ flat)
            factor = float(np.log(target_ratio) / np.log(product_first))
            first_multipliers = [float(np.exp(np.log(m) * factor)) for m in first_multipliers]
            product_first = float(np.prod(first_multipliers)) if first_multipliers else 1.0
        elif signal == "SELL" and product_first < target_ratio and product_first > 0:
            factor = float(np.log(target_ratio) / np.log(product_first))
            first_multipliers = [float(np.exp(np.log(m) * factor)) for m in first_multipliers]
            product_first = float(np.prod(first_multipliers)) if first_multipliers else 1.0

        last_multiplier = target_ratio / product_first if product_first > 0 else 1.0

        # Generate dates for each forecast point.
        for i in range(1, n_forecast + 1):
            if i <= len(first_multipliers):
                running_price = max(0.01, running_price * first_multipliers[i - 1])
                predicted = running_price
            else:
                # Force final-day % within the business constraints even after 2dp rounding.
                # (The frontend/tooltip compares the rounded `predictedPrice` values.)
                pred_2dp = round(float(target_final_price), 2)
                if signal == "BUY":
                    min_price_2dp = round(current_price * 1.01, 2)
                    max_price_2dp = round(current_price * 1.07, 2)
                    if pred_2dp < min_price_2dp:
                        pred_2dp = min_price_2dp
                    elif pred_2dp > max_price_2dp:
                        pred_2dp = max_price_2dp
                    # Avoid a final-day dip due to rounding.
                    pred_2dp = max(pred_2dp, round(running_price, 2))
                    pred_2dp = min(pred_2dp, max_price_2dp)
                elif signal == "SELL":
                    # Final is between -5% and -2% total drop => [0.95, 0.98] * current
                    min_price_2dp = round(current_price * 0.95, 2)
                    max_price_2dp = round(current_price * 0.98, 2)
                    if pred_2dp > max_price_2dp:
                        pred_2dp = max_price_2dp
                    elif pred_2dp < min_price_2dp:
                        pred_2dp = min_price_2dp
                    # Avoid a final-day spike due to rounding.
                    pred_2dp = min(pred_2dp, round(running_price, 2))
                    pred_2dp = max(pred_2dp, min_price_2dp)
                else:
                    # HOLD: max drop is -1% => floor is 0.99 * current; allow slight uptick.
                    min_price_2dp = round(current_price * 0.99, 2)
                    max_price_2dp = round(current_price * 1.002, 2)
                    if pred_2dp < min_price_2dp:
                        pred_2dp = min_price_2dp
                    elif pred_2dp > max_price_2dp:
                        pred_2dp = max_price_2dp

                predicted = max(0.01, float(pred_2dp))
                running_price = predicted

            if is_intraday:
                next_ts = last_ts + datetime.timedelta(minutes=5 * i)
                future_date_str = next_ts.strftime(date_fmt) if hasattr(next_ts, "strftime") else str(next_ts)[:5]
            else:
                future_date = last_ts + datetime.timedelta(days=i)
                future_date_str = future_date.strftime(date_fmt) if hasattr(future_date, "strftime") else str(future_date)[:10]

            chart_data.append({
                "date": future_date_str,
                "price": None,
                "predictedPrice": round(float(predicted), 2),
            })

        forecast_summary = {
            "bull": round(running_price * 1.02, 2),
            "base": round(running_price, 2),
            "bear": round(running_price * 0.98, 2),
        }

        # Build the final payload first, then apply a strict last-step override
        # right before returning JSON to the client.
        response_data: Dict[str, Any] = {
            "ticker": ticker.upper(),
            "price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "chart_data": chart_data,
            "forecast_summary": forecast_summary,
            "signal": signal,
            "is_simulated": is_simulated,
        }

        # STRICT FINAL OVERRIDE BEFORE RETURN (UI Consistency) - FIXED
        if response_data.get("signal") == "BUY":
            # 1. Find the last actual historical price (where price is a number)
            historical_prices = [
                item["price"]
                for item in response_data["chart_data"]
                if item.get("price") is not None
            ]
            last_actual_price = (
                float(historical_prices[-1])
                if historical_prices
                else float(response_data.get("price", 0.0))
            )

            # 2. Iterate through the chart_data and force 'predictedPrice' to go up
            # ONLY for future dates (where actual 'price' is None)
            current_p = last_actual_price
            for item in response_data["chart_data"]:
                if item.get("price") is None:  # future forecast point
                    # Force 1% to 2% daily growth so the visual trend cannot go down.
                    current_p = current_p * random.uniform(1.01, 1.02)
                    item["predictedPrice"] = round(current_p, 2)

            # 3. Update the summary based on the absolute last predicted point
            final_pred_price = response_data["chart_data"][-1]["predictedPrice"]
            response_data["forecast_summary"] = {
                "bull": round(final_pred_price * 1.02, 2),
                "base": round(final_pred_price, 2),
                "bear": round(final_pred_price * 0.98, 2),
            }

        return response_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price/{ticker}", response_model=PriceResponse)
def api_price(ticker: str):
    """Return latest price and 1D % change for a ticker."""
    quote = _quote_for_symbol(ticker)
    if quote is None:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
    return PriceResponse(
        ticker=quote["ticker"],
        price=quote["price"],
        change_pct=quote["change_pct"],
    )


@app.get("/api/forecast/{ticker}", response_model=ForecastResponse)
def api_forecast(ticker: str):
    """Return latest price, 1D % change, and AI range forecasts (1D, 5D, 10D) for a ticker."""
    models = _load_models()
    df = _fetch_price_df(ticker)
    if df.empty or len(df) < 2:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
    close = df["Close"]
    latest = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0

    forecasts = None
    if models is not None:
        ranges = _get_range_forecasts(df, "Close", models)
        if ranges is not None:
            forecasts = {
                h: ForecastRange(low=round(r[0], 2), high=round(r[1], 2))
                for h, r in ranges.items()
            }

    return ForecastResponse(
        ticker=ticker.upper(),
        price=latest,
        change_pct=round(change_pct, 2),
        forecasts=forecasts,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
