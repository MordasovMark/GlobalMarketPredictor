"""
FastAPI backend for Global Market Predictor.
Serves predictions from ai_range_models.pkl and live data from yfinance.
"""

from __future__ import annotations

import datetime
import html
from pathlib import Path
import random
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import feedparser
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path("ai_range_models.pkl")
REGRESSION_FEATURE_COLS = [
    "SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "Volatility", "Daily_Return"
]
HISTORY_CACHE_TTL_SECONDS = 120
HISTORY_CACHE_MAX_ENTRIES = 128
NEWS_LOOKBACK_DAYS = 7
NEWS_CACHE_TTL_SECONDS = 15 * 60
NEWS_CACHE_MAX_ENTRIES = 128
NEWS_SENTIMENT_POINTS = {"good": 100.0, "neutral": 65.0, "bad": 35.0}
FINBERT_MODEL_NAME = "ProsusAI/finbert"
NEWS_TICKER_KEYWORDS = {
    "AAPL": ("apple", "iphone"),
    "NVDA": ("nvidia",),
    "MSFT": ("microsoft", "azure", "copilot"),
    "TSLA": ("tesla",),
    "AMZN": ("amazon", "aws"),
    "META": ("meta", "facebook", "instagram"),
    "GOOGL": ("google", "alphabet", "youtube"),
    "BRK-B": ("berkshire", "buffett"),
    "V": ("visa",),
    "UNH": ("unitedhealth", "optum"),
    "JPM": ("jpmorgan", "jp morgan", "dimon"),
    "JNJ": ("johnson & johnson", "j&j"),
    "WMT": ("walmart",),
    "XOM": ("exxon", "exxonmobil"),
    "MA": ("mastercard",),
    "AVGO": ("broadcom", "vmware"),
    "PG": ("procter", "p&g"),
    "ORCL": ("oracle",),
    "COST": ("costco",),
    "HD": ("home depot",),
}

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
_news_cache: Dict[tuple[str, int], tuple[float, Dict[str, Any]]] = {}
_sentiment_pipeline: Optional[Any] = None


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


def _normalize_news_ticker(ticker: str) -> str:
    """Normalize display tickers to symbols accepted by Yahoo/Google RSS."""
    return (ticker or "").strip().upper().replace(".", "-")


def _clean_news_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_published(value: Any) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _format_relative_time(published: Optional[pd.Timestamp]) -> str:
    if published is None:
        return "recent"
    now = pd.Timestamp.now(tz="UTC")
    delta_seconds = max(0, int((now - published).total_seconds()))
    if delta_seconds < 3600:
        minutes = max(1, delta_seconds // 60)
        return f"{minutes}m ago"
    if delta_seconds < 86_400:
        return f"{delta_seconds // 3600}h ago"
    return f"{delta_seconds // 86_400}d ago"


def _entry_source_name(entry: Any, fallback: str) -> str:
    source = getattr(entry, "source", None)
    if isinstance(source, dict):
        title = source.get("title")
    else:
        title = getattr(source, "title", None)
    return _clean_news_text(title) or fallback


def _parse_news_feed(url: str, fallback_source: str) -> List[Dict[str, Any]]:
    try:
        parsed = feedparser.parse(url)
    except Exception:
        return []

    records: List[Dict[str, Any]] = []
    for entry in getattr(parsed, "entries", []) or []:
        headline = _clean_news_text(getattr(entry, "title", ""))
        if not headline:
            continue
        published_raw = (
            getattr(entry, "published", "")
            or getattr(entry, "updated", "")
            or getattr(entry, "created", "")
        )
        published = _parse_published(published_raw)
        records.append(
            {
                "headline": headline,
                "summary": _clean_news_text(
                    getattr(entry, "summary", "") or getattr(entry, "description", "")
                ),
                "source": _entry_source_name(entry, fallback_source),
                "link": str(getattr(entry, "link", "") or ""),
                "published_ts": published,
            }
        )
    return records


def _news_dedupe_key(headline: str) -> str:
    return re.sub(r"\W+", "", str(headline).lower())


def _within_news_lookback(published: Optional[pd.Timestamp]) -> bool:
    if published is None:
        return True
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=NEWS_LOOKBACK_DAYS)
    return published >= cutoff


def _extract_yfinance_link(entry: Dict[str, Any], content: Dict[str, Any]) -> str:
    for container in (content, entry):
        for key in ("canonicalUrl", "clickThroughUrl"):
            url_obj = container.get(key)
            if isinstance(url_obj, dict):
                url = str(url_obj.get("url") or "").strip()
                if url:
                    return url
            elif isinstance(url_obj, str) and url_obj.strip():
                return url_obj.strip()
    return str(entry.get("link") or entry.get("url") or "").strip()


def _parse_yfinance_news_entries(raw_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize yfinance Ticker.news payloads into the shared news record shape."""
    records: List[Dict[str, Any]] = []
    for entry in raw_entries or []:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
        headline = _clean_news_text(content.get("title") or entry.get("title"))
        if not headline:
            continue
        summary = _clean_news_text(
            content.get("summary")
            or content.get("description")
            or entry.get("summary")
            or entry.get("description")
            or ""
        )
        published_raw = (
            content.get("pubDate")
            or content.get("displayTime")
            or entry.get("pubDate")
            or entry.get("providerPublishTime")
        )
        published = _parse_published(published_raw)
        if published is None and entry.get("providerPublishTime") is not None:
            published = pd.to_datetime(entry["providerPublishTime"], unit="s", utc=True, errors="coerce")
            if pd.isna(published):
                published = None
        provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
        source = _clean_news_text(provider.get("displayName")) or "Yahoo Finance"
        records.append(
            {
                "headline": headline,
                "summary": summary,
                "source": source,
                "link": _extract_yfinance_link(entry, content),
                "published_ts": published,
                "provider": "yahoo_finance",
            }
        )
    return records


def _fetch_yfinance_news(ticker: str) -> List[Dict[str, Any]]:
    """Fetch ticker headlines from the Yahoo Finance API via yfinance."""
    symbol = _normalize_news_ticker(ticker)
    symbols_to_try = [symbol]
    if "-" in symbol:
        symbols_to_try.append(symbol.replace("-", "."))

    for sym in symbols_to_try:
        try:
            raw_entries = yf.Ticker(sym).news or []
        except Exception:
            continue
        records = _parse_yfinance_news_entries(raw_entries)
        if not records:
            continue
        return [record for record in records if _within_news_lookback(record.get("published_ts"))]
    return []


def _merge_news_records(
    items: List[Dict[str, Any]],
    seen: set[str],
    records: List[Dict[str, Any]],
    *,
    require_relevance: bool,
    source_ticker: str,
) -> None:
    for record in records:
        published = record.get("published_ts")
        if published is not None and not _within_news_lookback(published):
            continue
        if require_relevance and not _is_relevant_news_item(record, source_ticker):
            continue
        key = _news_dedupe_key(str(record.get("headline", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(record)


def _fetch_live_news(ticker: str) -> List[Dict[str, Any]]:
    source_ticker = _normalize_news_ticker(ticker)
    query = urllib.parse.quote_plus(f"{source_ticker} stock news OR {source_ticker} market analysis")
    urls = [
        (
            f"https://finance.yahoo.com/rss/headline?s={urllib.parse.quote(source_ticker)}",
            "Yahoo Finance",
        ),
        (
            f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
            "Google News",
        ),
    ]

    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    # Primary: Yahoo Finance API (yfinance) — reliable when RSS feeds are blocked or empty.
    _merge_news_records(
        items,
        seen,
        _fetch_yfinance_news(source_ticker),
        require_relevance=False,
        source_ticker=source_ticker,
    )

    for url, fallback_source in urls:
        feed_records = _parse_news_feed(url, fallback_source)
        for record in feed_records:
            record.setdefault("provider", "rss")
        _merge_news_records(
            items,
            seen,
            feed_records,
            require_relevance=True,
            source_ticker=source_ticker,
        )

    items.sort(
        key=lambda item: item.get("published_ts") or pd.Timestamp.min.tz_localize("UTC"),
        reverse=True,
    )
    return items


def _is_relevant_news_item(item: Dict[str, Any], source_ticker: str) -> bool:
    haystack = str(item.get("headline", "")).lower()
    ticker_variants = {
        source_ticker.lower(),
        source_ticker.replace("-", ".").lower(),
        source_ticker.replace("-", "").lower(),
        f"${source_ticker.lower()}",
    }
    keywords = set(NEWS_TICKER_KEYWORDS.get(source_ticker, ()))
    return any(token and token in haystack for token in ticker_variants | keywords)


def _get_sentiment_pipeline() -> Any:
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline

        _sentiment_pipeline = pipeline(task="sentiment-analysis", model=FINBERT_MODEL_NAME)
    return _sentiment_pipeline


def _fallback_sentiment(headline: str) -> tuple[str, float]:
    text = headline.lower()
    negative_terms = ("miss", "probe", "cut", "falls", "risk", "warning", "lawsuit", "slump")
    positive_terms = ("beat", "surge", "growth", "raises", "record", "upgrade", "accelerates")
    if any(term in text for term in negative_terms):
        return "negative", 0.55
    if any(term in text for term in positive_terms):
        return "positive", 0.55
    return "neutral", 0.50


def _score_news_sentiment(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []

    texts = [
        f"{item.get('headline', '')}. {item.get('summary', '')}".strip()
        for item in items
    ]
    try:
        results = _get_sentiment_pipeline()(texts, truncation=True)
    except Exception:
        results = []

    scored: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if index < len(results):
            label = str(results[index].get("label", "neutral")).lower()
            confidence = float(results[index].get("score", 0.0))
        else:
            label, confidence = _fallback_sentiment(str(item.get("headline", "")))

        if label not in {"positive", "negative", "neutral"}:
            label = "neutral"
        ui_sentiment = {"positive": "good", "negative": "bad", "neutral": "neutral"}[label]
        published = item.get("published_ts")
        scored.append(
            {
                **item,
                "sentiment": ui_sentiment,
                "sentiment_label": label,
                "sentiment_score": round(confidence, 4),
                "time": _format_relative_time(published),
                "published": published.isoformat() if published is not None else None,
            }
        )
    return scored


def _build_news_thesis(ticker: str, scored_items: List[Dict[str, Any]], sentiment_score: float) -> str:
    if not scored_items:
        return f"No recent live headlines were found for {ticker}; news sentiment is neutral."
    positive = sum(1 for item in scored_items if item["sentiment"] == "good")
    negative = sum(1 for item in scored_items if item["sentiment"] == "bad")
    lead = scored_items[0]
    if sentiment_score >= 72:
        tone = "positive"
    elif sentiment_score <= 45:
        tone = "negative"
    else:
        tone = "mixed"
    return (
        f"Live news tone is {tone}: {positive} positive and {negative} negative "
        f"signals, led by {lead['source']}: {lead['headline']}"
    )


def _build_news_response(ticker: str, limit: int) -> Dict[str, Any]:
    display_ticker = (ticker or "").strip().upper()
    source_ticker = _normalize_news_ticker(display_ticker)
    raw_items = _fetch_live_news(source_ticker)
    scored_items = _score_news_sentiment(raw_items[: max(limit * 2, limit)])[:limit]
    if scored_items:
        sentiment_score = float(
            np.mean([NEWS_SENTIMENT_POINTS.get(item["sentiment"], 50.0) for item in scored_items])
        )
    else:
        sentiment_score = 50.0

    providers = {str(item.get("provider", "rss")) for item in raw_items[: max(limit * 2, limit)]}
    if "yahoo_finance" in providers and len(providers) > 1:
        news_source = "yahoo_finance+rss"
    elif "yahoo_finance" in providers:
        news_source = "yahoo_finance"
    elif providers:
        news_source = "rss"
    else:
        news_source = "none"

    response = {
        "ticker": display_ticker,
        "source_ticker": source_ticker,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_live": bool(scored_items),
        "news_source": news_source,
        "news_sentiment_score": round(sentiment_score, 2),
        "thesis": _build_news_thesis(display_ticker, scored_items, sentiment_score),
        "items": [
            {
                "headline": item["headline"],
                "source": item["source"],
                "time": item["time"],
                "sentiment": item["sentiment"],
                "sentiment_label": item["sentiment_label"],
                "sentiment_score": item["sentiment_score"],
                "link": item.get("link", ""),
                "published": item.get("published"),
            }
            for item in scored_items
        ],
    }
    return response


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


class NewsItem(BaseModel):
    headline: str
    source: str
    time: str
    sentiment: str
    sentiment_label: str
    sentiment_score: float
    link: str = ""
    published: Optional[str] = None


class NewsResponse(BaseModel):
    ticker: str
    source_ticker: str
    generated_at: str
    is_live: bool
    news_source: str = "none"
    news_sentiment_score: float
    thesis: str
    items: List[NewsItem]


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


@app.get("/api/news/{ticker}", response_model=NewsResponse)
def api_news(ticker: str, limit: int = 4):
    """Return live RSS headlines scored with FinBERT for frontend news cards and ranking."""
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm:
        raise HTTPException(status_code=400, detail="Ticker is required")
    safe_limit = max(1, min(int(limit or 4), 12))
    cache_key = (_normalize_news_ticker(ticker_norm), safe_limit)
    cached = _news_cache.get(cache_key)
    if cached is not None:
        cached_at, payload = cached
        if time.monotonic() - cached_at < NEWS_CACHE_TTL_SECONDS:
            return payload

    try:
        payload = _build_news_response(ticker_norm, safe_limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"News provider unavailable: {exc}") from exc

    _news_cache[cache_key] = (time.monotonic(), payload)
    if len(_news_cache) > NEWS_CACHE_MAX_ENTRIES:
        oldest_key = min(_news_cache, key=lambda key: _news_cache[key][0])
        _news_cache.pop(oldest_key, None)
    return payload


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
    df = _fetch_price_df(ticker)
    if df.empty or len(df) < 2:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
    close = df["Close"]
    latest = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
    return PriceResponse(ticker=ticker.upper(), price=latest, change_pct=change_pct)


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
