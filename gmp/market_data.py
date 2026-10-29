"""Market data access: one place that talks to Yahoo Finance (or offline CSVs).

All functions return clean, tz-naive, date-sorted OHLCV frames with columns
``Open, High, Low, Close, Volume``. Results are cached so a busy dashboard
doesn't hammer Yahoo.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .cache import TTLCache
from .settings import settings
from .universe import BENCHMARK_FALLBACKS, to_yahoo

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
PERIOD_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827, "10y": 3653, "max": 36500}

_history_cache = TTLCache(ttl_seconds=15 * 60, max_entries=512)
_intraday_cache = TTLCache(ttl_seconds=120, max_entries=64)
_quote_cache = TTLCache(ttl_seconds=60, max_entries=512)
_fundamentals_cache = TTLCache(ttl_seconds=12 * 3600, max_entries=256)


class DataUnavailable(RuntimeError):
    """Raised when no usable price data exists for a symbol."""


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip().title() for c in out.columns]
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]
    for col in OHLCV:
        if col not in out.columns:
            out[col] = out["Close"] if col != "Volume" and "Close" in out.columns else np.nan
    out = out[OHLCV].apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    out.index = idx.tz_convert(None) if idx.tz is not None else idx
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Offline CSV source
# ---------------------------------------------------------------------------
def _read_yf_csv(path: Path) -> pd.DataFrame:
    """Read a CSV written by ``yf.download(...).to_csv()`` (3 header rows) or a plain one."""
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
    if len(head) >= 2 and head[1].lower().startswith("ticker"):
        df = pd.read_csv(path, header=0, skiprows=[1, 2], index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    return clean_ohlcv(df)


@lru_cache(maxsize=512)
def _csv_history(symbol: str) -> pd.DataFrame:
    assert settings.price_csv_dir is not None
    candidates = [symbol, symbol.replace("-", "."), symbol.replace(".", "-")]
    for name in candidates:
        path = settings.price_csv_dir / f"{name}.csv"
        if path.is_file():
            return _read_yf_csv(path)
    raise DataUnavailable(f"No offline CSV for {symbol}")


def _slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    days = PERIOD_DAYS.get(period)
    if not days or df.empty:
        return df
    return df[df.index >= df.index[-1] - pd.Timedelta(days=days)]


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------
def _yf():
    if not settings.network_enabled:
        raise DataUnavailable("Network access is disabled (GMP_NETWORK=0)")
    import yfinance as yf  # imported lazily: slow import, and optional in tests

    return yf


def _download_many(symbols: list[str], period: str, interval: str = "1d") -> dict[str, pd.DataFrame]:
    yf = _yf()
    raw = yf.download(
        symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for sym in symbols:
            if sym in level0:
                df = clean_ohlcv(raw[sym])
                if not df.empty:
                    out[sym] = df
    elif len(symbols) == 1:
        df = clean_ohlcv(raw)
        if not df.empty:
            out[symbols[0]] = df
    return out


def get_histories(tickers: Iterable[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Daily adjusted OHLCV for many tickers. Missing tickers are left out."""
    symbols = list(dict.fromkeys(to_yahoo(t) for t in tickers))
    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in symbols:
        hit = _history_cache.get((sym, period))
        if hit is not None:
            result[sym] = hit
        else:
            missing.append(sym)
    if not missing:
        return result

    if settings.price_csv_dir is not None:
        for sym in missing:
            try:
                df = _slice_period(_csv_history(sym), period)
            except DataUnavailable:
                continue
            _history_cache.set((sym, period), df)
            result[sym] = df
        return result

    # Batch in chunks so one bad symbol can't sink a huge request.
    for start in range(0, len(missing), 40):
        chunk = missing[start : start + 40]
        try:
            fetched = _download_many(chunk, period)
        except DataUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            log.warning("yfinance batch download failed for %s: %s", chunk, exc)
            fetched = {}
        for sym, df in fetched.items():
            _history_cache.set((sym, period), df)
            result[sym] = df
    return result


def get_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    sym = to_yahoo(ticker)
    df = get_histories([sym], period).get(sym)
    if df is None or len(df) < 2:
        raise DataUnavailable(f"No price data for {ticker}")
    return df


def get_benchmark(period: str = "2y") -> pd.DataFrame:
    histories = get_histories(list(BENCHMARK_FALLBACKS), period)
    for sym in BENCHMARK_FALLBACKS:
        if sym in histories and len(histories[sym]) > 50:
            return histories[sym]
    raise DataUnavailable("No benchmark data (SPY / ^GSPC)")


def get_intraday(ticker: str) -> pd.DataFrame:
    """Today's 5-minute bars."""
    sym = to_yahoo(ticker)
    if settings.price_csv_dir is not None:
        raise DataUnavailable("No intraday data in offline mode")

    def load() -> pd.DataFrame:
        yf = _yf()
        df = clean_ohlcv(yf.Ticker(sym).history(period="1d", interval="5m", auto_adjust=True))
        return df if len(df) >= 2 else None

    df = _intraday_cache.get_or_load(sym, load)
    if df is None:
        raise DataUnavailable(f"No intraday data for {ticker}")
    return df


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------
def _finnhub_quote(sym: str) -> dict | None:
    if not settings.finnhub_api_key or not settings.network_enabled:
        return None
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": sym.replace("-", "."), "token": settings.finnhub_api_key},
            timeout=settings.http_timeout_s,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        price, prev = float(data.get("c") or 0), float(data.get("pc") or 0)
        if price <= 0 or prev <= 0:
            return None
        return {"price": price, "prev_close": prev, "source": "finnhub"}
    except (requests.RequestException, ValueError):
        return None


def get_quotes(tickers: Iterable[str]) -> dict[str, dict]:
    """Latest price and day change for each ticker. Keys are Yahoo symbols."""
    symbols = list(dict.fromkeys(to_yahoo(t) for t in tickers))
    out: dict[str, dict] = {}
    need: list[str] = []
    for sym in symbols:
        hit = _quote_cache.get(sym)
        if hit is not None:
            out[sym] = hit
        else:
            need.append(sym)

    from_daily: list[str] = []
    for sym in need:
        q = _finnhub_quote(sym)
        if q is None:
            from_daily.append(sym)
        else:
            out[sym] = q
    if from_daily:
        if settings.price_csv_dir is not None:
            daily = get_histories(from_daily, "1mo")
        else:
            try:
                daily = _download_many(from_daily, "5d")
            except Exception as exc:  # noqa: BLE001
                log.warning("Quote download failed: %s", exc)
                daily = {}
        for sym, df in daily.items():
            if len(df) >= 2:
                out[sym] = {
                    "price": float(df["Close"].iloc[-1]),
                    "prev_close": float(df["Close"].iloc[-2]),
                    "source": "yahoo",
                    "as_of": df.index[-1].isoformat(),
                }

    for sym in need:
        q = out.get(sym)
        if q is None:
            continue
        change = q["price"] - q["prev_close"]
        q["change"] = change
        q["change_pct"] = change / q["prev_close"] * 100.0 if q["prev_close"] else 0.0
        _quote_cache.set(sym, q)
    return out


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
FUNDAMENTAL_FIELDS = {
    "marketCap": "market_cap",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "priceToBook": "price_to_book",
    "freeCashflow": "free_cashflow",
    "returnOnEquity": "roe",
    "profitMargins": "profit_margin",
    "grossMargins": "gross_margin",
    "debtToEquity": "debt_to_equity",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "dividendYield": "dividend_yield",
    "longBusinessSummary": "summary",
}


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def get_fundamentals(ticker: str) -> dict:
    """Valuation / quality fields from Yahoo ``info``. Empty dict when unavailable."""
    sym = to_yahoo(ticker)
    if settings.price_csv_dir is not None or not settings.network_enabled:
        return {}

    def load() -> dict:
        try:
            info = _yf().Ticker(sym).info or {}
        except Exception as exc:  # noqa: BLE001
            log.info("Fundamentals unavailable for %s: %s", sym, exc)
            return {}
        out: dict = {}
        for src, dst in FUNDAMENTAL_FIELDS.items():
            val = info.get(src)
            out[dst] = val if dst == "summary" else _finite(val)
        # Yahoo has reported dividendYield both as 0.0044 and as 0.44 over time.
        dy = out.get("dividend_yield")
        if dy is not None and dy > 0.25:
            out["dividend_yield"] = dy / 100.0
        return out

    return _fundamentals_cache.get_or_load(sym, load) or {}
