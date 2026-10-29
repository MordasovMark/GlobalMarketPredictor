"""US market Fear & Greed index, built from market data (same idea as CNN's).

Each component is a raw market measure, converted to 0-100 by asking "how
extreme is today versus the last year?" (rolling percentile). The index is
the average of the components that have data. Higher = greedier.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .cache import TTLCache
from .market_data import DataUnavailable, get_benchmark, get_histories
from .news import market_news
from .universe import training_universe

log = logging.getLogger(__name__)

_cache = TTLCache(ttl_seconds=30 * 60, max_entries=2)

COMPONENTS = {
    "momentum": ("Market momentum", "S&P 500 vs its 125-day average"),
    "strength": ("Stock price strength", "Net new 52-week highs vs lows across ~95 large caps"),
    "breadth": ("Stock price breadth", "Share of large caps trading above their 50-day average"),
    "volatility": ("Market volatility", "VIX vs its 50-day average (inverted: high VIX = fear)"),
    "safe_haven": ("Safe-haven demand", "Stocks vs Treasuries, 20-day return gap"),
    "junk_bonds": ("Junk-bond demand", "High-yield vs Treasury bonds, 20-day return gap"),
}


def label_for(value: float) -> str:
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


def rolling_percentile(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    def pct(w: np.ndarray) -> float:
        return float((w[:-1] < w[-1]).mean() * 100 + (w[:-1] == w[-1]).mean() * 50)

    return series.rolling(window, min_periods=min_periods).apply(pct, raw=True)


def _ret(df: pd.DataFrame | None, n: int) -> pd.Series | None:
    if df is None or len(df) < n + 5:
        return None
    return df["Close"] / df["Close"].shift(n) - 1


def raw_components(
    spx: pd.Series,
    universe: dict[str, pd.DataFrame],
    extras: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    raw = pd.DataFrame(index=spx.index)
    raw["momentum"] = spx / spx.rolling(125, min_periods=100).mean() - 1

    closes = pd.DataFrame({k: v["Close"] for k, v in universe.items()}).reindex(spx.index).ffill()
    if closes.shape[1] >= 10:
        hi = closes.rolling(252, min_periods=200).max()
        lo = closes.rolling(252, min_periods=200).min()
        valid = hi.notna().sum(axis=1).replace(0, np.nan)
        raw["strength"] = ((closes >= hi * 0.99).sum(axis=1) - (closes <= lo * 1.01).sum(axis=1)) / valid
        sma50 = closes.rolling(50, min_periods=40).mean()
        raw["breadth"] = (closes > sma50).sum(axis=1) / sma50.notna().sum(axis=1).replace(0, np.nan)

    vix = extras.get("^VIX")
    if vix is not None and len(vix) > 60:
        v = vix["Close"].reindex(spx.index).ffill()
        raw["volatility"] = -(v / v.rolling(50, min_periods=40).mean() - 1)
    else:  # fall back to realised volatility of the index
        rv = np.log(spx).diff().rolling(21).std()
        raw["volatility"] = -(rv / rv.rolling(252, min_periods=60).mean() - 1)

    spy20 = spx / spx.shift(20) - 1
    tlt20 = _ret(extras.get("TLT"), 20)
    if tlt20 is not None:
        raw["safe_haven"] = spy20 - tlt20.reindex(spx.index).ffill()
    hyg20, ief20 = _ret(extras.get("HYG"), 20), _ret(extras.get("IEF"), 20)
    if hyg20 is not None and ief20 is not None:
        raw["junk_bonds"] = (hyg20 - ief20).reindex(spx.index).ffill()
    return raw


def _drivers(scores: pd.Series) -> tuple[list[str], list[str]]:
    fear = [k for k, v in scores.sort_values().items() if v < 45][:3]
    greed = [k for k, v in scores.sort_values(ascending=False).items() if v > 55][:3]
    return fear, greed


def conclusion(value: float, scores: pd.Series) -> str:
    fear, greed = _drivers(scores)
    parts = [f"The index reads {round(value)} ({label_for(value)})."]
    if greed:
        parts.append("Pushing toward greed: " + ", ".join(COMPONENTS[k][0].lower() for k in greed) + ".")
    if fear:
        parts.append("Pushing toward fear: " + ", ".join(COMPONENTS[k][0].lower() for k in fear) + ".")
    parts.append(
        "Readings this extreme have historically been better contrarian than trend signals."
        if value <= 24 or value >= 76
        else "No extreme reading; sentiment alone is not a strong signal here."
    )
    return " ".join(parts)


def compute() -> dict:
    spx = get_benchmark("5y")["Close"]
    extras = get_histories(["^VIX", "TLT", "IEF", "HYG"], "5y")
    universe = get_histories(training_universe(), "2y")
    raw = raw_components(spx, universe, extras)
    scores = raw.apply(rolling_percentile).dropna(how="all")
    index = scores.mean(axis=1, skipna=True).dropna()
    if index.empty:
        raise DataUnavailable("Not enough history to compute Fear & Greed")

    latest = scores.loc[index.index[-1]].dropna()
    value = float(index.iloc[-1])

    def ago(days: int) -> float | None:
        target = index.index[-1] - pd.Timedelta(days=days)
        prior = index[index.index <= target]
        return round(float(prior.iloc[-1]), 1) if len(prior) else None

    monthly = index.resample("ME").last().dropna().tail(12)
    spx_m = spx.resample("ME").last().reindex(monthly.index)
    timeline = [
        {
            "month": d.strftime("%b"),
            "date": d.strftime("%Y-%m-%d"),
            "fearGreed": round(float(v), 1),
            "sp500": round(float(spx_m.loc[d]), 2),
        }
        for d, v in monthly.items()
    ]

    try:
        headlines = market_news()
    except Exception as exc:  # noqa: BLE001
        log.info("Market news unavailable: %s", exc)
        headlines = []
    by_conf = sorted(headlines, key=lambda i: -i["sentiment_score"])
    fear_news = [i for i in by_conf if i["sentiment"] == "bad"][:3]
    greed_news = [i for i in by_conf if i["sentiment"] == "good"][:3]

    return {
        "value": round(value, 1),
        "label": label_for(value),
        "as_of": index.index[-1].strftime("%Y-%m-%d"),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "history": [
            {"period": "Current", "value": round(value, 1)},
            {"period": "1 Week Ago", "value": ago(7)},
            {"period": "1 Month Ago", "value": ago(30)},
            {"period": "1 Year Ago", "value": ago(365)},
        ],
        "components": [
            {
                "key": k,
                "name": COMPONENTS[k][0],
                "description": COMPONENTS[k][1],
                "score": round(float(latest[k]), 1),
                "label": label_for(float(latest[k])),
            }
            for k in COMPONENTS
            if k in latest
        ],
        "timeline": timeline,
        "drivers": {"fear": fear_news, "greed": greed_news},
        "conclusion": conclusion(value, latest),
    }


def fear_greed() -> dict:
    return _cache.get_or_load("fg", compute)
