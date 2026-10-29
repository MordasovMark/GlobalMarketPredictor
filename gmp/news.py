"""Live headlines from Yahoo Finance + Google News RSS, scored for sentiment."""

from __future__ import annotations

import datetime as dt
import html
import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import feedparser
import pandas as pd
import requests

from .cache import TTLCache
from .sentiment import backend_name, score_texts
from .settings import settings
from .universe import to_display, to_yahoo

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 7
HALF_LIFE_HOURS = 48.0  # older headlines count less
USER_AGENT = "Mozilla/5.0 (compatible; GlobalMarketPredictor/2.0)"

COMPANY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple", "iphone"),
    "NVDA": ("nvidia",),
    "MSFT": ("microsoft", "azure"),
    "TSLA": ("tesla",),
    "AMZN": ("amazon", "aws"),
    "META": ("meta platforms", "facebook", "instagram", "zuckerberg"),
    "GOOGL": ("google", "alphabet", "youtube"),
    "BRK-B": ("berkshire", "buffett"),
    "V": ("visa",),
    "UNH": ("unitedhealth", "optum"),
    "JPM": ("jpmorgan", "jp morgan", "dimon"),
    "JNJ": ("johnson & johnson", "j&j"),
    "WMT": ("walmart",),
    "XOM": ("exxon",),
    "MA": ("mastercard",),
    "AVGO": ("broadcom",),
    "PG": ("procter", "p&g"),
    "ORCL": ("oracle",),
    "COST": ("costco",),
    "HD": ("home depot",),
}

_news_cache = TTLCache(ttl_seconds=15 * 60, max_entries=256)
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rss")


def clean_text(value) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_feed(url: str) -> bytes | None:
    if not settings.network_enabled:
        return None
    try:
        resp = requests.get(url, timeout=settings.http_timeout_s, headers={"User-Agent": USER_AGENT})
        return resp.content if resp.status_code == 200 else None
    except requests.RequestException as exc:
        log.info("RSS fetch failed for %s: %s", url, exc)
        return None


def parse_feed(content: bytes | str, fallback_source: str) -> list[dict]:
    parsed = feedparser.parse(content)
    records = []
    for entry in parsed.entries or []:
        headline = clean_text(entry.get("title"))
        if not headline:
            continue
        source = entry.get("source")
        source_title = source.get("title") if isinstance(source, dict) else getattr(source, "title", None)
        # Google News appends " - Publisher" to titles.
        if source_title and headline.endswith(f" - {source_title}"):
            headline = headline[: -len(source_title) - 3].strip()
        published = pd.to_datetime(entry.get("published") or entry.get("updated") or None, errors="coerce", utc=True)
        records.append(
            {
                "headline": headline,
                "summary": clean_text(entry.get("summary") or entry.get("description")),
                "source": clean_text(source_title) or fallback_source,
                "link": str(entry.get("link") or ""),
                "published_ts": None if pd.isna(published) else published,
            }
        )
    return records


def is_relevant(headline: str, symbol: str) -> bool:
    text = headline.lower()
    base = symbol.split("-")[0].lower()
    # Short tickers like V, MA, HD are ordinary words; only accept "$V" or names.
    patterns = [rf"\${re.escape(base)}\b"]
    if len(base) >= 3:
        patterns.append(rf"\b{re.escape(base)}\b")
    if any(re.search(p, text) for p in patterns):
        return True
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in COMPANY_KEYWORDS.get(symbol, ()))


def relative_time(ts: pd.Timestamp | None, now: pd.Timestamp | None = None) -> str:
    if ts is None:
        return "recent"
    now = now or pd.Timestamp.now(tz="UTC")
    secs = max(0, int((now - ts).total_seconds()))
    if secs < 3600:
        return f"{max(1, secs // 60)}m ago"
    if secs < 86_400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86_400}d ago"


def _collect(feeds: list[tuple[str, str]], keep) -> list[dict]:
    contents = list(_pool.map(lambda f: _fetch_feed(f[0]), feeds))
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)
    seen, items = set(), []
    for (_url, source), content in zip(feeds, contents):
        if not content:
            continue
        for rec in parse_feed(content, source):
            ts = rec["published_ts"]
            if ts is not None and ts < cutoff:
                continue
            if not keep(rec):
                continue
            key = re.sub(r"\W+", "", rec["headline"].lower())[:80]
            if key and key not in seen:
                seen.add(key)
                items.append(rec)
    items.sort(key=lambda r: r["published_ts"] or pd.Timestamp("1970-01-01", tz="UTC"), reverse=True)
    return items


def _score(items: list[dict]) -> list[dict]:
    sentiments = score_texts([f"{i['headline']}. {i['summary'][:300]}".strip() for i in items])
    now = pd.Timestamp.now(tz="UTC")
    ui = {"positive": "good", "negative": "bad", "neutral": "neutral"}
    out = []
    for item, s in zip(items, sentiments):
        ts = item["published_ts"]
        age_h = (now - ts).total_seconds() / 3600 if ts is not None else 24.0
        out.append(
            {
                "headline": item["headline"],
                "source": item["source"],
                "link": item["link"],
                "time": relative_time(ts, now),
                "published": ts.isoformat() if ts is not None else None,
                "sentiment": ui[s.label],
                "sentiment_label": s.label,
                "sentiment_score": round(s.confidence, 4),
                "_net": s.net * s.confidence if s.label != "neutral" else 0.0,
                "_weight": 0.5 ** (max(age_h, 0.0) / HALF_LIFE_HOURS),
            }
        )
    return out


def aggregate_score(scored: list[dict]) -> float:
    """0-100, 50 = neutral. Recency-weighted; shrinks to 50 with few headlines."""
    if not scored:
        return 50.0
    wsum = sum(i["_weight"] for i in scored)
    net = sum(i["_net"] * i["_weight"] for i in scored) / wsum if wsum else 0.0
    shrink = len(scored) / (len(scored) + 3)  # 1 headline -> 25% weight, 12 -> 80%
    return round(50 + 50 * max(-1.0, min(1.0, net * 1.6)) * shrink, 2)


def daily_flow(scored: list[dict], days: int = 7) -> list[dict]:
    today = dt.datetime.now(dt.UTC).date()
    buckets = {today - dt.timedelta(days=k): {"pos": 0, "neg": 0, "neu": 0} for k in range(days)}
    for i in scored:
        if not i["published"]:
            continue
        day = pd.Timestamp(i["published"]).date()
        if day in buckets:
            key = {"good": "pos", "bad": "neg"}.get(i["sentiment"], "neu")
            buckets[day][key] += 1
    return [{"w": d.strftime("%a"), **v} for d, v in sorted(buckets.items())]


def thesis(ticker: str, scored: list[dict], score: float) -> str:
    if not scored:
        return f"No recent headlines found for {ticker}; news sentiment is treated as neutral."
    pos = sum(i["sentiment"] == "good" for i in scored)
    neg = sum(i["sentiment"] == "bad" for i in scored)
    tone = "positive" if score >= 60 else "negative" if score <= 40 else "mixed"
    lead = scored[0]
    return f"News tone is {tone} ({pos} positive, {neg} negative of {len(scored)}). Latest: {lead['source']}: {lead['headline']}"


def ticker_news(ticker: str, limit: int = 4) -> dict:
    symbol = to_yahoo(ticker)
    display = to_display(symbol)

    def load() -> dict:
        query = urllib.parse.quote_plus(f'"{display}" stock')
        names = COMPANY_KEYWORDS.get(symbol)
        if names:
            query = urllib.parse.quote_plus(f'{names[0]} stock OR "{display}" stock')
        feeds = [
            (f"https://finance.yahoo.com/rss/headline?s={urllib.parse.quote(symbol)}", "Yahoo Finance"),
            (f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en", "Google News"),
        ]
        items = _collect(feeds, lambda r: is_relevant(r["headline"], symbol))[:24]
        scored = _score(items)
        score = aggregate_score(scored)
        return {
            "ticker": display,
            "source_ticker": symbol,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "is_live": bool(scored),
            "sentiment_backend": backend_name(),
            "news_sentiment_score": score,
            "headline_count": len(scored),
            "thesis": thesis(display, scored, score),
            "daily_flow": daily_flow(scored),
            "items": scored,
        }

    full = _news_cache.get_or_load(symbol, load)
    public = {k: v for k, v in full.items() if k != "items"}
    public["items"] = [{k: v for k, v in i.items() if not k.startswith("_")} for i in full["items"][:limit]]
    return public


def market_news(limit: int = 30) -> list[dict]:
    """Broad US market headlines, scored (used for fear / greed drivers)."""

    def load() -> list[dict]:
        q = urllib.parse.quote_plus("stock market OR Wall Street OR S&P 500")
        feeds = [
            (f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en", "Google News"),
            ("https://finance.yahoo.com/rss/headline?s=%5EGSPC", "Yahoo Finance"),
        ]
        return _score(_collect(feeds, lambda r: True)[:limit])

    items = _news_cache.get_or_load("__market__", load)
    return [{k: v for k, v in i.items() if not k.startswith("_")} for i in items]
