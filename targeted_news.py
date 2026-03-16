"""
Targeted News Resolver: scrape financial sites, resolve entities via LLM,
map to tickers/sectors, and aggregate sentiment with exponential decay.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import ASSET_MAP, get_tickers_for_sector

LOGGER_NAME = "targeted_news"
DEFAULT_DECAY_RATE = 0.1  # lambda in exp(-lambda * t_days)
RAW_NEWS_DIR = Path("data") / "raw_news"

logger = logging.getLogger(LOGGER_NAME)


def _configure_logging() -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def scrape_custom_site(url: str) -> pd.DataFrame:
    """
    Extract headlines from a specific financial news site.

    Supports Reuters (reuters.com) and Calcalist (calcalist.co.il) by default.
    For other URLs, uses a generic headline extraction from article links.

    Parameters
    ----------
    url : str
        Full URL of the news section page (e.g. Reuters Markets or Calcalist).

    Returns
    -------
    pandas.DataFrame
        Columns: title, link, published, source. `published` may be empty string
        if not available from the page.
    """
    _configure_logging()
    url_lower = url.lower()
    if "reuters.com" in url_lower:
        return _scrape_reuters(url)
    if "calcalist.co.il" in url_lower:
        return _scrape_calcalist(url)
    return _scrape_generic(url)


def _scrape_reuters(url: str) -> pd.DataFrame:
    """Extract headlines from a Reuters section page."""
    try:
        resp = requests.get(url, timeout=15, headers=_user_agent_headers())
        resp.raise_for_status()
    except Exception as exc:
        logger.exception("Failed to fetch Reuters URL %s: %s", url, exc)
        raise RuntimeError(f"Failed to fetch {url}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    records: List[Dict[str, str]] = []
    source_name = "Reuters"

    # Reuters uses data-testid and article links; look for headline links
    for a in soup.select("a[href*='/business/'], a[href*='/markets/']"):
        href = a.get("href") or ""
        if not href.startswith("http"):
            href = "https://www.reuters.com" + href.split("?")[0]
        title = (a.get_text() or "").strip()
        if len(title) < 10 or len(title) > 300:
            continue
        if any(x in href for x in ["/business/", "/markets/", "/world/"]):
            records.append({
                "title": title,
                "link": href,
                "published": "",
                "source": source_name,
            })

    # Deduplicate by link
    seen = set()
    unique: List[Dict[str, str]] = []
    for r in records:
        key = r["link"]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return pd.DataFrame(unique, columns=["title", "link", "published", "source"])


def _scrape_calcalist(url: str) -> pd.DataFrame:
    """Extract headlines from Calcalist (Israeli financial news)."""
    try:
        resp = requests.get(url, timeout=15, headers=_user_agent_headers())
        resp.raise_for_status()
    except Exception as exc:
        logger.exception("Failed to fetch Calcalist URL %s: %s", url, exc)
        raise RuntimeError(f"Failed to fetch {url}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []
    source_name = "Calcalist"

    for a in soup.select("a[href*='calcalist.co.il']"):
        href = (a.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        title = (a.get_text() or "").strip()
        if len(title) < 8 or len(title) > 250:
            continue
        records.append({
            "title": title,
            "link": href,
            "published": "",
            "source": source_name,
        })

    seen = set()
    unique = []
    for r in records:
        key = r["link"]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return pd.DataFrame(unique, columns=["title", "link", "published", "source"])


def _scrape_generic(url: str) -> pd.DataFrame:
    """Generic extraction: any link that looks like an article with a title."""
    try:
        resp = requests.get(url, timeout=15, headers=_user_agent_headers())
        resp.raise_for_status()
    except Exception as exc:
        logger.exception("Failed to fetch URL %s: %s", url, exc)
        raise RuntimeError(f"Failed to fetch {url}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    base = _base_url(url)
    source_name = base.replace("www.", "").split(".")[0].title()
    records = []

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href.startswith("http") and base:
            href = base.rstrip("/") + ("/" if not href.startswith("/") else "") + href
        if not href or "javascript:" in href or "#" == href:
            continue
        title = (a.get_text() or "").strip()
        if len(title) < 15 or len(title) > 280:
            continue
        records.append({
            "title": title,
            "link": href,
            "published": "",
            "source": source_name,
        })

    seen = set()
    unique = []
    for r in records:
        key = r["link"]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return pd.DataFrame(unique, columns=["title", "link", "published", "source"])


def _user_agent_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }


def _base_url(url: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


# ---------------------------------------------------------------------------
# LLM Entity Recognition
# ---------------------------------------------------------------------------

@dataclass
class EntityResult:
    ticker: Optional[str]
    category: Optional[str]
    relevance_score: float


def entity_recognition_llm(headline: str) -> EntityResult:
    """
    Use an LLM (OpenAI API) to extract ticker, category, and relevance from a headline.

    Input: News headline.
    Output: EntityResult with ticker (if specific company), category (e.g. Tech, Banks),
    and relevance_score in [0, 1].

    Set OPENAI_API_KEY in the environment. Falls back to a no-API stub if unset.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; using fallback entity recognition.")
        return _entity_fallback(headline)

    prompt = (
        "You are a financial news analyst. For the following headline, respond with ONLY a single JSON object "
        "and no other text. Use this exact format:\n"
        '{"ticker": "AAPL or null", "category": "Tech or Banks or Commodities or null", "relevance_score": 0.9}\n'
        "Rules: ticker is the stock symbol if a specific company is clearly mentioned, otherwise null. "
        "category is the sector (e.g. Tech, Banks, Commodities) if relevant, otherwise null. "
        "relevance_score is between 0 and 1 for how relevant this headline is to markets. "
        "Headline:\n" + headline
    )

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown code fence if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        ticker = data.get("ticker")
        if ticker is not None and not isinstance(ticker, str):
            ticker = None
        if ticker is not None:
            ticker = (ticker or "").strip() or None
        category = data.get("category")
        if category is not None and isinstance(category, str):
            category = category.strip() or None
        else:
            category = None
        score = float(data.get("relevance_score", 0.5))
        score = max(0.0, min(1.0, score))
        return EntityResult(ticker=ticker, category=category, relevance_score=score)
    except Exception as exc:
        logger.exception("LLM entity recognition failed for headline %r: %s", headline[:80], exc)
        return _entity_fallback(headline)


def _entity_fallback(headline: str) -> EntityResult:
    """Fallback when no API: try to match known tickers/sectors from config."""
    text = headline.upper()
    for category, tickers in ASSET_MAP.items():
        for t in tickers:
            sym = t.upper().replace(".TA", "").replace("=F", "")
            if sym in text or t.upper() in text:
                return EntityResult(ticker=t, category=category, relevance_score=0.7)
    if "BANK" in text or "BANKS" in text:
        return EntityResult(ticker=None, category="Banks", relevance_score=0.6)
    if "TECH" in text or "APPLE" in text or "GOOGLE" in text or "MICROSOFT" in text:
        return EntityResult(ticker=None, category="Tech", relevance_score=0.6)
    return EntityResult(ticker=None, category=None, relevance_score=0.3)


# ---------------------------------------------------------------------------
# Mapping logic
# ---------------------------------------------------------------------------


def map_entity_to_tickers(entity: EntityResult) -> List[str]:
    """
    Map entity recognition result to a list of tickers.

    - If ticker is set and in our universe, return [ticker].
    - If only category/sector is set, return all tickers in that sector from config.
    - Otherwise return [].
    """
    all_tickers = []
    for tickers in ASSET_MAP.values():
        all_tickers.extend(tickers)
    all_tickers = list(dict.fromkeys(all_tickers))

    if entity.ticker:
        if entity.ticker in all_tickers:
            return [entity.ticker]
        # Normalize .TA / =F for comparison
        for t in all_tickers:
            if t.upper() == entity.ticker.upper():
                return [t]
        return [entity.ticker]

    if entity.category:
        return get_tickers_for_sector(entity.category)
    return []


# ---------------------------------------------------------------------------
# Exponential decay aggregation
# ---------------------------------------------------------------------------


def aggregate_sentiment_exponential_decay(
    items: List[Tuple[float, datetime]],
    decay_rate: float = DEFAULT_DECAY_RATE,
    now: Optional[datetime] = None,
) -> float:
    """
    Aggregate sentiment scores with exponential decay by time.

    weight_i = exp(-decay_rate * days_since_i)
    weighted_score = sum(weight_i * score_i) / sum(weight_i)

    Parameters
    ----------
    items : list of (sentiment_score, datetime)
        Pairs of score and publication time.
    decay_rate : float
        Lambda; higher = older news decays faster.
    now : datetime, optional
        Reference time (default: utcnow()).

    Returns
    -------
    float
        Weighted average sentiment, or 0.0 if no items.
    """
    if not items:
        return 0.0
    now = now or datetime.now(timezone.utc)
    weighted_sum = 0.0
    weight_sum = 0.0
    for score, dt in items:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        days = max(0.0, delta.total_seconds() / 86400.0)
        w = float(math.exp(-decay_rate * days))
        weighted_sum += w * score
        weight_sum += w
    if weight_sum <= 0:
        return 0.0
    return weighted_sum / weight_sum


# ---------------------------------------------------------------------------
# Integration: run pipeline and merge with sentiment
# ---------------------------------------------------------------------------


def run_targeted_pipeline(
    url: str,
    sentiment_pipeline=None,
    decay_rate: float = DEFAULT_DECAY_RATE,
) -> pd.DataFrame:
    """
    Scrape URL -> entity recognition -> map to tickers -> run sentiment -> return
    enriched DataFrame with source and optional decay weight.

    If sentiment_pipeline is None, does not add sentiment columns (caller can
    run sentiment_analyzer.analyze_sentiment on the result).
    """
    _configure_logging()
    df = scrape_custom_site(url)
    if df.empty:
        return df

    source_name = df["source"].iloc[0] if "source" in df.columns else "Custom"
    if "published" not in df.columns:
        df["published"] = ""
    if "source" not in df.columns:
        df["source"] = source_name

    # Entity recognition and mapping
    ticker_lists: List[List[str]] = []
    for title in df["title"]:
        entity = entity_recognition_llm(str(title))
        tickers = map_entity_to_tickers(entity)
        ticker_lists.append(tickers)

    df["_mapped_tickers"] = ticker_lists

    if sentiment_pipeline is not None:
        try:
            titles = df["title"].astype(str).tolist()
            results = sentiment_pipeline(titles)
            df["sentiment_label"] = [str(r.get("label", "")) for r in results]
            df["sentiment_score"] = [float(r.get("score", 0.0)) for r in results]
        except Exception as exc:
            logger.exception("Sentiment pipeline failed: %s", exc)

    # Normalize published for decay
    df["_published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    if hasattr(df["_published_dt"].dt, "tz") and df["_published_dt"].dt.tz is not None:
        df["_published_dt"] = df["_published_dt"].dt.tz_localize(None)
    now_ref = datetime.now(timezone.utc).replace(tzinfo=None)
    df["_published_dt"] = df["_published_dt"].fillna(pd.Timestamp(now_ref))

    # Per-row decay weight (for optional use in aggregation)
    now_ref = datetime.now(timezone.utc).replace(tzinfo=None)
    df["decay_weight"] = df["_published_dt"].apply(
        lambda t: math.exp(-decay_rate * (pd.Timestamp(now_ref) - pd.Timestamp(t)).total_seconds() / 86400.0)
    )
    df.drop(columns=["_published_dt"], inplace=True)

    return df


def merge_targeted_news_into_assets(
    targeted_df: pd.DataFrame,
    raw_news_dir: Path = RAW_NEWS_DIR,
) -> None:
    """
    For each row in targeted_df, append it to the CSV of each mapped ticker.
    Requires columns: title, link, published, source, _mapped_tickers, and
    optionally sentiment_label, sentiment_score.
    """
    if targeted_df.empty or "_mapped_tickers" not in targeted_df.columns:
        return
    raw_news_dir = Path(raw_news_dir)
    raw_news_dir.mkdir(parents=True, exist_ok=True)

    out_cols = ["title", "published", "link", "source"]
    if "sentiment_label" in targeted_df.columns:
        out_cols.extend(["sentiment_label", "sentiment_score"])

    for _, row in targeted_df.iterrows():
        tickers = row.get("_mapped_tickers") or []
        if not isinstance(tickers, list):
            tickers = [tickers] if tickers else []
        for ticker in tickers:
            path = raw_news_dir / f"{ticker}_news.csv"
            new_row = {c: row.get(c, "") for c in out_cols}
            new_row["published"] = new_row.get("published") or ""
            try:
                if path.is_file():
                    existing = pd.read_csv(path)
                    if "source" not in existing.columns:
                        existing["source"] = ""
                    existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    existing = pd.DataFrame([new_row])
                existing.to_csv(path, index=False)
            except Exception as exc:
                logger.exception("Failed to append targeted news to %s: %s", path, exc)


if __name__ == "__main__":
    _configure_logging()
    # Example: scrape Reuters Markets and print first few rows
    reuters_url = "https://www.reuters.com/markets/"
    try:
        df = scrape_custom_site(reuters_url)
        print(f"Scraped {len(df)} headlines from Reuters.")
        print(df[["title", "source"]].head())
    except Exception as e:
        print("Scrape failed:", e)
