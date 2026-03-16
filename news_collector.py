from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, List, Tuple
import urllib.parse

import feedparser
import pandas as pd

from config import ASSET_MAP


LOGGER_NAME: Final[str] = "news_collector"
NEWS_LOOKBACK_DAYS: Final[int] = 7
RSS_FEEDS: Final[dict[str, str]] = {
    "Yahoo Finance (Global)": "https://finance.yahoo.com/news/rssindex",
    "CNBC Markets (Global)": "https://search.cnbc.com/rs/search/combinedcms/view.xml?id=10000664",
    "Times of Israel (Geopolitics/Security)": "https://www.timesofisrael.com/feed/",
    "Globes English (Israel)": "https://en.globes.co.il/webservice/rss/rssfeeder.asmx/FeederNode?iID=821",
}
GEO_RISK_KEYWORDS: Final[List[str]] = [
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
DEFAULT_OUTPUT_PATH: Final[Path] = Path("data") / "news_data.csv"
NEWS_SEARCH_BASE_URL: Final[str] = "https://news.google.com/rss/search"


logger = logging.getLogger(LOGGER_NAME)


def _build_default_output_path(ticker_or_name: str) -> Path:
    """
    Build a default CSV output path for a given ticker or asset name.

    Parameters
    ----------
    ticker_or_name : str
        Ticker symbol or human-readable asset name.

    Returns
    -------
    Path
        Path within the `data/` directory where news for this asset
        will be stored.
    """
    safe_name = (
        ticker_or_name.strip()
        .replace("^", "")
        .replace("=", "")
        .replace("/", "_")
        .replace(" ", "_")
    )
    if not safe_name:
        safe_name = "asset"
    return Path("data") / f"{safe_name.lower()}_news.csv"


def _configure_logging() -> None:
    """
    Configure basic logging for the news collection module.

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


def _build_news_search_rss_url(query: str) -> str:
    """
    Build an RSS URL for a textual news search query.

    This implementation uses the Google News RSS endpoint to search for
    arbitrary phrases such as "AAPL stock news OR AAPL market analysis".

    Parameters
    ----------
    query : str
        Text query describing the asset and context to search for.

    Returns
    -------
    str
        Fully-qualified RSS URL for the search query.
    """
    encoded_query = urllib.parse.quote_plus(query)
    return (
        f"{NEWS_SEARCH_BASE_URL}?q={encoded_query}"
        "&hl=en-US&gl=US&ceid=US:en"
    )


def geopolitical_risk_score(text: str) -> float:
    """
    Compute a simple geopolitical risk score in [0.0, 1.0] from text.

    This is a keyword-presence scanner intended for event-driven features.
    """
    if not text:
        return 0.0
    haystack = str(text).lower()
    matches = sum(1 for kw in GEO_RISK_KEYWORDS if kw.lower() in haystack)
    if matches <= 0:
        return 0.0
    return min(1.0, matches / float(len(GEO_RISK_KEYWORDS)))


def aggregate_daily_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-article news into daily features (last 7 days only).

    Returns a DataFrame with:
    - date (normalized published datetime)
    - geopolitical_risk_score (mean)
    - article_count (count)
    """
    if df.empty or "published" not in df.columns:
        return pd.DataFrame(columns=["date", "geopolitical_risk_score", "article_count"])
    pub = pd.to_datetime(df["published"], errors="coerce")
    tmp = df.copy()
    tmp["date"] = pub.dt.normalize()
    out = (
        tmp.dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(
            geopolitical_risk_score=("geopolitical_risk_score", "mean"),
            article_count=("geopolitical_risk_score", "size"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def _parse_feed(url: str, source_name: str) -> list[dict[str, object]]:
    """
    Parse an RSS/Atom feed URL into normalized record dicts.

    This is defensive: if a feed is down or malformed, we log and return [].
    """
    try:
        parsed = feedparser.parse(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch/parse feed %s (%s): %s", source_name, url, exc)
        return []

    if getattr(parsed, "bozo", False):
        bozo_exc = getattr(parsed, "bozo_exception", None)
        logger.warning(
            "Malformed feed %s (%s). Continuing. Error: %s",
            source_name,
            url,
            bozo_exc,
        )

    entries = getattr(parsed, "entries", []) or []
    records: list[dict[str, object]] = []
    for entry in entries:
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        published = getattr(entry, "published", "") or getattr(entry, "updated", "")

        full_text = f"{title} {summary}".strip()
        risk = geopolitical_risk_score(full_text)
        records.append(
            {
                "title": str(title),
                "published": str(published),
                "link": str(link),
                "summary": str(summary),
                "source": str(source_name),
                "geopolitical_risk_score": float(risk),
            }
        )
    return records


def _is_relevant_to_ticker(text: str, ticker: str) -> bool:
    """
    Heuristic relevance filter for broad market RSS feeds.

    We keep an entry if:
    - it mentions the ticker (case-insensitive), or
    - it mentions a '$TICKER' form.
    """
    if not ticker:
        return True
    hay = (text or "").lower()
    t = ticker.lower()
    return (t in hay) or (f"${t}" in hay)


def fetch_financial_news(
    search_query: str,
    output_path: Path | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Fetch financial news via an RSS feed for a given search query.

    This function downloads and parses a news RSS feed (currently backed
    by Google News) for the specified textual query, extracts the news
    `title`, `published`, and `link` fields for each entry, and returns
    the result as a :class:`pandas.DataFrame`. The resulting DataFrame is
    also written to a CSV file at the provided output path.

    Parameters
    ----------
    search_query : str
        Textual search phrase such as "AAPL stock news OR AAPL market
        analysis".
    output_path : Path, optional
        Filesystem path where the news CSV file will be saved. The parent
        directory will be created if it does not exist. If omitted, a
        default path derived from the query will be used.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per news item with columns
        `title`, `published`, and `link`.

    Raises
    ------
    RuntimeError
        If a network-related or parsing error occurs while fetching or
        processing the RSS feed, or if saving the CSV fails.
    ValueError
        If no news entries are found in the feed.
    """
    _configure_logging()

    if output_path is None:
        output_path = _build_default_output_path(search_query)

    # Collect from multiple professional feeds (broad coverage), then filter for this ticker.
    # We also keep Google News search as a fallback/booster for ticker-specific coverage.
    records: list[dict[str, object]] = []
    for source_name, url in RSS_FEEDS.items():
        records.extend(_parse_feed(url, source_name))

    rss_url = _build_news_search_rss_url(search_query)
    records.extend(_parse_feed(rss_url, "Google News Search"))

    if not records:
        msg = f"No news entries found across RSS feeds for query {search_query!r}."
        logger.warning(msg)
        df = pd.DataFrame(columns=["title", "published", "link", "summary", "source", "geopolitical_risk_score"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    try:
        df = pd.DataFrame.from_records(
            records,
            columns=["title", "published", "link", "summary", "source", "geopolitical_risk_score"],
        )
    except Exception as exc:  # noqa: BLE001
        msg = "Failed to construct DataFrame from RSS feed entries."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    # Ensure expected columns exist even if some feeds omit them.
    if "source" not in df.columns:
        df["source"] = ""
    if ticker:
        full_text = (df["title"].astype(str) + " " + df["summary"].astype(str)).astype(str)
        df = df[full_text.apply(lambda t: _is_relevant_to_ticker(t, ticker))].copy()

    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    pub = df["published"]
    if pub.dt.tz is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)).replace(tzinfo=None)
        pub = pub.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        cutoff = datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)
    df = df[pub.notna() & (pub >= cutoff)].copy()
    if df.empty:
        logger.warning("No news within the last %d days for query %r; saving empty file.", NEWS_LOOKBACK_DAYS, search_query)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("News data successfully saved to %s (%d articles within last %d days).", output_path, len(df), NEWS_LOOKBACK_DAYS)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to save news data to CSV at {output_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    return df


def _iter_assets() -> List[Tuple[str, str]]:
    """
    Iterate over all tickers from ASSET_MAP categories as (ticker, ticker) pairs.

    Returns
    -------
    list[tuple[str, str]]
        List of (name, ticker) pairs; name is the ticker symbol for search.
    """
    assets: List[Tuple[str, str]] = []
    for tickers in ASSET_MAP.values():
        for ticker in tickers:
            assets.append((str(ticker), str(ticker)))
    return assets


if __name__ == "__main__":
    _configure_logging()
    for name, ticker in _iter_assets():
        # Construct a more descriptive search phrase for this asset.
        search_phrase = f"{name} stock news OR {ticker} market analysis"
        try:
            output_path = Path("data") / "raw_news" / f"{ticker}_news.csv"
            news_df = fetch_financial_news(
                search_query=search_phrase,
                output_path=output_path,
                ticker=ticker,
            )
            # Also write a daily aggregated view for event-driven features.
            daily_path = Path("data") / "raw_news" / f"{ticker}_news_daily.csv"
            daily_df = aggregate_daily_news(news_df)
            daily_df.to_csv(daily_path, index=False)
            print(
                f"Fetched {len(news_df)} news items for {ticker} "
                f"and saved to {output_path}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "News collection failed for %s using search phrase %r: %s",
                ticker,
                search_phrase,
                exc,
            )


