"""Which stocks we track, and ticker normalisation.

* ``DASHBOARD`` – the 20 large caps shown in the UI (ratings, tables, portfolio).
* ``ASSET_MAP`` – the wider research universe by sector. The forecaster is
  trained on all US names here: more cross-sectional data makes a pooled
  model far more stable than 20 tickers alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Stock:
    ticker: str  # display ticker, e.g. "BRK.B"
    name: str
    sector: str
    domain: str


DASHBOARD: tuple[Stock, ...] = (
    Stock("AAPL", "Apple Inc.", "Technology", "apple.com"),
    Stock("NVDA", "NVIDIA Corp.", "Technology", "nvidia.com"),
    Stock("MSFT", "Microsoft Corp.", "Technology", "microsoft.com"),
    Stock("TSLA", "Tesla Inc.", "Consumer Discretionary", "tesla.com"),
    Stock("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "amazon.com"),
    Stock("META", "Meta Platforms", "Communication Services", "meta.com"),
    Stock("GOOGL", "Alphabet Inc.", "Communication Services", "google.com"),
    Stock("BRK.B", "Berkshire Hathaway", "Financials", "berkshirehathaway.com"),
    Stock("V", "Visa Inc.", "Financials", "visa.com"),
    Stock("UNH", "UnitedHealth Group", "Healthcare", "unitedhealthgroup.com"),
    Stock("JPM", "JPMorgan Chase", "Financials", "jpmorganchase.com"),
    Stock("JNJ", "Johnson & Johnson", "Healthcare", "jnj.com"),
    Stock("WMT", "Walmart Inc.", "Consumer Staples", "walmart.com"),
    Stock("XOM", "Exxon Mobil Corp.", "Energy", "exxonmobil.com"),
    Stock("MA", "Mastercard Inc.", "Financials", "mastercard.com"),
    Stock("AVGO", "Broadcom Inc.", "Technology", "broadcom.com"),
    Stock("PG", "Procter & Gamble", "Consumer Staples", "pg.com"),
    Stock("ORCL", "Oracle Corp.", "Technology", "oracle.com"),
    Stock("COST", "Costco Wholesale", "Consumer Staples", "costco.com"),
    Stock("HD", "Home Depot", "Consumer Discretionary", "homedepot.com"),
)

ASSET_MAP: dict[str, list[str]] = {
    "Technology & Semiconductors": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "ORCL",
        "CRM",
        "AMD",
        "INTC",
        "CSCO",
        "QCOM",
        "TXN",
        "IBM",
        "ADBE",
        "AMAT",
        "INTU",
    ],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR"],
    "Aerospace & Defense": ["RTX", "LMT", "BA", "GD", "NOC", "HWM", "TDG", "TXT", "LHX", "LDOS", "PLTR"],
    "Financials": ["JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "C", "AXP", "SPGI", "BLK", "SCHW", "BRK-B"],
    "Healthcare & Biotech": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "AMGN", "ISRG"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL"],
    "Energy & Industrials": ["XOM", "CVX", "COP", "CAT", "UNP", "HON", "UPS", "GE", "DE"],
    # Tel Aviv listings. Kept for research; excluded from the US model because
    # they trade on a different calendar and currency.
    "Israel Banks & Financials": ["LUMI.TA", "POLI.TA", "DSCT.TA", "MZTF.TA", "FIBI.TA"],
    "Israel Tech & Software": [
        "NICE.TA",
        "NVMI.TA",
        "CAMT.TA",
        "MTRX.TA",
        "HLAN.TA",
        "SPNS.TA",
        "AUDC.TA",
        "MAGIC.TA",
    ],
    "Israel Real Estate": ["AZRG.TA", "MLSR.TA", "MVNE.TA", "AMOT.TA", "SKBN.TA", "DANE.TA"],
    "Israel Energy & Green": ["DLEKG.TA", "ENLT.TA", "ENRG.TA", "ORA.TA"],
    "Israel Defense, Pharma & Industry": ["ESLT.TA", "TEVA.TA", "ICL.TA", "SPEN.TA", "ELEC.TA", "STRS.TA"],
}

# Market benchmark used for relative-strength features.
BENCHMARK = "SPY"
# Offline CSV snapshots may only have the index, not the ETF.
BENCHMARK_FALLBACKS = ("SPY", "^GSPC")

MACRO_SYMBOLS = ("SPY", "QQQ", "IWM")

_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")
_SHARE_CLASS_RE = re.compile(r"^([A-Z]{1,5})[.\-]([A-Z])$")


class InvalidTicker(ValueError):
    pass


def to_yahoo(ticker: str) -> str:
    """'brk.b' -> 'BRK-B'. Exchange suffixes like '.TA' are left alone."""
    t = (ticker or "").strip().upper()
    if not _TICKER_RE.match(t):
        raise InvalidTicker(f"Invalid ticker: {ticker!r}")
    m = _SHARE_CLASS_RE.match(t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return t


def to_display(ticker: str) -> str:
    """'BRK-B' -> 'BRK.B' (how the UI shows share classes)."""
    t = (ticker or "").strip().upper()
    m = _SHARE_CLASS_RE.match(t)
    return f"{m.group(1)}.{m.group(2)}" if m else t


def dashboard_tickers() -> list[str]:
    return [s.ticker for s in DASHBOARD]


def dashboard_stock(ticker: str) -> Stock | None:
    display = to_display(ticker)
    return next((s for s in DASHBOARD if s.ticker == display), None)


def training_universe() -> list[str]:
    """US research universe in Yahoo format, de-duplicated, order preserved."""
    seen: dict[str, None] = {}
    for sector, tickers in ASSET_MAP.items():
        if sector.startswith("Israel"):
            continue
        for t in tickers:
            seen.setdefault(to_yahoo(t), None)
    for s in DASHBOARD:
        seen.setdefault(to_yahoo(s.ticker), None)
    return list(seen)
