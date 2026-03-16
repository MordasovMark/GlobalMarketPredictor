from __future__ import annotations

"""
Central configuration for multi-asset, multi-horizon research.

This module defines asset universes and forecast horizons that can be
imported by ingestion, preprocessing, modeling, API, and dashboard
layers.
"""

from typing import Final, Dict, List


# ---------------------------------------------------------------------------
# Asset configuration – top S&P 100 stocks by distinct market sector
# ---------------------------------------------------------------------------

ASSET_MAP: Final[Dict[str, List[str]]] = {
    "Technology & Semiconductors": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "INTC",
        "CSCO", "QCOM", "TXN", "IBM", "ADBE", "AMAT", "INTU",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR",
    ],
    "Aerospace & Defense": [
        "RTX", "LMT", "BA", "GD", "NOC", "HWM", "TDG", "TXT", "LHX", "LDOS", "PLTR",
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "C", "AXP", "SPGI", "BLK", "SCHW",
    ],
    "Healthcare & Biotech": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "AMGN", "ISRG",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL",
    ],
    "Energy & Industrials": [
        "XOM", "CVX", "COP", "CAT", "UNP", "HON", "UPS", "GE", "DE",
    ],
    # -----------------------------------------------------------------------
    # Israeli market (TA-125) – appended; US sectors above unchanged
    # -----------------------------------------------------------------------
    "Israel Banks & Financials": [
        "LUMI.TA", "POLI.TA", "DSCT.TA", "MZTF.TA", "FIBI.TA",
    ],
    "Israel Tech & Software": [
        "NICE.TA", "NVMI.TA", "CAMT.TA", "MTRX.TA", "HLAN.TA",
        "SPNS.TA", "AUDC.TA", "MAGIC.TA",
    ],
    "Israel Real Estate": [
        "AZRG.TA", "MLSR.TA", "MVNE.TA", "AMOT.TA", "SKBN.TA", "DANE.TA",
    ],
    "Israel Energy & Green": [
        "NWMD.L.TA", "DLEKG.TA", "ENLT.TA", "ENRG.TA", "ORA.TA",
    ],
    "Israel Defense, Pharma & Industry": [
        "ESLT.TA", "TEVA.TA", "ICL.TA", "SPEN.TA", "ELEC.TA", "STRS.TA",
    ],
}

# Sector/category names returned by entity recognition map to ASSET_MAP keys.
SECTOR_MAP: Final[Dict[str, str]] = {
    "Banks": "Financials",
    "Tech": "Technology & Semiconductors",
    "Healthcare": "Healthcare & Biotech",
    "Consumer": "Consumer Discretionary",
    "Energy": "Energy & Industrials",
}


def get_tickers_for_sector(sector_or_category: str) -> List[str]:
    """
    Resolve a sector/category name to a list of tickers.

    If the name is an ASSET_MAP key (e.g. "TA-125 Banks"), return those tickers.
    Otherwise look up SECTOR_MAP (e.g. "Banks" -> "TA-125 Banks") and return
    that category's tickers. Return empty list if unknown.
    """
    sector = sector_or_category.strip()
    if sector in ASSET_MAP:
        return list(ASSET_MAP[sector])
    category = SECTOR_MAP.get(sector)
    if category and category in ASSET_MAP:
        return list(ASSET_MAP[category])
    return []


def get_all_tickers() -> List[str]:
    """
    Flatten the asset map into a single list of unique tickers.

    Returns
    -------
    list[str]
        All tickers across every category defined in ``ASSET_MAP``.
    """
    tickers: List[str] = []
    for symbols in ASSET_MAP.values():
        tickers.extend(symbols)
    # Preserve order while removing duplicates
    seen = set()
    unique: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# Horizon configuration
# ---------------------------------------------------------------------------

HORIZONS: Final[Dict[str, int]] = {
    "Tomorrow": 1,
    "Next Week": 7,
    "Next Month": 30,
}

# Backwards-compatible alias: same structure as ASSET_MAP (all categories, list of tickers).
ASSETS: Final[Dict[str, List[str]]] = dict(ASSET_MAP)

FORECAST_HORIZONS: Final[List[int]] = list(HORIZONS.values())

