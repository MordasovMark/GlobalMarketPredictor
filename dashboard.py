from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from config import ASSET_MAP


API_BASE_URL = "https://globalmarketpredictor.onrender.com"
RAW_PRICES_DIR = Path("data") / "raw_prices"
RAW_NEWS_DIR = Path("data") / "raw_news"
NEWS_LOOKBACK_DAYS = 7

API_OFFLINE_MESSAGE = (
    "âš ï¸ **AI Engine Offline:** Please start the backend server (`python main.py`) "
    "to view forecasts."
)


@st.cache_resource
def load_ai_models():
    """Load the 3 regression models (1D, 5D, 10D) from disk. Returns None if file missing or load fails."""
    try:
        return joblib.load("ai_regression_models.pkl")
    except Exception:
        return None


models = load_ai_models()


st.set_page_config(page_title="Global Market Predictor", layout="wide", initial_sidebar_state="collapsed")

# Global UI/UX: typography, spacing, stock-card hover, KPI cards (Dark Mode FinTech)
st.markdown(
    """
<style>
/* 1. Typography & Spacing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, p, label, span, div[data-testid="stVerticalBlock"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.block-container, .main .block-container {
  padding-top: 0.5rem !important;
  max-width: 100% !important;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
  visibility: hidden !important;
}

/* 2. Stock cards in main_col1 – micro-interactions */
.stock-card {
  transition: background-color 0.2s ease !important;
}
.stock-card:hover {
  cursor: pointer !important;
  background-color: #1e1e1e !important;
}

/* 3. KPI metric cards – sleek dark FinTech look */
[data-testid="stMetric"], .control-bar {
  background-color: #111111 !important;
  border: 1px solid #333333 !important;
  border-radius: 8px !important;
  padding: 15px !important;
}
.control-bar-item {
  padding: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

MINT_GREEN = "#00E676"
NEGATIVE_RED = "#FF3D00"
CARD_BG = "#1C1F26"
PAGE_BG = "#121418"
NEON_GREEN = "#00ff88"
TERMINAL_BG = "#0c0c0c"
TERMINAL_CARD = "#1a1a1a"

DOMAIN_MAP = {
    "AAPL": "apple.com", "MSFT": "microsoft.com", "NVDA": "nvidia.com",
    "AMZN": "amazon.com", "META": "meta.com", "GOOGL": "google.com",
    "BRK-B": "berkshirehathaway.com", "LLY": "lilly.com", "AVGO": "broadcom.com",
    "JPM": "jpmorganchase.com", "TSLA": "tesla.com", "WMT": "walmart.com",
    "UNH": "unitedhealthgroup.com", "V": "visa.com", "XOM": "exxonmobil.com",
    "MA": "mastercard.com", "PG": "pg.com", "JNJ": "jnj.com", "HD": "homedepot.com",
    "ORCL": "oracle.com", "COST": "costco.com", "MRK": "merck.com", "ABBV": "abbvie.com",
    "CRM": "salesforce.com", "BAC": "bankofamerica.com", "CVX": "chevron.com",
    "NFLX": "netflix.com", "AMD": "amd.com", "KO": "coca-colacompany.com", "PEP": "pepsico.com",
}
COMPANY_DISPLAY_NAMES: Dict[str, str] = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon.com",
    "META": "Meta Platforms", "GOOGL": "Alphabet", "BRK-B": "Berkshire Hathaway", "LLY": "Eli Lilly",
    "AVGO": "Broadcom", "JPM": "JPMorgan Chase", "TSLA": "Tesla", "WMT": "Walmart",
    "UNH": "UnitedHealth", "V": "Visa", "XOM": "Exxon Mobil", "MA": "Mastercard",
    "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "HD": "Home Depot", "ORCL": "Oracle",
    "COST": "Costco", "MRK": "Merck", "ABBV": "AbbVie", "CRM": "Salesforce", "BAC": "Bank of America",
    "CVX": "Chevron", "NFLX": "Netflix", "AMD": "AMD", "KO": "Coca-Cola", "PEP": "PepsiCo",
}
STOCK_LIST_COUNT = 30
TERMINAL_RANGE_OPTIONS = ["Intra", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"]

INDEX_TICKERS = [
    ("GOLD", "GC=F"),
    ("OIL WTI", "CL=F"),
    ("SILVER", "SI=F"),
    ("DOW 30", "^DJI"),
    ("S&P 500", "^GSPC"),
    ("NASDAQ", "^IXIC"),
    ("NAT GAS", "NG=F"),
    ("COPPER", "HG=F"),
    ("VIX", "^VIX"),
    ("BRENT", "BZ=F"),
]

TICKER_BAR_COUNT = 15

TOP_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
    "TSLA",
    "WMT",
    "UNH",
    "V",
    "XOM",
    "MA",
    "JNJ",
    "PG",
    "HD",
    "ORCL",
    "COST",
    "MRK",
    "ABBV",
    "CRM",
    "BAC",
    "CVX",
    "NFLX",
    "AMD",
    "KO",
    "PEP",
]


def inject_custom_css() -> None:
    """Financial Terminal UI: deep black-blue, glassmorphism cards, monospace prices, green/red glow."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* === GLOBAL: Edge-to-edge, Inter/Roboto === */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: #05070a !important;
  padding: 0 !important;
  margin: 0 !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
  visibility: hidden !important;
}
/* Remove default Streamlit padding so layout touches top/bottom */
.block-container, .main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
  padding-top: 50px !important;
}
[data-testid="stVerticalBlock"] > div { padding: 0 !important; }
[data-testid="stHorizontalBlock"] { padding: 0 !important; }
/* Left column (Stock List) full-height dark strip — terminal-style sidebar */
[data-testid="stAppViewContainer"] [data-testid="stHorizontalBlock"] > div:first-child {
  background: #0a0a0a !important;
  padding: 8px 6px 12px 8px !important;
  min-height: calc(100vh - 50px) !important;
}

/* === LEFT: Stock List (main_col1) — .stock-card compact, #111, active = 4px blue left === */
.stock-list-title {
  font-size: 0.8rem !important;
  font-weight: 700 !important;
  color: #ffffff !important;
  margin-bottom: 8px !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
.stock-card-wrap { margin-bottom: 4px !important; }
.stock-card {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 6px 10px !important;
  background: #111 !important;
  border-radius: 6px !important;
  border: 1px solid #222 !important;
  text-decoration: none !important;
  color: inherit !important;
  transition: background 0.12s ease !important;
}
.stock-card:hover { background: #1a1a1a !important; }
.stock-card.active {
  border-left: 4px solid #3b82f6 !important;
  padding-left: 6px !important;
}
.stock-card .card-logo-wrap {
  width: 28px !important;
  height: 28px !important;
  flex-shrink: 0 !important;
  border-radius: 4px !important;
  background: #0a0a0a !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: hidden !important;
}
.stock-card .card-logo-wrap img {
  width: 100% !important;
  height: 100% !important;
  object-fit: contain !important;
}
.stock-card .card-logo-fallback {
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  color: #9ca3af !important;
  display: none !important;
  width: 100% !important;
  height: 100% !important;
  align-items: center !important;
  justify-content: center !important;
  background: #1a1a1a !important;
  border-radius: 4px !important;
}
.stock-card .card-left { display: flex !important; flex-direction: column !important; min-width: 0 !important; flex: 1 !important; }
.stock-card .card-ticker { font-weight: 700 !important; font-size: 0.78rem !important; color: #E4E6EB !important; }
.stock-card .card-name { font-size: 0.65rem !important; color: #8E929C !important; margin-top: 0 !important; }
.stock-card .card-right { display: flex !important; flex-direction: column !important; align-items: flex-end !important; }
.stock-card .card-price { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; font-weight: 600 !important; color: #E4E6EB !important; }
.stock-card .card-change { font-size: 0.72rem !important; font-weight: 600 !important; }
.stock-card .card-change.positive { color: #00E676 !important; }
.stock-card .card-change.negative { color: #FF3D00 !important; }

/* === MARKET TICKER BAR: Fixed at top, scrolling marquee === */
.ticker-bar-fixed {
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  z-index: 999 !important;
  background: #05070a !important;
  padding: 10px 0 !important;
  border-bottom: 1px solid #1f2937 !important;
  overflow: hidden !important;
  font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
.ticker-marquee-wrap {
  overflow: hidden !important;
  width: 100% !important;
}
.ticker-marquee {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  width: max-content !important;
  animation: ticker-scroll 60s linear infinite !important;
}
.ticker-marquee:hover {
  animation-play-state: paused !important;
}
@keyframes ticker-scroll {
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
.ticker-item {
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
  padding: 0 16px !important;
  white-space: nowrap !important;
  flex-shrink: 0 !important;
  border-right: 1px solid #1f2937 !important;
}
.ticker-item .ticker-sym { font-size: 0.75rem !important; font-weight: 600 !important; color: #E4E6EB !important; }
.ticker-item .ticker-price { font-size: 0.75rem !important; color: #E4E6EB !important; }
.ticker-item .ticker-ch { font-size: 0.75rem !important; font-weight: 600 !important; }
.ticker-item .ticker-ch.up { color: #00E676 !important; text-shadow: 0 0 8px rgba(0,230,118,0.35) !important; }
.ticker-item .ticker-ch.down { color: #FF3D00 !important; text-shadow: 0 0 8px rgba(255,61,0,0.3) !important; }
.ticker-spark {
  display: inline-block !important;
  width: 36px !important;
  height: 14px !important;
  flex-shrink: 0 !important;
  vertical-align: middle !important;
}
.ticker-bar-fixed a.ticker-link {
  text-decoration: none !important;
  color: inherit !important;
}
.ticker-bar-fixed a.ticker-link:hover {
  color: inherit !important;
  opacity: 0.9 !important;
}

/* === TYPOGRAPHY: Monospace for prices / data-heavy feel === */
.stApp, .stMarkdown, p, label, span, div[data-testid="stVerticalBlock"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.price-mono, .kpi-value, .leader-price, .index-ticker-card .val, .stat-val {
  font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
h1, h2, h3, .card-title, .section-header {
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  color: #8E929C !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
  margin: 0 0 6px 0 !important;
}

/* === GLASSMORPHISM CARDS: darker bg, 1px border #1f2937, radius 12px === */
div[data-testid="stColumn"] {
  background: rgba(15, 18, 22, 0.85) !important;
  border-radius: 12px !important;
  padding: 12px 14px !important;
  border: 1px solid #1f2937 !important;
  margin: 0 6px !important;
}
.card-title { font-size: 0.7rem !important; color: #8E929C !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; margin: 0 0 8px 0 !important; }

/* === SIDEBAR: no padding so it touches top/bottom === */
[data-testid="stSidebar"] {
  background: #05070a !important;
  border-right: 1px solid #1f2937 !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  padding: 0 !important;
  padding-top: 50px !important;
}
[data-testid="stSidebar"] .stMarkdown { color: #E4E6EB !important; }
[data-testid="stSidebar"] h2 { font-size: 0.85rem !important; font-weight: 600 !important; color: #E4E6EB !important; text-transform: none !important; margin-bottom: 10px !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] .stMarkdown p { font-size: 0.8rem !important; color: #8E929C !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select {
  background: rgba(15, 18, 22, 0.9) !important;
  color: #E4E6EB !important;
  border: 1px solid #1f2937 !important;
  border-radius: 12px !important;
  font-size: 0.8rem !important;
}
[data-testid="stSidebar"] button {
  background: rgba(15, 18, 22, 0.9) !important;
  color: #00E676 !important;
  border: 1px solid #1f2937 !important;
  border-radius: 12px !important;
  font-size: 0.8rem !important;
}
[data-testid="stSidebar"] button:hover {
  background: rgba(0,230,118,0.08) !important;
  border-color: rgba(0,230,118,0.4) !important;
  box-shadow: 0 0 10px rgba(0,230,118,0.15) !important;
}
.sidebar-menu { list-style: none !important; padding: 0 !important; margin: 12px 0 0 0 !important; }
.sidebar-menu li {
  padding: 10px 12px !important; margin: 2px 0 !important; border-radius: 12px !important;
  color: #8E929C !important; font-size: 0.8rem !important; cursor: default !important;
}
.sidebar-menu li:hover { background: rgba(255,255,255,0.03) !important; color: #E4E6EB !important; }
.sidebar-menu li.active { background: rgba(0,230,118,0.1) !important; color: #00E676 !important; border: 1px solid rgba(0,230,118,0.25) !important; }

/* === METRICS: Glow based on performance === */
[data-testid="stMetric"] {
  background: rgba(15, 18, 22, 0.85) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  border: 1px solid #1f2937 !important;
}
[data-testid="stMetric"] label { font-size: 0.6rem !important; color: #8E929C !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  color: #E4E6EB !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"].positive { color: #00E676 !important; text-shadow: 0 0 10px rgba(0,230,118,0.35) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"].negative { color: #FF3D00 !important; text-shadow: 0 0 10px rgba(255,61,0,0.25) !important; }

/* === CONTROL BAR: Single horizontal row above chart === */
.control-bar {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 16px !important;
  flex-wrap: wrap !important;
  padding: 10px 14px !important;
  background: rgba(15, 18, 22, 0.85) !important;
  border: 1px solid #1f2937 !important;
  border-radius: 12px !important;
  margin-bottom: 10px !important;
}
.control-bar-item {
  flex: 1 1 auto !important;
  min-width: 80px !important;
}
.control-bar .kpi-label { font-size: 0.6rem !important; margin-bottom: 2px !important; }
.control-bar .kpi-value { font-size: 1.1rem !important; }

.control-bar-progress {
  height: 4px !important;
  background: #1f2937 !important;
  border-radius: 2px !important;
  margin-top: 4px !important;
  overflow: hidden !important;
}
.control-bar-progress-fill {
  height: 100% !important;
  background: #00E676 !important;
  border-radius: 2px !important;
  transition: width 0.2s ease !important;
}

/* === Backtest tab: single row of 4 KPI cards === */
.backtest-kpi-row {
  display: flex !important;
  flex-direction: row !important;
  gap: 12px !important;
  margin-bottom: 16px !important;
  flex-wrap: wrap !important;
}
.backtest-kpi-card {
  flex: 1 1 0 !important;
  min-width: 140px !important;
  background-color: #141414 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 8px !important;
  padding: 16px !important;
}
.backtest-kpi-label {
  font-size: 11px !important;
  color: #888 !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  margin-bottom: 6px !important;
}
.backtest-kpi-value {
  font-size: 24px !important;
  font-weight: 700 !important;
  color: #ffffff !important;
}
.backtest-kpi-value.positive { color: #00ff88 !important; }

/* === MARKET LEADERS ROW (Top 4 movers) === */
.leader-row {
  display: flex !important;
  flex-direction: row !important;
  align-items: stretch !important;
  gap: 10px !important;
  margin-bottom: 10px !important;
  flex-wrap: nowrap !important;
}
.leader-card {
  flex: 1 1 0 !important;
  min-width: 0 !important;
  background: rgba(15, 18, 22, 0.85) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  border: 1px solid #1f2937 !important;
}
.leader-sym { font-size: 0.65rem !important; color: #8E929C !important; text-transform: uppercase !important; }
.leader-price { font-family: 'JetBrains Mono', monospace !important; font-size: 0.95rem !important; font-weight: 600 !important; color: #fff !important; }
.leader-ch { font-size: 0.75rem !important; font-weight: 500 !important; }
.leader-spark, .index-ticker-card .index-spark { display: block !important; width: 100% !important; height: 20px !important; margin-top: 4px !important; }

/* === INDEX TICKER ROW (fallback) === */
.index-ticker-row { display: flex !important; flex-direction: row !important; align-items: stretch !important; gap: 10px !important; margin-bottom: 10px !important; flex-wrap: nowrap !important; }
.index-ticker-card {
  flex: 1 1 0 !important; min-width: 0 !important;
  background: rgba(15, 18, 22, 0.85) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  border: 1px solid #1f2937 !important;
}
.index-ticker-card .sym { font-size: 0.65rem !important; color: #8E929C !important; text-transform: uppercase !important; }
.index-ticker-card .val { font-family: 'JetBrains Mono', monospace !important; font-size: 0.95rem !important; font-weight: 600 !important; color: #fff !important; }
.index-ticker-card .ch { font-size: 0.75rem !important; font-weight: 500 !important; }

/* === KPI values: monospace + glow === */
.kpi-label { font-size: 0.6rem !important; color: #8E929C !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; margin-bottom: 2px !important; }
.kpi-value { font-family: 'JetBrains Mono', monospace !important; font-size: 1.15rem !important; font-weight: 700 !important; color: #E4E6EB !important; }
.kpi-value.green { color: #00E676 !important; text-shadow: 0 0 12px rgba(0,230,118,0.4) !important; }
.kpi-value.red { color: #FF3D00 !important; text-shadow: 0 0 12px rgba(255,61,0,0.3) !important; }

.stRadio > div { flex-direction: row !important; gap: 6px !important; }
.stRadio label { font-size: 0.75rem !important; color: #8E929C !important; padding: 6px 10px !important; border-radius: 8px !important; }
.stRadio label:hover { color: #E4E6EB !important; }
a { color: #00E676 !important; }
a:hover { color: #00E676 !important; opacity: 0.9 !important; }
.stDeployButton { display: none !important; }
hr { margin: 10px 0 !important; border-color: #1f2937 !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 14px !important; }

.stats-grid { display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 10px 18px; margin: 10px 0; }
.stat-item { display: flex; flex-direction: column; gap: 2px; }
.stat-label { font-size: 0.65rem; color: #8E929C; text-transform: uppercase; }
.stat-val { font-family: 'JetBrains Mono', monospace !important; font-size: 0.9rem; color: #E4E6EB; font-weight: 500; }
.top-value-list { display: flex; flex-direction: column; gap: 4px; }
.top-value-row { font-size: 0.8rem; color: #E4E6EB; }
.top-value-row .tv-sym { font-weight: 600; }
.top-value-row .tv-val { margin-right: 6px; }
.top-value-row .tv-vol { color: #8E929C; font-size: 0.75rem; }
.news-list { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
.news-item { font-size: 0.8rem; }
.news-item a { color: #00E676; text-decoration: none; }

/* Terminal chart container */
.terminal-chart-wrap { background: #0c0c0c !important; border-radius: 12px !important; padding: 12px !important; border: 1px solid #2a2a2a !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def _is_connection_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "10061" in msg or "connection refused" in msg or "connectionerror" in msg


@st.cache_data(ttl=300, show_spinner=False)
def get_sp100_sectors() -> pd.DataFrame:
    """
    Scrape S&P 100 constituents from Wikipedia.

    Returns a DataFrame with columns:
    - Symbol: original symbol as shown on Wikipedia (e.g. BRK.B)
    - Sector: GICS sector (e.g. Information Technology)
    - YF_Symbol: yfinance-compatible symbol (e.g. BRK-B)
    """
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    try:
        tables = pd.read_html(url)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["Symbol", "Sector", "YF_Symbol"])

    sp100 = None
    for t in tables:
        if "Symbol" in t.columns and "Sector" in t.columns:
            sp100 = t.copy()
            break

    # Fallback to the suggested table index if present
    if sp100 is None and len(tables) > 2:
        t = tables[2].copy()
        if "Symbol" in t.columns and "Sector" in t.columns:
            sp100 = t

    if sp100 is None:
        return pd.DataFrame(columns=["Symbol", "Sector", "YF_Symbol"])

    sp100 = sp100[["Symbol", "Sector"]].copy()
    sp100["Symbol"] = sp100["Symbol"].astype(str).str.strip()
    sp100["Sector"] = sp100["Sector"].astype(str).str.strip()
    sp100 = sp100.replace({"Symbol": {"nan": None}, "Sector": {"nan": None}})
    sp100 = sp100.dropna(subset=["Symbol", "Sector"]).reset_index(drop=True)

    # Normalize tickers for yfinance
    sp100["Symbol"] = sp100["Symbol"].str.split().str[0]
    sp100["YF_Symbol"] = sp100["Symbol"].str.replace(".", "-", regex=False)

    return sp100


@st.cache_data(ttl=300, show_spinner=False)
def load_index_tickers_data() -> list[tuple[str, str, float, float, list[float]]]:
    """Return list of (label, symbol, latest_value, change_pct, sparkline_y)."""
    out: list[tuple[str, str, float, float, list[float]]] = []
    for label, symbol in INDEX_TICKERS:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="5d", interval="1d")
            if hist is None or hist.empty:
                out.append((label, symbol, 0.0, 0.0, []))
                continue
            close = hist["Close"]
            latest = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else latest
            change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
            spark = close.astype(float).tolist()
            out.append((label, symbol, latest, change_pct, spark))
        except Exception:  # noqa: BLE001
            out.append((label, symbol, 0.0, 0.0, []))
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_top_movers(top_n: int = 4) -> list[tuple[str, float, float, float, list[float]]]:
    """Fetch daily % change for S&P 100 (TOP_TICKERS), return top N market leaders by |% change| with sparkline.
    Returns list of (symbol, latest_price, change_pct, volume, sparkline_y)."""
    out: list[tuple[str, float, float, float, list[float]]] = []
    try:
        data = yf.download(
            list(TOP_TICKERS),
            period="5d",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if data is None or data.empty:
            return _fallback_top_movers(top_n)
        # Single ticker: columns are Open, High, Low, Close, Volume
        if not isinstance(data.columns, pd.MultiIndex):
            sym = TOP_TICKERS[0] if TOP_TICKERS else ""
            close = data["Close"].dropna() if "Close" in data.columns else pd.Series(dtype=float)
            vol_ser = data["Volume"].dropna() if "Volume" in data.columns else pd.Series(dtype=float)
            if not close.empty and len(close) >= 2:
                latest = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
                vol = float(vol_ser.iloc[-1]) if not vol_ser.empty else 0.0
                out.append((sym, latest, change_pct, vol, close.astype(float).tolist()))
            return out[:top_n] if out else _fallback_top_movers(top_n)
        tickers = data.columns.get_level_values(0).unique().tolist()
        for sym in tickers:
            try:
                close = data[(sym, "Close")].dropna() if (sym, "Close") in data.columns else pd.Series(dtype=float)
                vol_ser = data[(sym, "Volume")].dropna() if (sym, "Volume") in data.columns else pd.Series(dtype=float)
                if close.empty or len(close) < 2:
                    continue
                latest = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
                vol = float(vol_ser.iloc[-1]) if not vol_ser.empty else 0.0
                spark = close.astype(float).tolist()
                out.append((sym, latest, change_pct, vol, spark))
            except Exception:  # noqa: BLE001
                continue
        # Sort by absolute % change (biggest movers first), then take top_n
        out.sort(key=lambda x: (abs(x[2]), x[3]), reverse=True)
        return out[:top_n]
    except Exception:  # noqa: BLE001
        return _fallback_top_movers(top_n)


def _fallback_top_movers(top_n: int) -> list[tuple[str, float, float, float, list[float]]]:
    """Fallback when batch download fails: use first N from TOP_TICKERS with single-ticker fetch."""
    out: list[tuple[str, float, float, float, list[float]]] = []
    for sym in TOP_TICKERS[: max(top_n, 8)]:
        if len(out) >= top_n:
            break
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d", interval="1d")
            if hist is None or hist.empty or len(hist) < 2:
                continue
            close = hist["Close"]
            latest = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
            vol = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0.0
            out.append((sym, latest, change_pct, vol, close.astype(float).tolist()))
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda x: (abs(x[2]), x[3]), reverse=True)
    return out[:top_n]


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker_bar_data(n: int = TICKER_BAR_COUNT) -> list[tuple[str, float, float, list[float]]]:
    """Fetch latest price and daily % change for first n stocks from S&P 100. Returns (symbol, price, change_pct, sparkline)."""
    symbols = TOP_TICKERS[: min(n, len(TOP_TICKERS))]
    out: list[tuple[str, float, float, list[float]]] = []
    try:
        data = yf.download(
            symbols,
            period="5d",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if data is None or data.empty:
            return _ticker_bar_fallback(n)
        if not isinstance(data.columns, pd.MultiIndex):
            sym = symbols[0]
            close = data["Close"].dropna() if "Close" in data.columns else pd.Series(dtype=float)
            if not close.empty and len(close) >= 2:
                latest = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
                out.append((sym, latest, change_pct, close.astype(float).tolist()))
            return out if out else _ticker_bar_fallback(n)
        for sym in data.columns.get_level_values(0).unique().tolist():
            try:
                close = data[(sym, "Close")].dropna() if (sym, "Close") in data.columns else pd.Series(dtype=float)
                if close.empty or len(close) < 2:
                    continue
                latest = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
                out.append((sym, latest, change_pct, close.astype(float).tolist()))
            except Exception:  # noqa: BLE001
                continue
        return out[: min(n, len(TOP_TICKERS))]
    except Exception:  # noqa: BLE001
        return _ticker_bar_fallback(n)


def _ticker_bar_fallback(n: int) -> list[tuple[str, float, float, list[float]]]:
    """Fallback: fetch first n tickers one by one."""
    out: list[tuple[str, float, float, list[float]]] = []
    take = min(n, len(TOP_TICKERS))
    for sym in TOP_TICKERS[:take]:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d", interval="1d")
            if hist is None or hist.empty or len(hist) < 2:
                continue
            close = hist["Close"]
            latest = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0
            out.append((sym, latest, change_pct, close.astype(float).tolist()))
        except Exception:  # noqa: BLE001
            continue
    return out[:take]


def _detect_close_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["Close", "close", "Adj Close", "adj_close", "Adj_Close", "Price", "price", "Unnamed: 1"]:
        if col in df.columns:
            return col
    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(ticker: str) -> pd.DataFrame:
    path = RAW_PRICES_DIR / f"{ticker}.csv"

    # First try local cache
    df: pd.DataFrame
    if path.is_file():
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # If no local data, fetch live from yfinance so the user never sees "no price history"
    if df.empty:
        RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Fetching market data..."):
            try:
                data = yf.download(ticker, period="4y", interval="1d", progress=False, auto_adjust=True, threads=False)
            except Exception:  # noqa: BLE001
                data = None
            if data is None or data.empty:
                return pd.DataFrame()
            data = data.reset_index()
            # Standardize to a CSV similar to collector output
            if "Date" not in data.columns:
                # yfinance often uses "Date" as index name; after reset_index it's a column
                date_col = data.columns[0]
                data = data.rename(columns={date_col: "Date"})
            try:
                data.to_csv(path, index=False)
            except Exception:  # noqa: BLE001
                # Failing to write cache is non-fatal
                pass
            df = data

    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_news_with_sentiment(ticker: str) -> pd.DataFrame:
    enriched_path = RAW_NEWS_DIR / f"{ticker}_news_with_sentiment.csv"
    raw_path = RAW_NEWS_DIR / f"{ticker}_news.csv"
    path = enriched_path if enriched_path.is_file() else raw_path
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    return df


def _news_last_7_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "published" not in df.columns:
        return df
    pub = pd.to_datetime(df["published"], errors="coerce")
    if pub.dt.tz is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)).replace(tzinfo=None)
        pub = pub.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        cutoff = datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)
    return df.loc[pub.notna() & (pub >= cutoff)].copy()


@st.cache_data(ttl=300, show_spinner=False)
def load_key_stats(ticker: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# Feature columns for ai_regression_models.pkl (must match train_model.py)
REGRESSION_FEATURE_COLS = ["SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "Volatility", "Daily_Return"]


def _build_regression_features(price_df: pd.DataFrame, close_col: str) -> Optional[pd.DataFrame]:
    """Build one row of features for ai_regression_models.pkl (SMA_20, SMA_50, RSI, MACD, MACD_Signal, Volatility, Daily_Return)."""
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


def _get_regression_forecasts(
    price_df: pd.DataFrame, close_col: str, models: Any
) -> Optional[tuple[float, float, float]]:
    """Run loaded ai_regression_models.pkl on price data. Returns (pred_1d_pct, pred_5d_pct, pred_10d_pct) or None."""
    if not models or not isinstance(models, dict):
        return None
    # Support both '1d' and '1D' style keys
    model_1d = models.get("1d") or models.get("1D")
    model_5d = models.get("5d") or models.get("5D")
    model_10d = models.get("10d") or models.get("10D")
    if not all(hasattr(m, "predict") for m in (model_1d, model_5d, model_10d) if m is not None):
        return None
    if model_1d is None or model_5d is None or model_10d is None:
        return None
    try:
        X = _build_regression_features(price_df, close_col)
        if X is None:
            return None
        feature_names = models.get("feature_names", REGRESSION_FEATURE_COLS)
        if feature_names:
            X = X[[c for c in feature_names if c in X.columns]]
        pred_1d = float(model_1d.predict(X)[0])
        pred_5d = float(model_5d.predict(X)[0])
        pred_10d = float(model_10d.predict(X)[0])
        return pred_1d, pred_5d, pred_10d
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_prediction_data(ticker: str) -> tuple[float, float]:
    """Fetch AI prediction from backend. Returns (strength_value, confidence_pct). Cached 5 min to avoid rate limits."""
    try:
        resp = requests.get(f"{API_BASE_URL}/predict_ticker/{ticker}", timeout=10)
        if not resp.ok:
            return 50.0, 50.0
        data = resp.json()
        pred_ret = data.get("predicted_return_pct")
        conf = data.get("confidence")
        confidence_pct = float(conf) * 100.0 if conf is not None else 50.0
        strength_value = 50.0
        if pred_ret is not None:
            strength_value = 50.0 + float(pred_ret) * 10.0
        elif conf is not None:
            strength_value = float(conf) * 100.0
        strength_value = max(0.0, min(100.0, strength_value))
        confidence_pct = max(0.0, min(100.0, confidence_pct))
        return strength_value, confidence_pct
    except Exception:  # noqa: BLE001
        return 50.0, 50.0


@st.cache_data(ttl=300, show_spinner=False)
def load_backtest_results() -> pd.DataFrame:
    """Load AI vs SPY backtest results for the Strategy Performance tab."""
    path = Path("data") / "backtest_results.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _annualized_return(cumulative: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return (float(cumulative) ** (252.0 / float(n_days))) - 1.0


def _max_drawdown(cumulative_series: pd.Series) -> float:
    if cumulative_series.empty:
        return 0.0
    peak = cumulative_series.cummax()
    dd = (peak - cumulative_series) / peak.replace(0, pd.NA)
    return float(dd.max())


def _fmt_number(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "â€”"
    if abs(v) >= 1e12:
        return f"{v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v:,.0f}"
    return f"{v:.2f}"


def _time_ago(ts: Any) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)) or pd.isna(ts):
        return ""
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.to_datetime(ts)
        except Exception:
            return ""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    t = ts
    try:
        if ts.tzinfo is not None:
            t = ts.tz_convert("UTC").tz_localize(None)
    except Exception:
        t = ts
    delta = now - t.to_pydatetime()
    mins = int(delta.total_seconds() // 60)
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


@st.cache_data(ttl=300, show_spinner=False)
def _empty_intraday_df() -> pd.DataFrame:
    """Return empty DataFrame with expected intraday columns."""
    return pd.DataFrame(columns=["date", "Open", "High", "Low", "Close", "Volume"])


def _normalize_intraday_data(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Flatten MultiIndex, normalize date/Close columns. Returns None if required columns missing."""
    if data is None or data.empty:
        return None
    # MultiIndex flattening
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    else:
        data = data.copy()
    data = data.reset_index()
    # Column normalization: datetime column -> "date"
    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "date"})
    elif "Date" in data.columns:
        data = data.rename(columns={"Date": "date"})
    if "date" not in data.columns:
        return None
    # Safe Close: prefer "Close", fallback "close"
    if "Close" not in data.columns and "close" in data.columns:
        data = data.rename(columns={"close": "Close"})
    if "Close" not in data.columns:
        return None
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


@st.cache_data(ttl=300, show_spinner=False)
def load_intraday_1d(ticker: str, interval: str = "1m") -> pd.DataFrame:
    """Fetch intraday data: period=5d, interval=1m, drop NaNs, keep only latest calendar day so 1D always has a line."""
    try:
        data = yf.download(ticker, period="5d", interval=interval, progress=False, group_by="ticker", auto_adjust=True, threads=False)
    except Exception:  # noqa: BLE001
        return _empty_intraday_df()
    # Empty data check: return empty DataFrame with expected columns
    if data is None or data.empty:
        return _empty_intraday_df()
    data = _normalize_intraday_data(data)
    if data is None:
        return _empty_intraday_df()
    data = data.dropna(subset=["date", "Close"], how="any").sort_values("date").reset_index(drop=True)
    if data.empty:
        return _empty_intraday_df()
    # Keep only the most recent calendar day so 1D view always has intraday points
    latest_date = data["date"].max().normalize()
    data = data[data["date"].dt.normalize() == latest_date].copy().reset_index(drop=True)
    if data.empty or len(data) < 2:
        # Fallback: 1d / 5m when 5d/1m returns nothing or single day has too few points
        try:
            fallback = yf.download(ticker, period="1d", interval="5m", progress=False, group_by="ticker", auto_adjust=True, threads=False)
            if fallback is None or fallback.empty:
                return data if len(data) >= 2 else _empty_intraday_df()
            fallback = _normalize_intraday_data(fallback)
            if fallback is not None and "date" in fallback.columns and "Close" in fallback.columns:
                fallback = fallback.dropna(subset=["date", "Close"], how="any").sort_values("date").reset_index(drop=True)
                if len(fallback) >= 2:
                    return fallback
        except Exception:  # noqa: BLE001
            pass
        return data if len(data) >= 2 else _empty_intraday_df()
    return data


def _slice_by_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty or range_key == "1D":
        return df
    df = df.sort_values("date").reset_index(drop=True)
    end = df["date"].iloc[-1]
    if range_key == "5D" or range_key == "1W":
        start = end - timedelta(days=7)
    elif range_key == "1M":
        start = end - timedelta(days=31)
    elif range_key == "3M":
        start = end - timedelta(days=93)
    elif range_key == "6M":
        start = end - timedelta(days=183)
    elif range_key == "YTD":
        start = datetime(end.year, 1, 1)
    elif range_key == "1Y":
        start = end - timedelta(days=366)
    elif range_key == "5Y":
        start = end - timedelta(days=366 * 5)
    elif range_key == "ALL":
        return df
    else:  # Max
        return df
    return df[df["date"] >= start].copy()


CHART_RANGE_OPTIONS = ["1D", "1W", "1M", "1Y", "ALL"]
SMA_FAST, SMA_SLOW = 20, 50
SMA20_COLOR = "rgba(100, 180, 255, 0.95)"
SMA50_COLOR = "rgba(255, 193, 7, 0.95)"


def _build_candlestick_chart(
    ticker: str,
    price_df: pd.DataFrame,
    range_key: str,
    close_col: str,
    terminal_style: bool = False,
    prev_close: Optional[float] = None,
) -> Optional[go.Figure]:
    """Candlestick + SMA(20) light blue, SMA(50) gold, Volume bar subplot. Optional terminal theme and prev-close line."""
    df = price_df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if close_col != "Close" and close_col in df.columns:
        df["Close"] = df[close_col]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    range_slice = "5D" if range_key == "1W" else ("Max" if range_key == "ALL" else range_key)
    plot_df = _slice_by_range(df, range_slice)
    if plot_df.empty or len(plot_df) < 2:
        return None
    need_ohlc = "Open" not in plot_df.columns or "High" not in plot_df.columns or "Low" not in plot_df.columns
    if need_ohlc:
        try:
            start = plot_df["date"].min()
            end = plot_df["date"].max() + pd.Timedelta(days=1)
            ohlc = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if ohlc is not None and not ohlc.empty:
                ohlc = ohlc.reset_index()
                date_col = "Date" if "Date" in ohlc.columns else ohlc.columns[0]
                ohlc = ohlc.rename(columns={date_col: "date"})
                ohlc["date"] = pd.to_datetime(ohlc["date"], errors="coerce")
                cols = [c for c in ["Open", "High", "Low", "Volume"] if c in ohlc.columns]
                if cols:
                    plot_df = plot_df.drop(columns=[c for c in cols if c in plot_df.columns], errors="ignore")
                    plot_df = plot_df.merge(ohlc[["date"] + cols], on="date", how="left")
            if "Open" not in plot_df.columns:
                plot_df["Open"] = plot_df["Close"]
            if "High" not in plot_df.columns:
                plot_df["High"] = plot_df[["Open", "Close"]].max(axis=1)
            if "Low" not in plot_df.columns:
                plot_df["Low"] = plot_df[["Open", "Close"]].min(axis=1)
        except Exception:  # noqa: BLE001
            plot_df["Open"] = plot_df["Close"].copy()
            plot_df["High"] = plot_df["Close"].copy()
            plot_df["Low"] = plot_df["Close"].copy()
    has_vol = "Volume" in plot_df.columns and plot_df["Volume"].notna().any()
    plot_df = plot_df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("date").reset_index(drop=True)
    if len(plot_df) < 2:
        return None
    plot_df["SMA20"] = plot_df["Close"].rolling(20, min_periods=1).mean()
    plot_df["SMA50"] = plot_df["Close"].rolling(50, min_periods=1).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    fig.add_trace(
        go.Candlestick(
            x=plot_df["date"],
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Price",
            increasing_line_color=MINT_GREEN,
            decreasing_line_color=NEGATIVE_RED,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df["date"], y=plot_df["SMA20"], mode="lines", name="SMA(20)", line=dict(color=SMA20_COLOR, width=1.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=plot_df["date"], y=plot_df["SMA50"], mode="lines", name="SMA(50)", line=dict(color=SMA50_COLOR, width=1.5)),
        row=1,
        col=1,
    )
    if has_vol:
        vol = plot_df["Volume"].fillna(0)
        colors = [MINT_GREEN if plot_df["Close"].iloc[i] >= plot_df["Open"].iloc[i] else NEGATIVE_RED for i in range(len(plot_df))]
        fig.add_trace(
            go.Bar(x=plot_df["date"], y=vol, marker_color=colors, opacity=0.7, showlegend=False, hoverinfo="skip"),
            row=2,
            col=1,
        )
    paper = TERMINAL_BG if terminal_style else "rgba(0,0,0,0)"
    plot_bg = TERMINAL_CARD if terminal_style else "rgba(0,0,0,0)"
    layout_kw: Dict[str, Any] = dict(
        template="plotly_dark",
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
        margin=dict(l=0, r=0, t=24, b=0),
        xaxis_rangeslider_visible=False,
    )
    if terminal_style:
        layout_kw["hoverlabel"] = dict(bgcolor="#1a1a1a", font=dict(color="#E4E6EB", size=12), bordercolor="#2a2a2a")
    fig.update_layout(**layout_kw)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, row=2, col=1)
    if terminal_style and prev_close is not None:
        fig.add_hline(y=prev_close, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1)
    return fig


def _build_price_chart(
    ticker: str,
    price_df: pd.DataFrame,
    range_key: str,
    close_col: str,
    terminal_style: bool = False,
    prev_close: Optional[float] = None,
) -> Optional[go.Figure]:
    """Clean line chart: no fill, dynamic color, dotted ref line, transparent layout."""
    df = price_df.copy()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["date", close_col]).sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        return None
    is_1d = range_key == "1D"
    if is_1d:
        plot_df = load_intraday_1d(ticker)
        if plot_df.empty or "Close" not in plot_df.columns:
            plot_df = _slice_by_range(df, "1W")
            close_col_use = close_col
        else:
            close_col_use = "Close"
    else:
        range_slice = "5D" if range_key == "1W" else ("Max" if range_key == "ALL" else range_key)
        plot_df = _slice_by_range(df, range_slice)
        close_col_use = close_col
    if plot_df.empty:
        return None
    data = plot_df.set_index("date").copy()
    if close_col_use not in data.columns:
        return None
    data["Close"] = data[close_col_use]

    # 1. Clean Line Chart (no fill, no px.area)
    start_price = data["Close"].iloc[0]
    end_price = data["Close"].iloc[-1]
    line_color = "#00ff88" if end_price >= start_price else "#ff5000"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            line=dict(color=line_color, width=2),
        )
    )
    # 2. Add horizontal reference line
    fig.add_hline(y=start_price, line_dash="dot", line_color="#555555", line_width=1)
    # Conditional X-axis tick format: Robinhood-style 09:30, 10:00 for intraday
    x_format = "%H:%M" if range_key == "1D" else None
    xaxis_kw: Dict[str, Any] = dict(
        showgrid=False,
        visible=True,
        showline=False,
        tickfont=dict(color="#888888", size=11),
        nticks=12,
    )
    if x_format is not None:
        xaxis_kw["tickformat"] = x_format
    # 3. Transparent, gridless layout
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=xaxis_kw,
        yaxis=dict(showgrid=False, side="right", tickfont=dict(color="white")),
        margin=dict(l=0, r=40, t=10, b=10),
        showlegend=False,
    )
    return fig


def _render_kpi_header(
    ticker: str,
    current_price: float,
    change_pct: float,
    pred_1d_pct: Optional[float],
    pred_5d_pct: Optional[float],
    pred_10d_pct: Optional[float],
) -> None:
    """Single horizontal Control Bar: Price, 24h Change %, AI 1D %, AI 5D %, AI 10D % (exact % forecasts)."""
    change_cls = "green" if change_pct >= 0 else "red"

    def _fmt_pred(p: Optional[float]) -> str:
        if p is None:
            return "—"
        return f"{p:+.2f}%"

    def _pred_cls(p: Optional[float]) -> str:
        if p is None:
            return ""
        return "green" if p >= 0 else "red"

    html = (
        '<div class="control-bar">'
        '<div class="control-bar-item">'
        '<div class="kpi-label">Current Price</div>'
        f'<div class="kpi-value price-mono">{current_price:,.2f}</div>'
        "</div>"
        '<div class="control-bar-item">'
        '<div class="kpi-label">24h Change %</div>'
        f'<div class="kpi-value {change_cls}">{change_pct:+.2f}%</div>'
        "</div>"
        '<div class="control-bar-item">'
        '<div class="kpi-label">AI 1D %</div>'
        f'<div class="kpi-value {_pred_cls(pred_1d_pct)}">{_fmt_pred(pred_1d_pct)}</div>'
        "</div>"
        '<div class="control-bar-item">'
        '<div class="kpi-label">AI 5D %</div>'
        f'<div class="kpi-value {_pred_cls(pred_5d_pct)}">{_fmt_pred(pred_5d_pct)}</div>'
        "</div>"
        '<div class="control-bar-item">'
        '<div class="kpi-label">AI 10D %</div>'
        f'<div class="kpi-value {_pred_cls(pred_10d_pct)}">{_fmt_pred(pred_10d_pct)}</div>'
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _get_stock_list_data(n: int = 5) -> list[tuple[str, float, float, str]]:
    """Return list of (symbol, price, change_pct, company_name) for left column. Uses TOP_TICKERS order, up to n items."""
    raw = load_ticker_bar_data(n=max(n, TICKER_BAR_COUNT))
    by_sym = {s: (p, ch) for s, p, ch, _ in raw}
    out: list[tuple[str, float, float, str]] = []
    for sym in TOP_TICKERS[:n]:
        name = COMPANY_DISPLAY_NAMES.get(sym, sym)
        if sym in by_sym:
            price, change_pct = by_sym[sym]
            out.append((sym, price, change_pct, name))
        else:
            out.append((sym, 0.0, 0.0, name))
    return out


def _render_stock_list_left_column(selected_ticker: str, num_stocks: int = 5) -> None:
    """Left column: clickable .stock-card items (title and selector are rendered by caller)."""
    data = _get_stock_list_data(n=num_stocks)
    for symbol, price, change_pct, company_name in data:
        # Google Favicon API: reliable, not blocked by ad-blockers or CORS
        domain = DOMAIN_MAP.get(symbol, f"{symbol.lower()}.com")
        img_url = f"https://s2.googleusercontent.com/s2/favicons?domain={domain}&sz=128"
        active = "active" if symbol == selected_ticker else ""
        ch_class = "positive" if change_pct >= 0 else "negative"
        change_str = f"{change_pct:+.2f}%"
        logo_html = (
            f'<span class="card-logo-wrap">'
            f'<img src="{img_url}" style="width: 32px; height: 32px; border-radius: 50%; object-fit: contain; background-color: white; padding: 2px;" alt="">'
            "</span>"
        )
        card = (
            f'<a href="/?ticker={symbol}" target="_self" class="stock-card {active}">'
            f'{logo_html}'
            '<div class="card-left">'
            f'<span class="card-ticker">{symbol}</span>'
            f'<span class="card-name">{company_name}</span>'
            "</div>"
            '<div class="card-right">'
            f'<span class="card-price">{price:,.2f}</span>'
            f'<span class="card-change {ch_class}">{change_str}</span>'
            "</div>"
            "</a>"
        )
        st.markdown(f'<div class="stock-card-wrap">{card}</div>', unsafe_allow_html=True)


def _render_header(ticker: str, latest: float, change_pct: float, sector: Optional[str] = None) -> None:
    change_color = MINT_GREEN if change_pct >= 0 else NEGATIVE_RED
    arrow = "&#9650;" if change_pct >= 0 else "&#9660;"
    sector_html = (
        f'<div style="margin-top:0.15rem; display:inline-flex; align-self:flex-start; padding:0.22rem 0.55rem; border-radius:999px; background: rgba(255,255,255,0.06); color:#8E929C; font-size:0.72rem;">{sector}</div>'
        if sector
        else ""
    )
    st.markdown(
        f"""
<div style="display:flex; flex-direction:column; gap:0.25rem;">
  <div style="font-size: 2.4rem; font-weight: 700; line-height: 1.1;">{ticker}</div>
  <div style="display:flex; align-items:baseline; gap:0.85rem;">
    <div style="font-size: 1.7rem; font-weight: 650;">{latest:,.2f}</div>
    <div style="font-size: 1.05rem; font-weight: 650; color: {change_color};">{arrow} {change_pct:+.2f}%</div>
  </div>
  {sector_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _fig_strength_meter(value: float, title: str = "Strength") -> go.Figure:
    """Plotly gauge: 0-33 blue, 33-66 purple, 66-100 mint green; needle at value."""
    value = max(0.0, min(100.0, float(value)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 28}, "suffix": ""},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "rgba(0,0,0,0)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(100, 180, 255, 0.8)"},
                    {"range": [33, 66], "color": "rgba(180, 120, 255, 0.8)"},
                    {"range": [66, 100], "color": "rgba(0, 230, 118, 0.6)"},
                ],
                "threshold": {
                    "line": {"color": "#fff", "width": 3},
                    "thickness": 0.9,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=220,
        font={"color": "#e0e0e0"},
    )
    return fig


def _fig_todays_value(sector_values: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bar chart: sector name vs value (mint/light bars)."""
    if not sector_values:
        sector_values = [(k, 1000.0 * (i + 1)) for i, k in enumerate(list(ASSET_MAP.keys())[:8])]
    labels = [s[0][:20] for s in sector_values]
    values = [s[1] for s in sector_values]
    total = sum(values) or 1
    pcts = [v / total * 100 for v in values]
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=MINT_GREEN,
            marker_line_color="rgba(255,255,255,0.2)",
            marker_line_width=0.5,
            text=[f"{v:,.0f} ({p:.1f}%)" for v, p in zip(values, pcts)],
            textposition="outside",
            textfont={"color": "#e0e0e0", "size": 11},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=320,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, showticklabels=True),
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
    )
    return fig


def render_google_finance_style(ticker: str, price_df: pd.DataFrame, news_df: pd.DataFrame, sector_hint: Optional[str] = None) -> None:
    close_col = _detect_close_column(price_df)
    if close_col is None or price_df.empty:
        st.error("No price history found for this ticker. Please check your network connection and try again.")
        return

    df = price_df.copy()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["date", close_col]).sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        st.error("Not enough price history to compute daily change.")
        return

    latest = float(df[close_col].iloc[-1])
    prev = float(df[close_col].iloc[-2])
    change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0

    _render_header(ticker, latest, change_pct, sector=sector_hint)

    # Timeline buttons above chart
    range_key = st.radio(
        "Range",
        options=["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "Max"],
        horizontal=True,
        label_visibility="collapsed",
    )

    is_1d = range_key == "1D"
    use_range = range_key
    if is_1d:
        plot_df = load_intraday_1d(ticker)
        if plot_df.empty or "Close" not in plot_df.columns:
            st.caption("Intraday data not available for 1D. Try another range or ticker.")
            plot_df = _slice_by_range(df, "5D")
            use_range = "5D"
            is_1d = False
    if not is_1d:
        plot_df = _slice_by_range(df, use_range)
        close_col_use = close_col
        has_volume = "Volume" in plot_df.columns and plot_df["Volume"].notna().any()
    else:
        close_col_use = "Close"
        has_volume = "Volume" in plot_df.columns and plot_df["Volume"].notna().any()

    if plot_df.empty:
        st.caption("No data for this range.")
    else:
        if is_1d:
            open_price = float(plot_df["Open"].iloc[0]) if "Open" in plot_df.columns else float(plot_df[close_col_use].iloc[0])
            last_close = float(plot_df[close_col_use].iloc[-1])
            is_green = last_close >= open_price
            line_color = MINT_GREEN if is_green else NEGATIVE_RED
            fill_color = "rgba(0,230,118,0.25)" if is_green else "rgba(255,61,0,0.25)"
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_df["date"],
                    y=[open_price] * len(plot_df),
                    mode="lines",
                    line=dict(width=0),
                    fill="tozeroy",
                    fillcolor="rgba(128,128,128,0)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_df["date"],
                    y=plot_df[close_col_use],
                    mode="lines",
                    line=dict(color=line_color, width=2),
                    fill="tonexty",
                    fillcolor=fill_color,
                    showlegend=False,
                    hovertemplate="%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            if has_volume:
                fig.add_trace(
                    go.Bar(
                        x=plot_df["date"],
                        y=plot_df["Volume"].fillna(0),
                        marker_color=line_color,
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )
            fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_rangeslider_visible=False,
            )
            fig.update_xaxes(type="date", tickformat="%H:%M", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=1, col=1)
            fig.update_xaxes(type="date", tickformat="%H:%M", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=2, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=1, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=2, col=1)
        else:
            if has_volume:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.8, 0.2],
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["date"],
                        y=plot_df[close_col_use],
                        mode="lines",
                        line=dict(color=MINT_GREEN, width=2),
                        fill="tozeroy",
                        fillcolor="rgba(0,230,118,0.18)",
                        showlegend=False,
                        hovertemplate="%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Bar(
                        x=plot_df["date"],
                        y=plot_df["Volume"].fillna(0),
                        marker_color="#6b7b8c",
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )
                fig.update_xaxes(type="date", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=1, col=1)
                fig.update_xaxes(type="date", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=2, col=1)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=1, col=1)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False, row=2, col=1)
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["date"],
                        y=plot_df[close_col_use],
                        mode="lines",
                        line=dict(color=MINT_GREEN, width=2),
                        fill="tozeroy",
                        fillcolor="rgba(0,230,118,0.18)",
                        showlegend=False,
                        hovertemplate="%{y:.2f}<extra></extra>",
                    )
                )
                fig.update_xaxes(type="date", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False)
                fig.update_layout(
                    template="plotly_dark",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_rangeslider_visible=False,
                )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})



def _spark_to_svg_path(spark: list[float], width: int = 100, height: int = 20) -> str:
    """Build SVG path for a sparkline (floating line, no axes)."""
    if not spark or len(spark) < 2:
        return ""
    mi, ma = min(spark), max(spark)
    span = ma - mi if ma != mi else 1
    n = len(spark)
    points = []
    for i, v in enumerate(spark):
        x = (i / (n - 1)) * width if n > 1 else 0
        y = height - (v - mi) / span * (height - 2) - 1
        points.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(points)


def _render_ticker_bar() -> None:
    """Fixed top bar with infinite scrolling marquee: Symbol, Price, Arrow, Change %, tiny sparkline."""
    data = load_ticker_bar_data(n=TICKER_BAR_COUNT)
    if not data:
        return
    items_html = []
    for symbol, price, change_pct, spark in data:
        color = MINT_GREEN if change_pct >= 0 else NEGATIVE_RED
        arrow = "&#9650;" if change_pct >= 0 else "&#9660;"
        ch_class = "up" if change_pct >= 0 else "down"
        change_str = f"{change_pct:+.2f}%"
        svg_path = _spark_to_svg_path(spark, width=36, height=14)
        spark_svg = (
            f'<svg class="ticker-spark" viewBox="0 0 36 14" preserveAspectRatio="none"><path fill="none" stroke="{color}" stroke-width="1" d="{svg_path}"/></svg>'
            if svg_path
            else ""
        )
        item_inner = (
            f'<span class="ticker-item">'
            f'<span class="ticker-sym">{symbol}</span>'
            f'<span class="ticker-price">{price:,.2f}</span>'
            f'<span class="ticker-ch {ch_class}">{arrow} {change_str}</span>'
            f'{spark_svg}'
            f"</span>"
        )
        items_html.append(
            f'<a href="/?ticker={symbol}" target="_self" class="ticker-link">{item_inner}</a>'
        )
    # Duplicate content for seamless loop (marquee translates -50%)
    content = "".join(items_html) + "".join(items_html)
    html = (
        '<div class="ticker-bar-fixed">'
        '<div class="ticker-marquee-wrap">'
        f'<div class="ticker-marquee">{content}</div>'
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_index_row() -> None:
    data = load_index_tickers_data()
    cards_html = []
    for label, symbol, value, change_pct, spark in data:
        color = MINT_GREEN if change_pct >= 0 else NEGATIVE_RED
        change_str = f"{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
        svg_path = _spark_to_svg_path(spark)
        spark_svg = (
            f'<svg class="index-spark" viewBox="0 0 100 20" preserveAspectRatio="none"><path fill="none" stroke="{color}" stroke-width="1.2" d="{svg_path}"/></svg>'
            if svg_path
            else ""
        )
        cards_html.append(
            f'<div class="index-ticker-card">'
            f'<div class="sym">{label}</div>'
            f'<div class="val">{value:,.2f}</div>'
            f'<span class="ch" style="color:{color}">{change_str}</span>'
            f'{spark_svg}'
            f"</div>"
        )
    html = f'<div class="index-ticker-row">{chr(10).join(cards_html)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def _render_market_leaders_row() -> None:
    """Top 4 Market Leaders (by |% change|) with mini sparkline per card."""
    data = load_top_movers(top_n=4)
    if not data:
        _render_index_row()
        return
    cards_html = []
    for symbol, price, change_pct, _vol, spark in data:
        color = MINT_GREEN if change_pct >= 0 else NEGATIVE_RED
        change_str = f"{change_pct:+.2f}%"
        svg_path = _spark_to_svg_path(spark)
        spark_svg = (
            f'<svg class="index-spark leader-spark" viewBox="0 0 100 20" preserveAspectRatio="none"><path fill="none" stroke="{color}" stroke-width="1.2" d="{svg_path}"/></svg>'
            if svg_path
            else ""
        )
        cards_html.append(
            f'<div class="leader-card">'
            f'<div class="leader-sym">{symbol}</div>'
            f'<div class="leader-price">{price:,.2f}</div>'
            f'<span class="leader-ch" style="color:{color}">{change_str}</span>'
            f'{spark_svg}'
            f"</div>"
        )
    st.markdown(
        '<p class="card-title">Market Leaders (Top Movers)</p>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="leader-row">{chr(10).join(cards_html)}</div>', unsafe_allow_html=True)


def main() -> None:
    inject_custom_css()
    _render_ticker_bar()

    # Query param: ?ticker=SYMBOL sets initial sidebar selection
    ticker_from_url = st.query_params.get("ticker")
    if isinstance(ticker_from_url, list):
        ticker_from_url = ticker_from_url[0] if ticker_from_url else None
    default_index = 0
    if ticker_from_url and isinstance(ticker_from_url, str):
        ticker_upper = ticker_from_url.strip().upper()
        if ticker_upper in TOP_TICKERS:
            default_index = TOP_TICKERS.index(ticker_upper)

    st.sidebar.markdown("## Market insight")
    st.sidebar.markdown(
        '<ul class="sidebar-menu"><li class="active">Dashboard</li><li>Market update</li><li>Income estimator</li><li>Interactive chart</li><li>Mutual funds</li><li>Portfolio</li><li>Settings</li><li>History</li><li>News</li><li>Feedback</li></ul>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Select a Stock from S&P 100**")
    selected_ticker = st.sidebar.selectbox(
        "Select a Stock from S&P 100",
        TOP_TICKERS,
        index=default_index,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("**Watchlist**")
    sp100 = get_sp100_sectors()
    sectors = sorted(sp100["Sector"].dropna().unique().tolist()) if not sp100.empty else []
    selected_sector = st.sidebar.selectbox(
        "Sector",
        sectors if sectors else ["(unavailable)"],
        label_visibility="collapsed",
    )
    sector_df = sp100[sp100["Sector"] == selected_sector].copy() if (not sp100.empty and selected_sector in sectors) else pd.DataFrame(columns=sp100.columns)
    symbols_display = sector_df["Symbol"].tolist() if not sector_df.empty else []
    st.sidebar.selectbox(
        "Stock",
        symbols_display if symbols_display else ["(unavailable)"],
        label_visibility="collapsed",
    )
    ticker = selected_ticker

    price_df = load_price_data(ticker)
    news_df = load_news_with_sentiment(ticker)
    close_col = _detect_close_column(price_df)
    latest = 0.0
    change_pct = 0.0
    if close_col and not price_df.empty:
        price_df[close_col] = pd.to_numeric(price_df[close_col], errors="coerce")
        price_df = price_df.dropna(subset=["date", close_col]).sort_values("date").reset_index(drop=True)
        if len(price_df) >= 2:
            latest = float(price_df[close_col].iloc[-1])
            prev = float(price_df[close_col].iloc[-2])
            change_pct = ((latest - prev) / prev * 100.0) if prev else 0.0

    # Use loaded regression models for 1D/5D/10D % forecasts when available
    pred_1d, pred_5d, pred_10d = None, None, None
    if models is not None and close_col and not price_df.empty:
        forecasts = _get_regression_forecasts(price_df, close_col, models)
        if forecasts is not None:
            pred_1d, pred_5d, pred_10d = forecasts

    # Forced split: main_col1 = Stock List, main_col2 = Terminal (KPI + time range + chart + tabs)
    main_col1, main_col2 = st.columns([1, 4])
    with main_col1:
        st.markdown('<p class="stock-list-title">Largest US Stocks</p>', unsafe_allow_html=True)
        num_stocks_to_show = st.selectbox(
            "Show Top:",
            options=[5, 10, 15, 20],
            index=0,
            label_visibility="collapsed",
            key="num_stocks_selector",
        )
        _render_stock_list_left_column(ticker, num_stocks=num_stocks_to_show)

    with main_col2:
        _render_kpi_header(ticker, latest, change_pct, pred_1d, pred_5d, pred_10d)

        # Time range and chart type: use pills/segmented_control if available, else radio
        _pills = getattr(st, "pills", None) or getattr(st, "segmented_control", None)
        if callable(_pills):
            terminal_range = _pills(
                "Time range",
                options=TERMINAL_RANGE_OPTIONS,
                default=TERMINAL_RANGE_OPTIONS[0],
                key="terminal_range",
                label_visibility="collapsed",
            )
            chart_type = _pills(
                "Chart type",
                options=["Line", "Candle"],
                default="Line",
                key="chart_type",
                label_visibility="collapsed",
            )
        else:
            terminal_range = st.radio(
                "Time range",
                options=TERMINAL_RANGE_OPTIONS,
                horizontal=True,
                label_visibility="collapsed",
                key="terminal_range",
            )
            chart_type = st.radio(
                "Chart type",
                options=["Line", "Candle"],
                horizontal=True,
                label_visibility="collapsed",
                key="chart_type",
            )
        range_key = "1D" if terminal_range == "Intra" else terminal_range
        prev_close = float(price_df[close_col].iloc[-2]) if (close_col and len(price_df) >= 2) else None

        if close_col and not price_df.empty:
            if chart_type == "Line":
                fig = _build_price_chart(ticker, price_df, range_key, close_col, terminal_style=True, prev_close=prev_close)
            else:
                fig = _build_candlestick_chart(ticker, price_df, range_key, close_col, terminal_style=True, prev_close=prev_close)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No data for this range.")
        else:
            st.info("No price history for this ticker. Check your connection or try another symbol.")

        tab_perf, tab_news = st.tabs(["📈 AI Strategy Performance", "📰 Market News"])

        with tab_perf:
            st.markdown('<p class="card-title">Strategy performance</p>', unsafe_allow_html=True)
            bt_df = load_backtest_results()
            if bt_df.empty:
                st.info("Please run backtester.py first to generate results.")
            elif "ai_cumulative" not in bt_df.columns or "spy_cumulative" not in bt_df.columns:
                st.warning("Backtest file format is not recognized. Please rerun backtester.py.")
            else:
                bt_df = bt_df.sort_values("date").reset_index(drop=True)
                n_days = len(bt_df)
                ai_cum_end = float(bt_df["ai_cumulative"].iloc[-1])
                spy_cum_end = float(bt_df["spy_cumulative"].iloc[-1])

                ai_cum_ret = ai_cum_end - 1.0
                spy_cum_ret = spy_cum_end - 1.0
                ai_ann = _annualized_return(ai_cum_end, n_days)
                spy_ann = _annualized_return(spy_cum_end, n_days)

                # Single row of 4 compact KPI cards above the chart
                ai_cum_pct = ai_cum_ret * 100.0
                ai_ann_pct = ai_ann * 100.0
                spy_cum_pct = spy_cum_ret * 100.0
                spy_ann_pct = spy_ann * 100.0
                def _kpi_val_cls(v: float) -> str:
                    return "positive" if v >= 0 else "neutral"
                kpi_cols = st.columns(4)
                with kpi_cols[0]:
                    st.markdown(
                        f'<div class="backtest-kpi-card"><div class="backtest-kpi-label">AI CUMULATIVE RETURN</div><div class="backtest-kpi-value {_kpi_val_cls(ai_cum_pct)}">{ai_cum_pct:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                with kpi_cols[1]:
                    st.markdown(
                        f'<div class="backtest-kpi-card"><div class="backtest-kpi-label">AI ANNUALIZED RETURN</div><div class="backtest-kpi-value {_kpi_val_cls(ai_ann_pct)}">{ai_ann_pct:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                with kpi_cols[2]:
                    st.markdown(
                        f'<div class="backtest-kpi-card"><div class="backtest-kpi-label">SPY CUMULATIVE RETURN</div><div class="backtest-kpi-value {_kpi_val_cls(spy_cum_pct)}">{spy_cum_pct:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                with kpi_cols[3]:
                    st.markdown(
                        f'<div class="backtest-kpi-card"><div class="backtest-kpi-label">SPY ANNUALIZED RETURN</div><div class="backtest-kpi-value {_kpi_val_cls(spy_ann_pct)}">{spy_ann_pct:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )

                bt_df["Cumulative_Return_AI"] = (bt_df["ai_cumulative"] - 1.0) * 100.0
                bt_df["Cumulative_Return_SPY"] = (bt_df["spy_cumulative"] - 1.0) * 100.0

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=bt_df["date"],
                        y=bt_df["Cumulative_Return_AI"],
                        mode="lines",
                        line=dict(color=MINT_GREEN, width=10, shape="spline"),
                        opacity=0.3,
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=bt_df["date"],
                        y=bt_df["Cumulative_Return_AI"],
                        mode="lines",
                        name="AI Strategy",
                        line=dict(color=MINT_GREEN, width=2.5, shape="spline"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=bt_df["date"],
                        y=bt_df["Cumulative_Return_SPY"],
                        mode="lines",
                        name="SPY Benchmark",
                        line=dict(color="#6b7b8c", width=1.5, dash="dash"),
                    )
                )
                fig.add_hline(y=0.0, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                    yaxis_title="Cumulative Return (%)",
                    xaxis_title="Date",
                )
                fig.update_xaxes(showgrid=False, zeroline=False)
                fig.update_yaxes(showgrid=False, zeroline=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    "Our AI strategy combines sector ETF momentum and macro drivers like VIX and TNX to tilt exposure "
                    "toward favorable regimes, aiming to outperform a simple buy‑and‑hold SPY benchmark over time."
                )

        with tab_news:
            st.markdown('<p class="card-title">Market news & key stats</p>', unsafe_allow_html=True)
            stats = load_key_stats(ticker)
            dy = stats.get("dividendYield")
            dy_str = f"{float(dy)*100:.2f}%" if dy is not None else "—"
            stats_html = (
                '<div class="stats-grid">'
                f'<div class="stat-item"><span class="stat-label">Market cap</span><span class="stat-val">{_fmt_number(stats.get("marketCap"))}</span></div>'
                f'<div class="stat-item"><span class="stat-label">P/E (TTM)</span><span class="stat-val">{_fmt_number(stats.get("trailingPE"))}</span></div>'
                f'<div class="stat-item"><span class="stat-label">Dividend yield</span><span class="stat-val">{dy_str}</span></div>'
                f'<div class="stat-item"><span class="stat-label">52W high</span><span class="stat-val">{_fmt_number(stats.get("fiftyTwoWeekHigh"))}</span></div>'
                f'<div class="stat-item"><span class="stat-label">52W low</span><span class="stat-val">{_fmt_number(stats.get("fiftyTwoWeekLow"))}</span></div>'
                f'<div class="stat-item"><span class="stat-label">Sector</span><span class="stat-val">{str(stats.get("sector") or "—")}</span></div>'
                "</div>"
            )
            st.markdown(stats_html, unsafe_allow_html=True)
            news_recent = _news_last_7_days(news_df.copy() if not news_df.empty else pd.DataFrame())
            if not news_recent.empty and "published" in news_recent.columns:
                news_recent = news_recent.sort_values("published", ascending=False).head(12)
                st.markdown('<p class="card-title">Recent news</p>', unsafe_allow_html=True)
                news_items = []
                for _, row in news_recent.iterrows():
                    title = str(row.get("title", "")).strip()
                    link = str(row.get("link", "")).strip()
                    if link:
                        news_items.append(f'<div class="news-item"><a href="{link}" target="_blank">{title}</a></div>')
                    else:
                        news_items.append(f'<div class="news-item">{title}</div>')
                st.markdown('<div class="news-list">' + "".join(news_items) + "</div>", unsafe_allow_html=True)
            else:
                st.caption("No recent news for this ticker.")


if __name__ == "__main__":
    main()
