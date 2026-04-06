import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
// framer-motion REMOVED — it was loading a duplicate React instance from a
// parent node_modules tree, causing "Invalid hook call" and a blank screen.
// All animations replaced with CSS transitions.
import { Search, ArrowLeft, TrendingUp, TrendingDown, CheckCircle, AlertTriangle, Minus, Info, Star, X, Briefcase } from 'lucide-react';
import {
  ComposedChart,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';

// ── React Error Boundary — catches any child render crash and shows a recoverable fallback ──
class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, info) { console.error('[GlobalMarketPredictor] Uncaught render error:', error, info); }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#0B0E14] flex items-center justify-center">
          <div className="text-center max-w-md px-6">
            <p className="text-2xl font-bold text-white mb-3">Something went wrong</p>
            <p className="text-sm text-gray-400 mb-6">{String(this.state.error?.message || 'Unknown error')}</p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="px-5 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const CARD_BASE = 'bg-[#12131a] border border-slate-800/80 rounded-xl';
// Finnhub API key with hard-coded fallback so missing env does not break live data.
const FINNHUB_API_TOKEN = (import.meta.env?.VITE_FINNHUB_KEY || 'd6r89m1r01qgdhqdj95gd6r89m1r01qgdhqdj960')?.trim?.();
const FEAR_GREED_API_URL = 'http://127.0.0.1:5000/api/fear-greed';

/** FastAPI backend base URL: local dev vs production (Render). Uses hostname so prod builds never default to localhost. */
function resolveApiBaseUrl() {
  if (typeof window === 'undefined') {
    return 'https://globalmarketpredictor.onrender.com';
  }
  const host = window.location.hostname;
  if (host === 'localhost' || host === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  return 'https://globalmarketpredictor.onrender.com';
}
const API_BASE_URL = resolveApiBaseUrl().replace(/\/$/, '');
const LIVE_POLL_MS = 60 * 1000;
const WATCHLIST_STORAGE_KEY = 'globalMarketPredictor_watchlist_v1';
const WATCHLIST_MAX = 30;

function loadWatchlistTickers() {
  try {
    const raw = localStorage.getItem(WATCHLIST_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return [...new Set(parsed.map((t) => String(t || '').trim().toUpperCase()).filter(Boolean))].slice(0, WATCHLIST_MAX);
  } catch {
    return [];
  }
}

function saveWatchlistTickers(tickers) {
  try {
    localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(tickers));
  } catch {
    /* ignore quota / private mode */
  }
}
const FINNHUB_QUOTE_URL = (ticker) =>
  `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(ticker)}&token=${FINNHUB_API_TOKEN}`;
const FINNHUB_CANDLE_URL = (ticker, from, to) =>
  `https://finnhub.io/api/v1/stock/candle?symbol=${encodeURIComponent(ticker)}&resolution=D&from=${from}&to=${to}&token=${FINNHUB_API_TOKEN}`;

async function fetchLiveQuote(ticker) {
  try {
    if (!FINNHUB_API_TOKEN) {
      console.warn('[GlobalMarketPredictor] Missing VITE_FINNHUB_KEY; skipping Finnhub quote fetch.');
      return null;
    }
    const res = await fetch(FINNHUB_QUOTE_URL(ticker));
    const data = await res.json();
    if (!res.ok) return null;
    const c = Number(data?.c);
    if (!Number.isFinite(c) || c <= 0) return null;
    const d = Number(data?.d);
    const dp = Number(data?.dp);
    return {
      price: c,
      change: Number.isFinite(d) ? d : 0,
      changePercent: Number.isFinite(dp) ? dp : 0,
    };
  } catch (_) {
    return null;
  }
}

async function fetchLiveCandles(ticker) {
  const to = Math.floor(Date.now() / 1000);
  const from = to - (30 * 24 * 60 * 60);
  try {
    if (!FINNHUB_API_TOKEN) {
      console.warn('[GlobalMarketPredictor] Missing VITE_FINNHUB_KEY; skipping Finnhub candle fetch.');
      return null;
    }
    const res = await fetch(FINNHUB_CANDLE_URL(ticker, from, to));
    const data = await res.json();
    if (!res.ok || data?.s !== 'ok' || !Array.isArray(data?.t) || !Array.isArray(data?.c)) return null;
    const fmt = new Intl.DateTimeFormat('en-US', { month: 'numeric', day: 'numeric', timeZone: 'UTC' });
    const rows = data.t
      .map((ts, idx) => {
        const close = Number(data.c[idx]);
        if (!Number.isFinite(close)) return null;
        return {
          date: fmt.format(new Date(Number(ts) * 1000)),
          value: +close.toFixed(2),
        };
      })
      .filter(Boolean);
    return rows.length > 0 ? rows : null;
  } catch (_) {
    return null;
  }
}

function formatSyncedTime(ts) {
  if (!ts) return '—';
  return new Date(ts).toLocaleTimeString('en-GB');
}

function LiveIndicator({ lastSynced }) {
  return (
    <div className="inline-flex items-center gap-1.5 text-[10px] text-gray-500 uppercase tracking-wider">
      <span className="relative flex h-1.5 w-1.5 shrink-0">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-70" />
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
      </span>
      <span>Last synced: {formatSyncedTime(lastSynced)}</span>
    </div>
  );
}

/**
 * SINGLE SOURCE OF TRUTH for AI signal derivation.
 * Every component — table, detail header, gauge — calls this function.
 * Score thresholds:  0-40 → Sell | 41-65 → Hold | 66-84 → Buy | 85-100 → Strong Buy
 */
function getSignal(score) {
  const s = Math.min(100, Math.max(0, score ?? 0));
  if (s >= 85) return { label: 'Strong Buy', hex: '#22c55e', textCls: 'text-green-400',   bgCls: 'bg-green-400/10',   borderCls: 'border-green-400/30',   Icon: CheckCircle   };
  if (s >= 66) return { label: 'Buy',         hex: '#10b981', textCls: 'text-emerald-400', bgCls: 'bg-emerald-400/10', borderCls: 'border-emerald-400/25', Icon: CheckCircle   };
  if (s >= 41) return { label: 'Hold',        hex: '#eab308', textCls: 'text-amber-400',   bgCls: 'bg-amber-400/10',   borderCls: 'border-amber-400/25',   Icon: Minus         };
  return             { label: 'Sell',         hex: '#ef4444', textCls: 'text-red-400',      bgCls: 'bg-red-400/10',     borderCls: 'border-red-400/25',     Icon: AlertTriangle };
}

// Exactly 20 US stocks — aiSignal/confidence REMOVED; signal is derived via getSignal(STOCK_AI_RATINGS[ticker].score)
const TOP_US_STOCKS = [
  { ticker: 'AAPL',  name: 'Apple Inc.',           price: 228.42, changePercent:  1.24, high52: 237.23, low52: 164.08, avgVolume: 52_100_000, pe: 29.5, dividendYield: 0.44, domain: 'apple.com',             marketCap: 3500 },
  { ticker: 'NVDA',  name: 'NVIDIA Corp.',          price: 118.92, changePercent:  3.42, high52: 140.76, low52:  39.23, avgVolume: 98_500_000, pe: 54.2, dividendYield: 0.03, domain: 'nvidia.com',            marketCap: 2900 },
  { ticker: 'MSFT',  name: 'Microsoft Corp.',       price: 415.86, changePercent:  0.89, high52: 430.82, low52: 309.45, avgVolume: 22_400_000, pe: 36.1, dividendYield: 0.72, domain: 'microsoft.com',         marketCap: 3100 },
  { ticker: 'TSLA',  name: 'Tesla Inc.',            price: 242.18, changePercent: -2.15, high52: 299.29, low52: 138.80, avgVolume: 98_200_000, pe: 68.4, dividendYield: 0.00, domain: 'tesla.com',             marketCap:  770 },
  { ticker: 'AMZN',  name: 'Amazon.com Inc.',       price: 198.34, changePercent:  1.12, high52: 211.26, low52: 118.35, avgVolume: 48_300_000, pe: 43.2, dividendYield: 0.00, domain: 'amazon.com',            marketCap: 2060 },
  { ticker: 'META',  name: 'Meta Platforms',        price: 518.22, changePercent:  2.08, high52: 531.49, low52: 285.25, avgVolume: 16_200_000, pe: 25.8, dividendYield: 0.39, domain: 'meta.com',              marketCap: 1320 },
  { ticker: 'GOOGL', name: 'Alphabet Inc.',         price: 172.55, changePercent:  0.56, high52: 182.94, low52: 102.22, avgVolume: 24_100_000, pe: 24.3, dividendYield: 0.50, domain: 'google.com',            marketCap: 2150 },
  { ticker: 'BRK.B', name: 'Berkshire Hathaway',   price: 408.50, changePercent:  0.34, high52: 435.00, low52: 352.00, avgVolume:  3_200_000, pe: 22.1, dividendYield: 0.00, domain: 'berkshirehathaway.com', marketCap:  900 },
  { ticker: 'V',     name: 'Visa Inc.',             price: 278.90, changePercent:  0.45, high52: 290.96, low52: 239.51, avgVolume:  6_800_000, pe: 31.6, dividendYield: 0.76, domain: 'visa.com',              marketCap:  570 },
  { ticker: 'UNH',   name: 'UnitedHealth Group',   price: 558.20, changePercent: -0.88, high52: 571.00, low52: 445.00, avgVolume:  3_500_000, pe: 18.4, dividendYield: 1.58, domain: 'unitedhealthgroup.com', marketCap:  520 },
  { ticker: 'JPM',   name: 'JPMorgan Chase',        price: 198.76, changePercent: -0.22, high52: 211.00, low52: 155.00, avgVolume:  9_100_000, pe: 11.2, dividendYield: 2.32, domain: 'jpmorganchase.com',     marketCap:  570 },
  { ticker: 'JNJ',   name: 'Johnson & Johnson',     price: 157.40, changePercent:  0.62, high52: 175.00, low52: 142.00, avgVolume:  8_200_000, pe: 16.8, dividendYield: 3.28, domain: 'jnj.com',               marketCap:  380 },
  { ticker: 'WMT',   name: 'Walmart Inc.',          price: 188.65, changePercent:  1.35, high52: 195.00, low52: 155.00, avgVolume: 18_400_000, pe: 28.9, dividendYield: 1.02, domain: 'walmart.com',           marketCap:  510 },
  { ticker: 'XOM',   name: 'Exxon Mobil Corp.',     price: 118.30, changePercent:  0.77, high52: 125.00, low52:  98.00, avgVolume: 16_800_000, pe: 13.4, dividendYield: 3.18, domain: 'exxonmobil.com',        marketCap:  470 },
  { ticker: 'MA',    name: 'Mastercard Inc.',       price: 485.60, changePercent:  0.72, high52: 510.00, low52: 385.00, avgVolume:  5_200_000, pe: 35.2, dividendYield: 0.55, domain: 'mastercard.com',        marketCap:  470 },
  { ticker: 'AVGO',  name: 'Broadcom Inc.',         price: 178.22, changePercent:  1.67, high52: 195.00, low52: 125.00, avgVolume:  4_100_000, pe: 28.4, dividendYield: 1.42, domain: 'broadcom.com',          marketCap:  820 },
  { ticker: 'PG',    name: 'Procter & Gamble',      price: 164.55, changePercent: -0.30, high52: 175.00, low52: 140.00, avgVolume:  5_800_000, pe: 24.1, dividendYield: 2.34, domain: 'pg.com',                marketCap:  390 },
  { ticker: 'ORCL',  name: 'Oracle Corp.',          price: 142.88, changePercent: -1.12, high52: 155.00, low52: 115.00, avgVolume:  9_500_000, pe: 19.6, dividendYield: 1.26, domain: 'oracle.com',            marketCap:  390 },
  { ticker: 'COST',  name: 'Costco Wholesale',      price: 845.90, changePercent:  0.91, high52: 890.00, low52: 680.00, avgVolume:  2_100_000, pe: 52.3, dividendYield: 0.60, domain: 'costco.com',            marketCap:  375 },
  { ticker: 'HD',    name: 'Home Depot',            price: 385.20, changePercent: -0.41, high52: 420.00, low52: 315.00, avgVolume:  4_200_000, pe: 23.8, dividendYield: 2.30, domain: 'homedepot.com',         marketCap:  380 },
];

function getStockMeta(ticker) {
  const u = String(ticker || '').toUpperCase();
  return TOP_US_STOCKS.find((s) => s.ticker?.toUpperCase() === u);
}

/** Demo portfolio: fixed positions for UI only (tickers must exist in TOP_US_STOCKS). */
const DEMO_PORTFOLIO_HOLDINGS = [
  { ticker: 'AAPL', shares: 12, avgCost: 198.5 },
  { ticker: 'MSFT', shares: 8, avgCost: 402.0 },
  { ticker: 'NVDA', shares: 15, avgCost: 95.25 },
  { ticker: 'GOOGL', shares: 10, avgCost: 158.0 },
  { ticker: 'AMZN', shares: 14, avgCost: 185.0 },
  { ticker: 'META', shares: 6, avgCost: 485.0 },
];

/** Builds demo portfolio rows with live prices plus AI signal from STOCK_AI_RATINGS (same engine as stock detail). */
function buildDemoPortfolioRows(liveQuotes) {
  return DEMO_PORTFOLIO_HOLDINGS.map((h) => {
    const meta = getStockMeta(h.ticker);
    const q = liveQuotes[h.ticker];
    const price = Number(q?.price ?? meta?.price ?? 0);
    const shares = Number(h.shares);
    const avgCost = Number(h.avgCost);
    const costBasis = shares * avgCost;
    const marketValue = shares * price;
    const pl = marketValue - costBasis;
    const plPct = costBasis > 0 ? (pl / costBasis) * 100 : 0;
    const rating = STOCK_AI_RATINGS[h.ticker];
    const aiScore = rating?.score ?? 0;
    const signal = getSignal(aiScore);
    return {
      ticker: h.ticker,
      shares,
      avgCost,
      name: meta?.name ?? h.ticker,
      price,
      costBasis,
      marketValue,
      pl,
      plPct,
      aiScore,
      signal,
      aiSummary: rating?.summary ?? '',
    };
  });
}

function sumDemoPortfolioTotals(rows) {
  const costBasis = rows.reduce((s, r) => s + r.costBasis, 0);
  const marketValue = rows.reduce((s, r) => s + r.marketValue, 0);
  const pl = marketValue - costBasis;
  const plPct = costBasis > 0 ? (pl / costBasis) * 100 : 0;
  return { costBasis, marketValue, pl, plPct };
}

/** Mock algo performance — replace when API provides analytics. */
const SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK = {
  totalTradesExecuted: 47,
  winRatePct: 63.8,
  bestTrade: { ticker: 'NVDA', label: '+$3,180', sub: '+14.6% round-trip' },
  worstTrade: { ticker: 'META', label: '−$624', sub: '−2.1% round-trip' },
};

/** Mock recent executions — replace when API supplies model trade log. */
const SIGNAL_DEMO_RECENT_MODEL_ACTIONS_MOCK = [
  { date: '2026-04-04', ticker: 'NVDA', action: 'BUY', shares: 18, executionPrice: 118.4 },
  { date: '2026-04-03', ticker: 'MSFT', action: 'SELL', shares: 4, executionPrice: 416.25 },
  { date: '2026-04-02', ticker: 'AAPL', action: 'BUY', shares: 25, executionPrice: 226.1 },
  { date: '2026-04-01', ticker: 'GOOGL', action: 'BUY', shares: 22, executionPrice: 171.88 },
  { date: '2026-03-31', ticker: 'AMZN', action: 'SELL', shares: 9, executionPrice: 199.5 },
  { date: '2026-03-28', ticker: 'META', action: 'BUY', shares: 5, executionPrice: 512.4 },
  { date: '2026-03-27', ticker: 'NVDA', action: 'SELL', shares: 12, executionPrice: 115.05 },
  { date: '2026-03-26', ticker: 'AAPL', action: 'SELL', shares: 10, executionPrice: 223.9 },
];

/**
 * Full paper-trade log for performance drill-down modal.
 * `pl` = realized P/L on SELL legs (null on BUY / opens).
 * `isWin` = true when pl > 0 on a closing leg (for win-rate drill-down).
 */
const SIGNAL_DEMO_TRADE_HISTORY_FULL_MOCK = [
  { date: '2026-04-04', ticker: 'NVDA', action: 'BUY', shares: 18, executionPrice: 118.4, pl: null, isWin: null },
  { date: '2026-04-03', ticker: 'MSFT', action: 'SELL', shares: 4, executionPrice: 416.25, pl: 412.5, isWin: true },
  { date: '2026-04-02', ticker: 'AAPL', action: 'BUY', shares: 25, executionPrice: 226.1, pl: null, isWin: null },
  { date: '2026-04-01', ticker: 'GOOGL', action: 'BUY', shares: 22, executionPrice: 171.88, pl: null, isWin: null },
  { date: '2026-03-31', ticker: 'AMZN', action: 'SELL', shares: 9, executionPrice: 199.5, pl: -88.2, isWin: false },
  { date: '2026-03-30', ticker: 'NVDA', action: 'BUY', shares: 30, executionPrice: 116.2, pl: null, isWin: null },
  { date: '2026-03-29', ticker: 'GOOGL', action: 'SELL', shares: 15, executionPrice: 173.4, pl: 521.4, isWin: true },
  { date: '2026-03-28', ticker: 'META', action: 'BUY', shares: 5, executionPrice: 512.4, pl: null, isWin: null },
  { date: '2026-03-27', ticker: 'NVDA', action: 'SELL', shares: 12, executionPrice: 115.05, pl: 1842.6, isWin: true },
  { date: '2026-03-26', ticker: 'AAPL', action: 'SELL', shares: 10, executionPrice: 223.9, pl: 156.0, isWin: true },
  { date: '2026-03-25', ticker: 'MSFT', action: 'BUY', shares: 10, executionPrice: 409.8, pl: null, isWin: null },
  { date: '2026-03-24', ticker: 'META', action: 'SELL', shares: 4, executionPrice: 508.1, pl: -624.0, isWin: false },
  { date: '2026-03-23', ticker: 'AMZN', action: 'BUY', shares: 20, executionPrice: 192.4, pl: null, isWin: null },
  { date: '2026-03-22', ticker: 'NVDA', action: 'SELL', shares: 20, executionPrice: 112.8, pl: 3180.0, isWin: true },
  { date: '2026-03-21', ticker: 'AAPL', action: 'BUY', shares: 15, executionPrice: 221.5, pl: null, isWin: null },
  { date: '2026-03-20', ticker: 'GOOGL', action: 'BUY', shares: 12, executionPrice: 169.2, pl: null, isWin: null },
  { date: '2026-03-19', ticker: 'MSFT', action: 'SELL', shares: 6, executionPrice: 405.0, pl: -198.0, isWin: false },
  { date: '2026-03-18', ticker: 'META', action: 'BUY', shares: 8, executionPrice: 518.0, pl: null, isWin: null },
  { date: '2026-03-17', ticker: 'NVDA', action: 'BUY', shares: 14, executionPrice: 114.5, pl: null, isWin: null },
  { date: '2026-03-16', ticker: 'AMZN', action: 'SELL', shares: 12, executionPrice: 195.2, pl: 310.8, isWin: true },
  { date: '2026-03-15', ticker: 'AAPL', action: 'SELL', shares: 8, executionPrice: 219.4, pl: -142.4, isWin: false },
  { date: '2026-03-14', ticker: 'GOOGL', action: 'SELL', shares: 10, executionPrice: 170.1, pl: 96.5, isWin: true },
  { date: '2026-03-13', ticker: 'MSFT', action: 'BUY', shares: 5, executionPrice: 412.3, pl: null, isWin: null },
  { date: '2026-03-12', ticker: 'META', action: 'SELL', shares: 3, executionPrice: 514.2, pl: 48.6, isWin: true },
  { date: '2026-03-11', ticker: 'NVDA', action: 'BUY', shares: 22, executionPrice: 110.9, pl: null, isWin: null },
  { date: '2026-03-10', ticker: 'AMZN', action: 'BUY', shares: 15, executionPrice: 189.7, pl: null, isWin: null },
  { date: '2026-03-09', ticker: 'AAPL', action: 'BUY', shares: 20, executionPrice: 224.8, pl: null, isWin: null },
  { date: '2026-03-08', ticker: 'GOOGL', action: 'SELL', shares: 8, executionPrice: 168.0, pl: -62.4, isWin: false },
  { date: '2026-03-07', ticker: 'MSFT', action: 'SELL', shares: 5, executionPrice: 418.6, pl: 285.5, isWin: true },
  { date: '2026-03-06', ticker: 'META', action: 'BUY', shares: 6, executionPrice: 505.5, pl: null, isWin: null },
  { date: '2026-03-05', ticker: 'NVDA', action: 'SELL', shares: 16, executionPrice: 108.2, pl: 412.8, isWin: true },
  { date: '2026-03-04', ticker: 'AMZN', action: 'SELL', shares: 18, executionPrice: 188.1, pl: -221.4, isWin: false },
  { date: '2026-03-03', ticker: 'AAPL', action: 'SELL', shares: 12, executionPrice: 227.3, pl: 378.6, isWin: true },
  { date: '2026-03-02', ticker: 'GOOGL', action: 'BUY', shares: 18, executionPrice: 166.4, pl: null, isWin: null },
  { date: '2026-03-01', ticker: 'MSFT', action: 'BUY', shares: 8, executionPrice: 401.2, pl: null, isWin: null },
  { date: '2026-02-28', ticker: 'META', action: 'SELL', shares: 5, executionPrice: 499.8, pl: -198.5, isWin: false },
  { date: '2026-02-27', ticker: 'NVDA', action: 'BUY', shares: 25, executionPrice: 107.1, pl: null, isWin: null },
  { date: '2026-02-26', ticker: 'AMZN', action: 'BUY', shares: 10, executionPrice: 185.5, pl: null, isWin: null },
  { date: '2026-02-25', ticker: 'AAPL', action: 'SELL', shares: 6, executionPrice: 231.2, pl: 192.3, isWin: true },
  { date: '2026-02-24', ticker: 'GOOGL', action: 'SELL', shares: 14, executionPrice: 165.8, pl: 144.2, isWin: true },
  { date: '2026-02-23', ticker: 'MSFT', action: 'SELL', shares: 4, executionPrice: 398.4, pl: -72.0, isWin: false },
  { date: '2026-02-22', ticker: 'META', action: 'BUY', shares: 4, executionPrice: 522.0, pl: null, isWin: null },
  { date: '2026-02-21', ticker: 'NVDA', action: 'SELL', shares: 10, executionPrice: 105.6, pl: 890.4, isWin: true },
  { date: '2026-02-20', ticker: 'AMZN', action: 'SELL', shares: 25, executionPrice: 182.3, pl: 505.0, isWin: true },
  { date: '2026-02-19', ticker: 'AAPL', action: 'BUY', shares: 30, executionPrice: 228.0, pl: null, isWin: null },
  { date: '2026-02-18', ticker: 'GOOGL', action: 'BUY', shares: 8, executionPrice: 163.5, pl: null, isWin: null },
  { date: '2026-02-17', ticker: 'MSFT', action: 'BUY', shares: 6, executionPrice: 395.0, pl: null, isWin: null },
  { date: '2026-02-16', ticker: 'META', action: 'SELL', shares: 2, executionPrice: 510.0, pl: -42.0, isWin: false },
];

function filterSignalDemoTradeHistory(mode, tickerFilter) {
  const all = [...SIGNAL_DEMO_TRADE_HISTORY_FULL_MOCK];
  if (mode === 'ticker' && tickerFilter) {
    const u = String(tickerFilter).toUpperCase();
    return all.filter((r) => String(r.ticker).toUpperCase() === u);
  }
  if (mode === 'wins') {
    return all.filter((r) => r.pl != null && r.pl > 0);
  }
  return all;
}

function AlgoExecutionBadge({ action }) {
  const isBuy = String(action || '').toUpperCase() === 'BUY';
  const Icon = isBuy ? CheckCircle : AlertTriangle;
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wide border ${
        isBuy
          ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25'
          : 'text-red-400 bg-red-400/10 border-red-400/25'
      }`}
    >
      <Icon className="w-3.5 h-3.5 shrink-0" />
      {isBuy ? 'BUY' : 'SELL'}
    </span>
  );
}

function SignalDemoTradeHistoryModal({ open, title, subtitle, rows, onClose, onSelectStock }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
      <div
        className="absolute inset-0 bg-black/65 backdrop-blur-[2px]"
        aria-hidden
        onMouseDown={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="signal-demo-trade-history-title"
        className={`relative w-full max-w-4xl max-h-[min(88vh,720px)] flex flex-col rounded-2xl border border-slate-700/80 shadow-2xl ${CARD_BASE} overflow-hidden z-[1]`}
        style={{ boxShadow: '0 25px 50px -12px rgba(0,0,0,0.65), 0 0 0 1px rgba(56,189,248,0.12)' }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-start justify-between gap-4 px-5 py-4 border-b border-slate-800 bg-[#0f1118]/95">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em] mb-1">Trade history</p>
            <h2 id="signal-demo-trade-history-title" className="text-lg font-semibold text-white tracking-tight truncate">
              {title}
            </h2>
            {subtitle ? <p className="text-xs text-gray-500 mt-1">{subtitle}</p> : null}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 p-2 rounded-lg border border-slate-700/80 bg-slate-900/60 text-gray-400 hover:text-white hover:border-slate-500 hover:bg-slate-800/80 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[640px]">
            <thead className="sticky top-0 bg-[#0f1118] border-b border-slate-800 z-[1]">
              <tr>
                <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Date</th>
                <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider w-12" aria-label="Logo" />
                <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Symbol</th>
                <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Action</th>
                <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Shares</th>
                <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Price</th>
                <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">P/L</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-12 px-4 text-center text-sm text-gray-500">
                    No trades match this view.
                  </td>
                </tr>
              ) : (
                rows.map((row, idx) => {
                  const stock = getStockMeta(row.ticker);
                  const pl = row.pl;
                  const plShow = pl == null ? null : Number(pl);
                  return (
                    <tr
                      key={`${row.date}-${row.ticker}-${idx}`}
                      onClick={() => {
                        if (stock && onSelectStock) {
                          onSelectStock(stock);
                          onClose();
                        }
                      }}
                      className={`border-b border-slate-800/50 hover:bg-white/[0.04] transition-colors ${stock && onSelectStock ? 'cursor-pointer' : ''}`}
                    >
                      <td className="py-2.5 px-3 text-sm text-gray-300 tabular-nums whitespace-nowrap">{row.date}</td>
                      <td className="py-2.5 px-3 align-middle">
                        <StockLogo ticker={row.ticker} />
                      </td>
                      <td className="py-2.5 px-3 text-sm font-medium text-white tabular-nums">{row.ticker}</td>
                      <td className="py-2.5 px-3 align-middle">
                        <AlgoExecutionBadge action={row.action} />
                      </td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{row.shares}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-white tabular-nums">
                        ${Number(row.executionPrice).toFixed(2)}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${
                          plShow == null ? 'text-gray-600' : plShow >= 0 ? 'text-emerald-400' : 'text-red-400'
                        }`}
                      >
                        {plShow == null ? '—' : `${plShow >= 0 ? '+' : ''}$${Math.abs(plShow).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        <div className="shrink-0 px-5 py-3 border-t border-slate-800 bg-[#12131a]/95 text-[11px] text-gray-500">
          {rows.length} row{rows.length !== 1 ? 's' : ''} in this view · paperbook simulation
        </div>
      </div>
    </div>
  );
}

const STOCK_ABOUT = {
  AAPL:  "Apple Inc. designs, manufactures and markets consumer electronics, software and services. Its product ecosystem spans the iPhone, Mac, iPad, Apple Watch and AirPods, bolstered by a rapidly expanding suite of subscription services — iCloud, Apple TV+, Apple Arcade and Apple Pay — that drive high-margin recurring revenue.",
  NVDA:  "NVIDIA Corporation is the world leader in accelerated computing. Its GPU architecture powers modern AI training and inference workloads across hyperscale data centers, autonomous vehicles and robotics. The CUDA software platform creates deep developer lock-in, while partnerships with every major cloud provider cement its role as the backbone of the global AI infrastructure buildout.",
  MSFT:  "Microsoft Corporation delivers cloud computing, productivity software and enterprise services to individuals and organizations worldwide. Azure is the second-largest cloud platform globally, Microsoft 365 commands the enterprise productivity market, and the company's early OpenAI partnership has positioned Copilot as the leading enterprise AI assistant.",
  TSLA:  "Tesla Inc. designs and manufactures electric vehicles, energy storage systems and solar products. Beyond hardware, Tesla's software-defined vehicle platform and Full Self-Driving subscription generate recurring revenue. The company's Gigafactory network provides scale advantages, though margin pressure from aggressive pricing cycles remains a near-term watch item.",
  AMZN:  "Amazon.com is the world's largest e-commerce and cloud computing company. AWS — its cloud division — contributes the majority of operating income despite representing a fraction of revenue. Prime membership, third-party seller services, digital advertising and an expanding healthcare vertical (One Medical, Amazon Pharmacy) provide multiple durable growth vectors.",
  META:  "Meta Platforms operates the world's largest social media ecosystem — Facebook, Instagram, WhatsApp and Threads — reaching over 3 billion daily active users. Advertising monetization drives near-term profits while heavy investment in Reality Labs and AI infrastructure positions Meta for the next computing platform. Improving cost discipline has reignited margin expansion.",
  GOOGL: "Alphabet Inc. is the parent of Google, which commands roughly 90% of global search and powers the digital advertising duopoly alongside Meta. Google Cloud is the fastest-growing of the hyperscaler trio. YouTube, Waymo, DeepMind and a prolific venture portfolio offer optionality far beyond core search, while AI integration across all products defends its moat.",
  'BRK.B': "Berkshire Hathaway is Warren Buffett's diversified conglomerate, holding wholly-owned businesses spanning insurance (GEICO, Gen Re), railroads (BNSF), utilities and manufacturing alongside a massive publicly-traded equity portfolio anchored by Apple, Bank of America and Coca-Cola. The unmatched insurance float provides low-cost capital to fund acquisitions indefinitely.",
  V:     "Visa Inc. operates the world's largest retail electronic payments network, enabling transactions in over 200 countries across 4 billion credentials. The asset-light model collects a small fee on every transaction without taking credit risk, generating extraordinary free cash flow. Digital payment volume growth and tap-to-pay adoption in emerging markets are multi-decade secular tailwinds.",
  UNH:   "UnitedHealth Group is the largest US health insurer and one of the most vertically integrated healthcare companies globally. Its Optum segment — combining pharmacy benefit management, healthcare services and analytics — generates diversified revenue streams that reduce dependence on traditional insurance margins. Scale advantages and data assets are formidable competitive moats.",
  JPM:   "JPMorgan Chase is the largest US bank by assets, operating a best-in-class franchise across consumer banking, commercial lending, investment banking and asset management. Industry-leading technology investment, a fortress balance sheet and disciplined underwriting have produced consistent returns through multiple credit cycles. CEO Jamie Dimon's stewardship commands premium market confidence.",
  JNJ:   "Johnson & Johnson is a global pharmaceutical and MedTech leader following its consumer health spinoff (Kenvue). The innovative medicine segment focuses on oncology, immunology and neuroscience, while MedTech covers surgical robotics, orthopaedics and vision. A deep pipeline, AAA credit rating and 60+ consecutive years of dividend increases define its blue-chip status.",
  WMT:   "Walmart Inc. is the world's largest retailer by revenue, operating nearly 11,000 stores across 24 countries under multiple banners. Its membership program (Walmart+), first-party advertising business and fast-growing e-commerce operations are structurally shifting the margin profile higher. Supply chain scale provides unmatched pricing power in any economic environment.",
  XOM:   "Exxon Mobil Corporation is one of the world's largest publicly traded oil and gas companies, with integrated operations spanning upstream exploration, refining, chemicals and lubricants. The $60B Pioneer Natural Resources acquisition meaningfully expanded Permian Basin output. Capital discipline, a strong balance sheet and a growing low-carbon portfolio in carbon capture and hydrogen round out the investment thesis.",
  MA:    "Mastercard Incorporated operates a global payments technology network connecting 3+ billion cardholders with 100+ million merchants in over 210 countries. Like Visa, its fee-based model carries zero credit risk while capturing value from every swipe, tap and click. Mastercard's Vocalink and real-time payments capabilities are extending relevance well beyond card rails.",
  AVGO:  "Broadcom Inc. is a global semiconductor and infrastructure software leader. Its custom ASIC business designs AI networking chips for Google, Meta and ByteDance, making it a key beneficiary of hyperscaler AI capex. The $69B VMware acquisition adds a large, recurring software revenue stream, transforming Broadcom into a hybrid chip-and-software franchise with expanding margins.",
  PG:    "Procter & Gamble Co. is a global consumer goods company with a portfolio of 65+ brands spanning fabric care, home care, baby care, feminine care and grooming — including Tide, Pampers, Gillette and Oral-B. Pricing power, category leadership in premium segments and a relentless focus on productivity savings sustain mid-single-digit EPS growth through commodity and consumer cycles.",
  ORCL:  "Oracle Corporation provides database technology, cloud infrastructure (OCI) and enterprise applications to organizations worldwide. OCI's rapid growth and competitive pricing against AWS and Azure are attracting workload migrations, while the AI demand for Oracle's GPU clusters has driven a backlog surge. However, legacy on-premise licensing headwinds and elevated debt from cloud investment warrant monitoring.",
  COST:  "Costco Wholesale Corporation operates members-only warehouse clubs offering a curated selection of merchandise at thin margins, monetizing primarily through annual membership fees. This model drives exceptional member loyalty (>90% renewal rates), high inventory turns and below-market pricing that is structurally difficult to replicate. E-commerce and international expansion extend the long-term runway.",
  HD:    "The Home Depot is the world's largest home improvement retailer, serving both professional contractors (Pro) and do-it-yourself consumers. The Pro segment — roughly half of revenue — drives higher ticket sizes and repeat purchasing. The SRS Distribution acquisition deepens Pro penetration in specialty trades. Near-term results face headwind from a soft housing turnover cycle.",
};

// AI composite score (0–100) + analyst summary + three-model breakdown per ticker
const STOCK_AI_RATINGS = {
  AAPL: {
    score: 84,
    summary: "Best-in-class brand loyalty and expanding services revenue justify the premium multiple. Hardware upgrade cycles remain healthy. Minor concern: revenue growth deceleration in saturated smartphone markets.",
    models: [
      { name: "DeepValue Engine",    score: 88, justification: "Apple's $110B+ annual FCF and price-to-FCF below its 5-year average signal moderate undervaluation relative to earnings power. Debt-to-equity of 1.9× is comfortably covered by 28× interest coverage. EPS CAGR of 11% projected through 2027 outpaces the current 29× multiple, and the services segment's 73% gross margin structurally lifts consolidated profitability each quarter." },
      { name: "Sentiment NLP",       score: 82, justification: "CEO commentary on India manufacturing scale-up and Vision Pro enterprise adoption carried measurably positive sentiment in the last two earnings transcripts. Analyst upgrade-to-downgrade ratio stands at 3.2:1 over the trailing 90 days. Social listening data shows iPhone brand loyalty scores at a multi-year high entering the fall product cycle, limiting downside risk to near-term demand estimates." },
      { name: "Tech-Momentum Net",   score: 81, justification: "Price has held above both the 50-day and 200-day SMA through three consecutive weeks of sector rotation, confirming trend resilience. MACD histogram is tightening but remains positive territory. OBV trend line supports institutional accumulation. Resistance at the 52-week high of $237.23 — a confirmed weekly close above that level would trigger a momentum extension signal." },
    ],
  },
  NVDA: {
    score: 92,
    summary: "Dominant AI infrastructure play with near-monopoly GPU pricing power and a multi-year data center backlog. Elevated valuation assumes continued execution — any demand miss or ASP compression could trigger sharp de-rating.",
    models: [
      { name: "DeepValue Engine",    score: 82, justification: "P/E of 54× and EV/EBITDA of 45× reflect a meaningful growth premium that is difficult to justify on traditional intrinsic value models alone. FCF yield of 2.1% sits below the risk-free rate threshold. However, the exceptional duration of the CUDA moat and the multi-year datacenter capex cycle extend the DCF terminal value materially, partially offsetting the near-term valuation concern." },
      { name: "Sentiment NLP",       score: 97, justification: "Hyperscaler earnings calls have referenced NVIDIA GPU constraints in over 80% of recent transcripts — the highest concentration in our NLP corpus. CEO Jensen Huang's GTC keynote generated the top NLP sentiment score in our dataset this year. Supply scarcity narrative is fully embedded in institutional consensus language, with buy-side commentary describing Blackwell allocation as a strategic imperative rather than a discretionary spend." },
      { name: "Tech-Momentum Net",   score: 96, justification: "Relative strength vs. the Philadelphia Semiconductor Index stands at the 94th percentile over a 6-month horizon. MACD is accelerating above the signal line with expanding histogram amplitude. RSI at 71 is elevated but has not yet reached the 80+ exhaustion threshold seen in prior peak cycles. Institutional flow data confirms systematic accumulation on every dip since the $80 support zone was established." },
    ],
  },
  MSFT: {
    score: 88,
    summary: "Azure re-acceleration combined with Copilot enterprise monetization creates a durable compounding engine. Fortress balance sheet and disciplined capital allocation. Fairly valued versus long-term earnings power.",
    models: [
      { name: "DeepValue Engine",    score: 91, justification: "Azure's revenue share has expanded from 32% to 38% over the trailing eight quarters, with cloud-segment margins running 800bps above core enterprise software. EV/FCF of 31× represents a modest premium for a business compounding free cash flow at 16% annually. The $100B+ net cash position provides significant optionality for incremental capital returns and strategic acquisitions without balance sheet stress." },
      { name: "Sentiment NLP",       score: 87, justification: "References to 'Copilot enterprise adoption' and 'AI-driven productivity uplift' appeared 34 times in the last earnings call transcript — a 3× year-over-year increase in frequency. Corporate channel checks indicate accelerating Microsoft 365 E5 upgrades driven specifically by AI feature sets. Analyst consensus has drifted upward across all major coverage groups, with price target revisions running 2.1:1 positive over the past quarter." },
      { name: "Tech-Momentum Net",   score: 85, justification: "MSFT outperformed the Nasdaq-100 by 8.3 percentage points over the trailing 6 months. Price is advancing within a defined ascending channel with intact higher lows. ADX of 28 confirms a trending rather than ranging environment. Momentum signals remain constructive, though the pace of relative outperformance has moderated slightly from Q3 highs, suggesting some near-term mean reversion risk." },
    ],
  },
  TSLA: {
    score: 51,
    summary: "Structural EV margin compression and intensifying Chinese competition weigh heavily on near-term outlook. Full Self-Driving optionality remains speculative. Premium valuation demands operational execution the company has struggled to consistently deliver.",
    models: [
      { name: "DeepValue Engine",    score: 38, justification: "Automotive gross margins have compressed from 28% to 17% over six consecutive quarters of aggressive price cuts, structurally impairing the earnings quality that previously justified the premium multiple. EV/EBITDA of 52× assumes near-perfect execution in a market where BYD is gaining share at Tesla's direct expense. FCF generation has become inconsistent and increasingly reliant on lumpy regulatory credit sales to mask the deterioration." },
      { name: "Sentiment NLP",       score: 58, justification: "Management language in recent earnings transcripts has shifted from 'accelerating demand signals' to 'strategic volume prioritization' — a statistically significant change in our NLP sentiment model that historically precedes guidance risk. Short interest remains elevated at 3.8% of float. FSD v13 beta feedback is net positive in enthusiast communities but professional analyst commentary on NHTSA review timelines is cautious." },
      { name: "Tech-Momentum Net",   score: 55, justification: "TSLA has underperformed the XLY consumer discretionary ETF by 22 percentage points over the trailing 12 months. Price action has been range-bound between $215 and $270 for three consecutive months, indicating indecision. RSI hovering at 44 — below neutral but not at the oversold extreme that would trigger contrarian long signals. MACD line remains below its signal line, sustaining the intermediate-term bearish bias." },
    ],
  },
  AMZN: {
    score: 82,
    summary: "AWS re-acceleration and digital advertising scale are powerful margin drivers operating well below their potential. Prime ecosystem stickiness is unmatched globally. Core retail still dilutes consolidated margins, creating an earnings inflection story.",
    models: [
      { name: "DeepValue Engine",    score: 85, justification: "AWS operating margin has expanded to 38% from 30% two years ago, demonstrating significant operating leverage as fixed infrastructure costs are amortized across a widening revenue base. The advertising segment, growing at 19% YoY, is near-pure-margin incremental revenue that consensus models underweight. Retail operating income inflection is real — North America segment now generates a 5.3% operating margin versus breakeven 18 months ago." },
      { name: "Sentiment NLP",       score: 80, justification: "Andy Jassy's commentary on AI inference demand and the Bedrock platform has shifted tone from defensive to assertive over the last three quarters. AWS win-rate channel checks confirm improving competitive positioning in financial services and healthcare verticals. Prime Video ad-tier language remains cautious following churn data, creating a modest offset to the overall positive sentiment signal in recent transcripts." },
      { name: "Tech-Momentum Net",   score: 81, justification: "AMZN has formed a textbook cup-and-handle base on the weekly chart, with the breakout above $185 confirmed on above-average volume. Relative strength vs. XLY has been positive for four consecutive months. MACD crossed bullish in early Q4 and remains constructive. Volume on up-days has exceeded volume on down-days by a 1.8× ratio over the trailing 20 sessions, confirming institutional demand." },
    ],
  },
  META: {
    score: 87,
    summary: "Advertising revenue machine with industry-leading engagement across 3B+ daily users. Dramatic cost discipline has sharply expanded margins. Reality Labs capex is a long-term wildcard, but AI-enhanced ad targeting is extending the competitive moat.",
    models: [
      { name: "DeepValue Engine",    score: 87, justification: "Revenue per daily active user has expanded 18% YoY while headcount discipline has driven EBITDA margins to 42% — approximately 600bps above the estimated secular steady state but defensible given the advertising pricing flywheel. EV/FCF of 22× is a material discount to the S&P 500 growth cohort for a business compounding FCF at 25%+. The 4.8% buyback yield provides tangible price floor support during market dislocations." },
      { name: "Sentiment NLP",       score: 89, justification: "Zuckerberg's AI infrastructure commentary and Llama open-source strategy consistently generate the highest NLP sentiment scores in our mega-cap technology corpus. App family engagement metrics are cited with increasing conviction in each successive earnings call. Agency channel checks point to an accelerating budget allocation shift toward Reels monetization, with CPM rates rising quarter-over-quarter across all geographies." },
      { name: "Tech-Momentum Net",   score: 85, justification: "META ranked in the top five momentum performers in the S&P 500 over the trailing 12 months. Price held the 50-day SMA during the October sector correction while peer communication services names broke key support. Bollinger Band width is contracting — historically a precursor to a directional expansion move in this name. Systematic institutional flow data shows continued net buying on every pullback." },
    ],
  },
  GOOGL: {
    score: 83,
    summary: "Search moat remains durable despite near-term AI disruption concerns. Cloud segment is accelerating materially. Trading at a discount to peers relative to earnings power. Key risk: regulatory overhang and monetization pressure in search.",
    models: [
      { name: "DeepValue Engine",    score: 85, justification: "Alphabet trades at 17× forward FCF — a material discount to both Meta and Microsoft despite generating comparable FCF margins. The $80B+ annual capital return program via buybacks provides durable EPS accretion independent of revenue growth. Google Cloud approaching segment-level breakeven represents a significant earnings catalyst that consensus models only partially price in at current multiples." },
      { name: "Sentiment NLP",       score: 80, justification: "Search advertising sentiment from agency channel checks remains constructive, with no material evidence of AI chatbot traffic cannibalizing commercial query intent volumes. YouTube creator economy sentiment improved following the expanded ad revenue sharing program announcement. Regulatory overhang language appears with elevated frequency in analyst notes, creating an artificial pessimism discount that has historically been mean-reverting." },
      { name: "Tech-Momentum Net",   score: 83, justification: "GOOGL has outperformed the XLC communication services ETF by 11 percentage points year-to-date. ADX at 32 confirms a trending regime with directional conviction. Price has tested the $183 resistance level twice and failed to break, suggesting distribution risk near the high. MACD histogram is positive and widening, which supports continuation of the current intermediate-term leg higher." },
    ],
  },
  'BRK.B': {
    score: 74,
    summary: "Conservative compounding vehicle with an unmatched balance sheet and insurance float optionality. Succession planning risk and structural underexposure to high-growth sectors limit near-term re-rating catalyst. Ideal core defensive holding.",
    models: [
      { name: "DeepValue Engine",    score: 83, justification: "Book value per share CAGR of 9.5% over 10 years with zero holding-company debt and a $168B cash reserve creates the most defensible risk/reward profile in the large-cap universe. The insurance float of $175B generates effectively free leverage on the equity portfolio. Price-to-book of 1.55× sits near the lower bound of Buffett's stated buyback threshold, providing a hard fundamental floor." },
      { name: "Sentiment NLP",       score: 70, justification: "The 2024 annual shareholder letter adopted a notably cautious tone regarding equity valuations broadly, which our NLP model flagged as a rare negative signal from management. Commentary around finding 'needle-moving' acquisition candidates was widely interpreted as acknowledging elevated market pricing. Succession planning questions re-emerged with elevated media frequency following the loss of Charlie Munger, creating a modest overhang." },
      { name: "Tech-Momentum Net",   score: 68, justification: "BRK.B has lagged the S&P 500 by 6 percentage points over the trailing 12 months, reflecting its low-beta, defensive characteristics in a risk-on, momentum-driven market environment. RSI at 51 is precisely neutral. Price is consolidating in a tight range rather than establishing a trend. Relative momentum vs. the financial sector index has been flat, indicating no near-term catalyst is being priced by quantitative systematic strategies." },
    ],
  },
  V: {
    score: 85,
    summary: "The most asset-efficient large-cap business globally. Payment digitization is an unstoppable multi-decade tailwind. Near-zero credit risk model generates extraordinary free cash flow. Only concern: elevated valuation and ongoing antitrust scrutiny.",
    models: [
      { name: "DeepValue Engine",    score: 89, justification: "Visa's asset-light model generates a 51% net profit margin with essentially zero credit risk on the balance sheet — a combination that is structurally unmatched in financial services. FCF conversion above 95% of net income enables a sustainable 3.2% buyback yield alongside consistent dividend growth. EV/EBITDA of 26× is above the S&P 500 median but justified by the secular TAM expansion in global digital payment penetration." },
      { name: "Sentiment NLP",       score: 83, justification: "Management commentary on cross-border travel recovery and value-added services revenue growth has been consistently positive across the last four quarterly earnings calls. Merchant acceptance partnership announcements have accelerated, with language around 'tap-to-pay' penetration in emerging markets cited as a multi-year growth vector. Antitrust scrutiny language appears in analyst reports with moderate frequency but has not yet escalated to a valuation discount in consensus models." },
      { name: "Tech-Momentum Net",   score: 83, justification: "V has delivered steady outperformance vs. the XLF financial sector ETF, with positive relative momentum maintained for six consecutive months. Price action is constructive — higher lows on weekly charts with volume confirming each up-leg. MACD is trending above zero with no divergence signals. The $290 resistance level has been tested and is approaching a resolution that systematic momentum models would treat as a breakout entry." },
    ],
  },
  UNH: {
    score: 68,
    summary: "Operational excellence and vertical integration via Optum remain powerful, but recent regulatory and legal headwinds have created meaningful sentiment overhang. Core healthcare demand is secular and non-cyclical.",
    models: [
      { name: "DeepValue Engine",    score: 72, justification: "UNH trades at 14× forward earnings — a 30% discount to its 5-year average — largely reflecting the regulatory and legal uncertainty that has compressed the multiple. Optum's revenue-per-member trajectory remains structurally intact. FCF generation of $22B+ annually provides a significant margin of safety at current prices, and the dividend yield of 1.6% offers partial return while the overhang resolves." },
      { name: "Sentiment NLP",       score: 62, justification: "Earnings call transcripts show a measurable increase in defensive language around regulatory exposure and legal proceedings over the past three quarters — a pattern our NLP model associates with elevated downside risk. Media sentiment has turned markedly negative following high-profile incidents and Congressional scrutiny. Buy-side commentary is increasingly cautious, with several large-cap healthcare funds disclosing reduced position sizes in recent 13F filings." },
      { name: "Tech-Momentum Net",   score: 70, justification: "UNH has underperformed the XLV healthcare ETF by 14 percentage points over the trailing 6 months, reflecting the idiosyncratic regulatory headlines rather than sector-wide weakness. RSI has recovered from oversold territory to 47, suggesting the worst of the selling pressure may be exhausted. Price is attempting to base-build above the $530 support level, which technicians are monitoring as a potential recovery pivot." },
    ],
  },
  JPM: {
    score: 76,
    summary: "Best-managed US bank with diversified revenue across consumer, commercial, IB and asset management. Credit quality is holding well through the cycle. Trading and IB volatility offset by consumer banking consistency. Fairly valued.",
    models: [
      { name: "DeepValue Engine",    score: 80, justification: "JPM's tangible book value per share CAGR of 12% over the past decade, combined with an ROTCE consistently above 17%, places it in the top decile of global banking institutions on capital efficiency metrics. P/TBV of 2.1× is above sector peers but warranted by the quality differential. Net interest income is benefiting from the higher-for-longer rate environment, and credit loss reserves remain conservatively positioned." },
      { name: "Sentiment NLP",       score: 74, justification: "CEO Jamie Dimon's public commentary has struck a cautious macro tone over the last two quarters, which our NLP model interprets as a hedge-in-advance signal rather than fundamental deterioration. Investment banking pipeline language has improved notably from H1 lows, with M&A advisory and ECM described as 'selectively improving.' Consumer deposit stability commentary is positive and consistent across all recent communications." },
      { name: "Tech-Momentum Net",   score: 73, justification: "JPM has tracked the KBW Bank Index closely with marginal outperformance, reflecting its bellwether status. Price is trading in a consolidation range between $185 and $211, with no clear trending signal from the ADX reading of 18. Relative momentum vs. the broader financial sector is neutral. A definitive break above $211 on volume would trigger a momentum re-entry signal in our model framework." },
    ],
  },
  JNJ: {
    score: 77,
    summary: "MedTech and innovative medicine franchise generates reliable free cash flow post-Kenvue spin. Dividend Aristocrat with 60+ year growth streak. Oncology and immunology pipeline progression is the key upside catalyst over the next 3 years.",
    models: [
      { name: "DeepValue Engine",    score: 82, justification: "Post-Kenvue spin, JNJ's remaining MedTech and pharmaceutical businesses generate 35%+ operating margins with highly durable FCF characteristics. The 3.3× debt/EBITDA is manageable given the AAA credit rating — the only industrial company globally retaining this designation. Trading at 13× forward earnings, a 20% discount to the large-cap pharmaceutical peer group, implies a margin of safety for patient capital." },
      { name: "Sentiment NLP",       score: 73, justification: "Innovative medicine pipeline commentary has turned progressively more confident, with oncology and immunology candidates cited with greater specificity in recent investor day materials. Talc litigation resolution language has moved from 'ongoing uncertainty' to 'path toward resolution,' which our NLP model treats as a meaningful sentiment improvement. Analyst coverage notes cite the pipeline as underappreciated versus consensus earnings models." },
      { name: "Tech-Momentum Net",   score: 75, justification: "JNJ has modestly outperformed the XLV healthcare ETF over the trailing 6 months after a prolonged period of underperformance tied to litigation overhang. Price has recovered above the 200-day SMA and is holding. RSI at 54 is neutral with upward bias. The $175 overhead resistance level — which previously acted as support — is the next meaningful test for the intermediate-term trend." },
    ],
  },
  WMT: {
    score: 80,
    summary: "The advertising and e-commerce transformation of the business model is significantly underappreciated by consensus. Walmart+ membership is gaining traction. Valuation has substantially re-rated — limited near-term upside but compelling long-term quality.",
    models: [
      { name: "DeepValue Engine",    score: 79, justification: "At 28× forward earnings, Walmart's valuation has re-rated significantly above its historical 20× multiple, reflecting the market's belated recognition of the advertising and fintech margin uplift. The advertising business alone — growing at 28% YoY — carries an estimated 70%+ operating margin, which structurally changes the consolidated earnings quality over a 3-5 year horizon. FCF generation remains robust at $16B+ annually, supporting the dividend and buyback program." },
      { name: "Sentiment NLP",       score: 83, justification: "Management's messaging around Walmart+ subscriber growth and first-party data monetization has become significantly more confident over the past two earnings cycles. The language shift from 'investing in capabilities' to 'harvesting returns' in advertising is a positive NLP signal our model assigns high conviction. Sell-side channel checks on grocery market share gains in key urban markets are consistently positive and ahead of consensus." },
      { name: "Tech-Momentum Net",   score: 78, justification: "WMT has delivered strong absolute performance over 12 months but relative outperformance vs. the XLP consumer staples ETF has moderated, suggesting the re-rating premium is now largely priced. Price is well above all major moving averages in a healthy trend. RSI at 62 shows momentum intact without excess. A pullback to the 50-day SMA around $178 would represent a technically constructive re-entry level for momentum buyers." },
    ],
  },
  XOM: {
    score: 65,
    summary: "Permian Basin volume growth post-Pioneer acquisition is a structural tailwind. Capital discipline much improved versus prior cycles. Energy transition risk is a long-term structural concern. Valuation looks reasonable on mid-cycle earnings.",
    models: [
      { name: "DeepValue Engine",    score: 70, justification: "At 13× forward earnings and a 3.2% dividend yield, XOM offers reasonable value on mid-cycle oil price assumptions of $75-$80/bbl WTI. The Pioneer acquisition added 1.3M Boepd of low-cost Permian production that should drive 15%+ FCF growth through 2026. Debt-to-capital of 15% is the lowest in XOM's modern history, providing resilience through commodity price cycles that have historically impaired less disciplined E&P operators." },
      { name: "Sentiment NLP",       score: 61, justification: "Energy sector language in institutional commentary has shifted to a cautious neutral, with macro concerns around China demand and OPEC+ production discipline dominating the narrative. XOM's internal investor communications remain constructive on Permian execution, but the broader sentiment environment is creating valuation compression across the integrated oil complex. ESG-driven divestiture flows from certain institutional categories remain a low-level but persistent headwind." },
      { name: "Tech-Momentum Net",   score: 63, justification: "XOM has materially underperformed the XLE energy ETF over the trailing 6 months despite strong operational execution, reflecting sector-wide macro headwinds. Price is range-bound between $110 and $125 with declining volume on up-moves — a bearish technical pattern. MACD has been oscillating around zero without conviction. A close above $125 with energy sector participation would be required to generate a constructive momentum signal." },
    ],
  },
  MA: {
    score: 84,
    summary: "Mirror image of Visa with superior international exposure and a slightly faster growth profile. Real-time payment network investments are extending the total addressable market well beyond card rails. Justified premium to Visa given trajectory.",
    models: [
      { name: "DeepValue Engine",    score: 88, justification: "Mastercard's revenue mix is structurally superior to Visa's, with higher international exposure providing incremental growth as emerging market card penetration advances. EV/FCF of 27× carries a premium to the market but is supported by 16% FCF CAGR visibility through 2027. The Vocalink and real-time payment network investments are optionality that traditional DCF models undervalue — they extend the TAM into account-to-account and government payment infrastructure." },
      { name: "Sentiment NLP",       score: 82, justification: "Cross-border travel commentary from management has been consistently positive across four consecutive earnings calls, with Europe and Asia-Pacific cited as the primary volume drivers ahead of consensus. The Mastercard Economics Institute publications reinforce the company's positioning as a thought leader in payments data, generating positive brand-level sentiment among institutional decision-makers. Regulatory antitrust language appears but is not cited as an imminent risk." },
      { name: "Tech-Momentum Net",   score: 82, justification: "MA has maintained a positive relative strength ranking vs. the XLF financial ETF for eight consecutive months — one of the longest such streaks in the current cycle. Price action features consistent higher highs and higher lows on the weekly chart, a hallmark of a healthy primary uptrend. MACD histogram amplitude is gradually expanding, confirming broadening momentum participation rather than a narrowing trend." },
    ],
  },
  AVGO: {
    score: 86,
    summary: "Custom AI ASIC franchise for hyperscalers is a durable competitive advantage with high switching costs. VMware integration is progressing ahead of schedule. Elevated post-acquisition leverage is offset by extraordinary free cash flow generation.",
    models: [
      { name: "DeepValue Engine",    score: 84, justification: "AVGO's custom ASIC franchise for Google TPUs and Meta's MTIA represents $10B+ in revenue from customers who face 3-5 year redesign cycles to switch away — creating deep structural switching costs that our DCF model values at a substantial premium. Post-VMware leverage of 3.8× debt/EBITDA is offset by $20B+ annual FCF generation, implying full deleveraging within 2.5 years at the current pace of integration synergy realization." },
      { name: "Sentiment NLP",       score: 88, justification: "VMware integration commentary has progressed from 'on track' to 'ahead of schedule' in the past two earnings calls — a language shift our NLP model associates with near-term upward estimate revisions. Hyperscaler partner commentary on custom silicon roadmaps increasingly names Broadcom alongside internal teams, confirming the market position. Institutional analyst reports describe the AI networking ASIC opportunity as 'still in the first inning' relative to GPU-centric positioning." },
      { name: "Tech-Momentum Net",   score: 85, justification: "AVGO has been among the top-decile semiconductor performers by relative strength over the trailing year, competing with NVDA for sector leadership. Price is advancing in a defined channel with no signs of distribution on high-volume days. RSI at 65 reflects strong momentum without the overbought excess that has preceded corrections in this name historically. Systematic flow models show accelerating institutional accumulation since the VMware close." },
    ],
  },
  PG: {
    score: 72,
    summary: "Proven pricing power and portfolio reshaping towards premium segments support margin expansion. Reliable compounder but limited alpha generation potential at current valuation. Ideal defensive anchor for risk-off portfolio positioning.",
    models: [
      { name: "DeepValue Engine",    score: 77, justification: "At 24× forward earnings, PG offers limited margin of safety for value-oriented investors but provides exceptionally reliable earnings quality — a key attribute during periods of macro uncertainty. Organic volume growth has stabilized following the price-driven growth phase, suggesting the business is executing the transition to unit growth on schedule. The 2.3% dividend yield, backed by 67 consecutive years of increases, provides a return floor independent of multiple expansion." },
      { name: "Sentiment NLP",       score: 70, justification: "Management commentary on pricing and promotional environment has become more measured in recent quarters, indicating that the easy pricing-driven EPS gains of 2022-2023 are largely exhausted. Channel checks from retail partners suggest promotional activity has been increasing modestly — a potential early signal of competitive pressure in the fabric care and baby care categories. Analyst tone remains constructive but cautious on near-term top-line growth velocity." },
      { name: "Tech-Momentum Net",   score: 68, justification: "PG has underperformed the XLP consumer staples ETF by 4 percentage points over the trailing 6 months, reflecting rotation toward higher-growth staples names. Price is in a well-defined but shallow uptrend with relatively low ADX of 16, indicating limited trend conviction. RSI at 52 is neutral. The name performs best as a defensive flight-to-quality trade — momentum signals are constructive primarily in risk-off market regimes." },
    ],
  },
  ORCL: {
    score: 55,
    summary: "OCI GPU cluster demand is a genuine positive surprise that is repricing the business. However, legacy database licensing erosion is structural, and elevated debt from cloud investment limits financial flexibility materially.",
    models: [
      { name: "DeepValue Engine",    score: 48, justification: "Oracle's total debt of $88B represents 4.9× adjusted EBITDA — one of the highest leverage ratios among large-cap technology companies — significantly constraining financial flexibility for further strategic investment. Legacy on-premise database licensing revenue, which carries 90%+ margins, is in structural decline as enterprise workloads migrate to cloud-native alternatives. EV/FCF of 24× prices in the OCI growth story fully, leaving limited margin of safety on a fundamental basis." },
      { name: "Sentiment NLP",       score: 60, justification: "OCI GPU cluster commentary has evolved from a side note to a central earnings narrative over the past three quarters, and management's confidence in the AI infrastructure backlog is a genuine sentiment positive. However, the elevated frequency of debt management and leverage language in transcripts is a concern flag our NLP model tracks carefully. Analyst coverage is split between enthusiasts focused on the AI opportunity and skeptics focused on balance sheet risk." },
      { name: "Tech-Momentum Net",   score: 56, justification: "ORCL has delivered positive absolute performance driven by AI infrastructure sentiment but has struggled to sustain breakouts above the $155 resistance level on multiple attempts. Volume patterns on failed breakout days show distribution rather than accumulation. RSI cycling between 45 and 62 without trend conviction. Relative performance vs. the IGV software ETF is negative — suggesting the market views the AI tailwind as a temporary rather than structural re-rating catalyst." },
    ],
  },
  COST: {
    score: 81,
    summary: "Warehouse model with captive, loyal membership base is structurally superior to traditional retail. Renewal rates above 90% provide predictable recurring revenue. Rich valuation requires near-flawless execution — Costco's track record justifies the premium.",
    models: [
      { name: "DeepValue Engine",    score: 83, justification: "Costco's 90%+ membership renewal rate generates a recurring revenue base that functions more like a subscription business than traditional retail, warranting a structural premium to the sector multiple. EV/EBITDA of 30× is elevated but supported by the FCF visibility that the membership fee stream provides — approximately $5B annually with minimal capex requirement. E-commerce penetration remains below 10%, creating a durable long-term growth channel at high incremental margins." },
      { name: "Sentiment NLP",       score: 79, justification: "Membership fee increase commentary — the first since 2017 — was met with positive analyst sentiment, with sell-side notes highlighting the historically low member churn following prior increases. Management language around international expansion into underserved markets carries elevated conviction in recent transcripts. The Kirkland Signature private label brand is increasingly cited as a moat-deepening asset, with NLP sentiment around it tracking at multi-year highs." },
      { name: "Tech-Momentum Net",   score: 80, justification: "COST has delivered consistent outperformance vs. the XRT retail ETF across multiple market environments, including the 2022 drawdown. Price is approaching the $890 all-time high level — a test that momentum models are monitoring closely for a potential breakout. RSI at 61 suggests momentum is present but not overextended. The long-term uptrend channel remains intact, with the 200-day SMA providing dynamic support well below current prices." },
    ],
  },
  HD: {
    score: 70,
    summary: "Housing turnover slowdown is a material near-term headwind to comparable store sales. Pro contractor segment provides resilience. SRS Distribution acquisition improves long-term margin mix. Patient investors should be rewarded when the rate cycle turns.",
    models: [
      { name: "DeepValue Engine",    score: 73, justification: "HD's P/E of 23× embeds a housing recovery scenario that has been deferred by the persistence of higher mortgage rates. The Pro segment, representing roughly 50% of revenue, provides meaningful insulation from DIY consumer weakness and is structurally growing its wallet share of the specialty trades market. FCF yield of 3.8% is the highest in 4 years and is fully covered by operations — a compelling valuation anchor for patient capital." },
      { name: "Sentiment NLP",       score: 68, justification: "Management commentary on housing turnover has maintained a cautious tone across the past four earnings calls, with language around 'deferred big-ticket projects' cited as a persistent headwind. The SRS Distribution acquisition is discussed with strategic conviction, but integration commentary is appropriately measured. Sell-side consensus has been trending slightly downward on comp sales estimates, though the rate-sensitive thesis makes expectations already fairly conservative." },
      { name: "Tech-Momentum Net",   score: 68, justification: "HD has underperformed the S&P 500 Consumer Discretionary sector by 8 percentage points over the trailing year, reflecting the overhang from housing market sensitivity. Price is range-bound between $350 and $420, lacking directional conviction. ADX of 14 confirms a non-trending, choppy environment. Momentum models have this name in a 'watch-list' rather than 'active-long' category — waiting for evidence of housing demand recovery before re-engaging." },
    ],
  },
};

// Per-ticker news feed — 4 items each: headline, source, time, sentiment (good|bad|neutral)
const STOCK_NEWS = {
  AAPL: [
    { headline: "Apple Intelligence EU rollout clears final regulatory hurdle, launches this quarter", source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "iPhone 17 supply chain data signals record Asia pre-orders ahead of September event", source: "Reuters",   time: "3h ago",  sentiment: "good"    },
    { headline: "Apple TV+ subscriber growth stalls as Netflix and Amazon dominate content spend",    source: "CNBC",      time: "6h ago",  sentiment: "bad"     },
    { headline: "Services segment on track for $100B annual run rate, well ahead of consensus",       source: "MS Research",time: "9h ago",  sentiment: "good"    },
  ],
  NVDA: [
    { headline: "Blackwell GPU allocation fully committed through Q3 2026 as hyperscaler orders surge",   source: "Bloomberg",     time: "30m ago", sentiment: "good"    },
    { headline: "New China export curbs could reduce NVIDIA data center revenue by up to 15%",           source: "Reuters",       time: "2h ago",  sentiment: "bad"     },
    { headline: "Microsoft Azure placing record GPU cluster order for 2025 infrastructure buildout",     source: "The Information",time: "4h ago",  sentiment: "good"    },
    { headline: "NVIDIA NVLink 5.0 unveiled at developer conference, doubling interconnect bandwidth",   source: "TechCrunch",    time: "6h ago",  sentiment: "good"    },
  ],
  MSFT: [
    { headline: "Azure cloud growth re-accelerates to 33% YoY, beating consensus by three percentage points", source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Copilot enterprise seat additions tracking 2× faster than internal forecast, sources say",   source: "The Information", time: "3h ago",  sentiment: "good"    },
    { headline: "EU regulators launch formal probe into Microsoft Teams bundling practices",                  source: "Reuters",   time: "5h ago",  sentiment: "bad"     },
    { headline: "OpenAI partnership structure limits third-party cloud monetization, analysts caution",       source: "FT",        time: "1d ago",  sentiment: "neutral" },
  ],
  TSLA: [
    { headline: "Tesla Q3 deliveries miss analyst estimates as aggressive price cuts fail to clear inventory", source: "Reuters",  time: "2h ago",  sentiment: "bad"     },
    { headline: "BYD and local rivals take record China EV market share at Tesla's direct expense",           source: "Bloomberg", time: "4h ago",  sentiment: "bad"     },
    { headline: "FSD v13 beta receives broadly positive tester feedback; NHTSA review remains pending",       source: "Electrek",  time: "6h ago",  sentiment: "good"    },
    { headline: "Tesla Megapack orders surge as utility-scale battery storage demand hits all-time high",     source: "CNBC",      time: "8h ago",  sentiment: "good"    },
  ],
  AMZN: [
    { headline: "AWS revenue growth accelerates for second straight quarter, reaching 19% YoY",           source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Amazon ad business surpasses $60B annual run rate, closing gap with Google at pace",     source: "Reuters",   time: "3h ago",  sentiment: "good"    },
    { headline: "Prime Video ad-tier churn spikes following price increase, disappointing management",    source: "WSJ",       time: "5h ago",  sentiment: "bad"     },
    { headline: "Amazon Pharmacy expansion to 50 cities ahead of schedule, One Medical synergies cited", source: "CNBC",      time: "8h ago",  sentiment: "neutral" },
  ],
  META: [
    { headline: "Meta AI assistant reaches 1B monthly active users across Facebook and Instagram",          source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Threads surpasses 300M DAU, directly threatening X's remaining advertiser relationships", source: "Reuters",   time: "3h ago",  sentiment: "good"    },
    { headline: "Reality Labs posts $4.5B quarterly operating loss; investor patience increasingly strained", source: "FT",      time: "5h ago",  sentiment: "bad"     },
    { headline: "Meta ad pricing rises 14% QoQ driven by improved AI-powered targeting performance",        source: "CNBC",     time: "7h ago",  sentiment: "good"    },
  ],
  GOOGL: [
    { headline: "Google Search holds 90% global share despite Bing AI launch — Similarweb data",    source: "Bloomberg", time: "2h ago",  sentiment: "good"    },
    { headline: "Google Cloud accelerates to 28% growth, narrowing gap with Azure and AWS meaningfully", source: "Reuters", time: "3h ago",  sentiment: "good"    },
    { headline: "DOJ antitrust ruling could force Google to divest its ad tech stack entirely",         source: "WSJ",      time: "5h ago",  sentiment: "bad"     },
    { headline: "YouTube Shorts CPMs now on par with long-form content for first time",                 source: "CNBC",     time: "8h ago",  sentiment: "good"    },
  ],
  'BRK.B': [
    { headline: "Berkshire cash reserves hit record $189B, signaling Buffett finds limited value at current prices", source: "Bloomberg", time: "2h ago",  sentiment: "neutral" },
    { headline: "BNSF Railway intermodal volumes recover 4% QoQ as freight markets stabilise",                      source: "Reuters",   time: "4h ago",  sentiment: "good"    },
    { headline: "Berkshire further reduces Apple stake; single-position concentration now below 40%",               source: "FT",        time: "6h ago",  sentiment: "neutral" },
    { headline: "GEICO combined loss ratio improves to best reading in six years after underwriting reforms",        source: "WSJ",       time: "1d ago",  sentiment: "good"    },
  ],
  V: [
    { headline: "Visa cross-border travel volume up 16% YoY as global tourism surpasses 2019 levels",  source: "Bloomberg", time: "1h ago",  sentiment: "good" },
    { headline: "Visa Value-Added Services revenue growing 3× faster than core network volumes",        source: "Reuters",   time: "3h ago",  sentiment: "good" },
    { headline: "DOJ files antitrust suit alleging Visa monopolised the US debit card market",          source: "WSJ",       time: "5h ago",  sentiment: "bad"  },
    { headline: "Tap-to-pay penetration in Southeast Asia accelerating Visa's volume growth thesis",    source: "CNBC",      time: "8h ago",  sentiment: "neutral" },
  ],
  UNH: [
    { headline: "UnitedHealth raises full-year guidance on stronger-than-expected Optum performance",          source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Congress proposes pharmacy benefit manager reform bill directly targeting Optum Rx unit",     source: "Reuters",   time: "3h ago",  sentiment: "bad"     },
    { headline: "2025 Medicare Advantage rate cuts come in below worst-case analyst scenario",                 source: "WSJ",       time: "5h ago",  sentiment: "neutral" },
    { headline: "Optum Health clinic expansion adds 200 locations in Q3, ahead of full-year build target",    source: "CNBC",      time: "7h ago",  sentiment: "good"    },
  ],
  JPM: [
    { headline: "JPMorgan Q3 net interest income beats estimates as loan margins hold above guidance",  source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Investment banking fees surge 35% YoY as M&A pipeline reopens across sectors",        source: "Reuters",   time: "3h ago",  sentiment: "good"    },
    { headline: "JPMorgan increases loan-loss provisions citing rising consumer delinquency rates",     source: "FT",        time: "5h ago",  sentiment: "bad"     },
    { headline: "Dimon warns of 'troubling times ahead' citing geopolitical and fiscal deficit risks", source: "CNBC",      time: "7h ago",  sentiment: "neutral" },
  ],
  JNJ: [
    { headline: "J&J Darzalex hits $10B annual sales milestone four years ahead of original forecast", source: "Bloomberg", time: "2h ago",  sentiment: "good" },
    { headline: "Talc bankruptcy settlement clears final legal hurdle, removing $10B liability overhang", source: "Reuters", time: "4h ago",  sentiment: "good" },
    { headline: "MedTech segment margins compress as hospital capex cycle remains cautious",            source: "FT",        time: "6h ago",  sentiment: "bad"  },
    { headline: "J&J raises dividend for 62nd consecutive year, reaffirming Dividend Aristocrat status", source: "WSJ",    time: "1d ago",  sentiment: "good" },
  ],
  WMT: [
    { headline: "Walmart+ membership tops 25M subscribers, accelerating toward Amazon Prime parity",    source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Walmart advertising revenue grows 26% YoY, cementing top-5 US digital ad position",   source: "Reuters",   time: "2h ago",  sentiment: "good"    },
    { headline: "Grocery price deflation compresses same-store sales growth below prior-year comparisons", source: "WSJ",   time: "5h ago",  sentiment: "neutral" },
    { headline: "E-commerce fulfillment cost-reduction initiative saving $500M annually, ahead of plan", source: "CNBC",   time: "7h ago",  sentiment: "good"    },
  ],
  XOM: [
    { headline: "ExxonMobil Permian output reaches record 1.2M bbl/day following Pioneer integration",  source: "Bloomberg", time: "2h ago",  sentiment: "good" },
    { headline: "Brent crude slides below $74 on OPEC+ supply increase signals, margin headwind cited", source: "Reuters",   time: "4h ago",  sentiment: "bad"  },
    { headline: "ExxonMobil carbon capture project receives $1.2B DOE funding approval",                source: "FT",        time: "6h ago",  sentiment: "good" },
    { headline: "Downstream refining margins tighten as global refinery capacity additions accelerate", source: "WSJ",       time: "8h ago",  sentiment: "bad"  },
  ],
  MA: [
    { headline: "Mastercard cross-border volumes up 18% YoY, beating all major consensus estimates",        source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Mastercard Move real-time payments platform now live across 140+ countries",               source: "Reuters",   time: "3h ago",  sentiment: "good"    },
    { headline: "DOJ antitrust probe of debit network parallels Visa suit; MA shares fall 3% in pre-market", source: "WSJ",    time: "5h ago",  sentiment: "bad"     },
    { headline: "Mastercard raises full-year net revenue growth guidance to 12-13% from 11-12%",            source: "CNBC",     time: "7h ago",  sentiment: "good"    },
  ],
  AVGO: [
    { headline: "Broadcom custom AI chip revenue from hyperscalers set to exceed $12B in FY2025",       source: "Bloomberg",      time: "1h ago",  sentiment: "good"    },
    { headline: "VMware integration ahead of schedule; $8.5B synergy target raised to $9B by management", source: "Reuters",     time: "3h ago",  sentiment: "good"    },
    { headline: "Broadcom leverage ratio concerns mount among fixed-income investors post-acquisition",  source: "FT",             time: "5h ago",  sentiment: "neutral" },
    { headline: "Google and Meta expand custom ASIC co-design partnerships with Broadcom through 2027", source: "The Information", time: "7h ago",  sentiment: "good"    },
  ],
  PG: [
    { headline: "P&G premium pricing sticks across categories despite consumer trade-down concerns",          source: "Bloomberg", time: "2h ago",  sentiment: "good"    },
    { headline: "Organic revenue growth moderates to 3% as volume recovery lags price contribution",         source: "Reuters",   time: "4h ago",  sentiment: "neutral" },
    { headline: "P&G increases quarterly dividend 5%, marking 68th consecutive annual increase",             source: "WSJ",       time: "6h ago",  sentiment: "good"    },
    { headline: "Asia and Africa volume recovery outpacing management forecast by a meaningful margin",       source: "CNBC",      time: "8h ago",  sentiment: "good"    },
  ],
  ORCL: [
    { headline: "Oracle cloud GPU bookings surge 78% QoQ as hyperscalers diversify away from AWS",       source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Remaining performance obligations hit $98B, doubling in just two years",                source: "Reuters",   time: "3h ago",  sentiment: "good"    },
    { headline: "Legacy on-premise database licensing decline accelerates, now -9% YoY vs -5% prior",   source: "FT",        time: "5h ago",  sentiment: "bad"     },
    { headline: "Oracle net debt-to-EBITDA at highest since 2010 acquisition cycle, limiting buybacks", source: "WSJ",       time: "8h ago",  sentiment: "bad"     },
  ],
  COST: [
    { headline: "Costco renewal rates hit all-time record of 93% globally in latest quarterly filing",   source: "Bloomberg", time: "1h ago",  sentiment: "good"    },
    { headline: "Costco gold bullion sales surpass $200M/month, disrupting traditional jewellery retail", source: "Reuters",  time: "3h ago",  sentiment: "good"    },
    { headline: "First membership fee increase in seven years draws mixed consumer reaction online",     source: "WSJ",       time: "5h ago",  sentiment: "neutral" },
    { headline: "International comparable sales growth of 8% significantly outpacing domestic segment", source: "CNBC",      time: "7h ago",  sentiment: "good"    },
  ],
  HD: [
    { headline: "Home Depot Pro segment grows 11% YoY as SRS Distribution integration delivers ahead of plan", source: "Bloomberg", time: "2h ago",  sentiment: "good" },
    { headline: "Existing home sales hit 14-year low, creating direct headwind for DIY remodeling spend",       source: "Reuters",   time: "4h ago",  sentiment: "bad"  },
    { headline: "Home Depot raises dividend 10%, signaling management confidence in multi-year outlook",        source: "WSJ",       time: "6h ago",  sentiment: "good" },
    { headline: "Comparable store sales down 3.2% as high mortgage rates suppress discretionary projects",     source: "CNBC",      time: "8h ago",  sentiment: "bad"  },
  ],
};

// StockLogo: Parqet API (symbol-based); onError → slate-800 circle fallback
// size: 'sm' (26px default), 'md' (40px), 'lg' (56px)
function StockLogo({ ticker, size = 'sm' }) {
  const [failed, setFailed] = useState(false);
  const symbol = (ticker || '').replace('.', '-');
  const sizeClass = size === 'lg' ? 'w-14 h-14 text-xl' : size === 'md' ? 'w-10 h-10 text-sm' : 'w-[26px] h-[26px] text-[11px]';
  if (failed || !ticker) {
    return (
      <span className={`flex items-center justify-center rounded-full bg-slate-800 font-bold text-slate-300 shrink-0 select-none uppercase ${sizeClass}`}>
        {ticker ? ticker.replace('.', '').slice(0, 1) : '?'}
      </span>
    );
  }
  return (
    <img
      src={`https://assets.parqet.com/logos/symbol/${encodeURIComponent(symbol)}?format=png`}
      alt=""
      aria-hidden="true"
      loading="lazy"
      className={`rounded-full object-cover bg-slate-800 shrink-0 ${sizeClass}`}
      onError={() => setFailed(true)}
    />
  );
}

// Deterministic chart data seeded from ticker chars so it never re-randomizes on re-render
function generateDetailChartData(stock, points = 90) {
  const seed = stock.ticker.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 1);
  const rng = (i) => Math.abs(Math.sin(seed * 0.001 + i * 1.618)) ;
  const base = stock.price;
  const bias = (stock.changePercent ?? 0) / 100;
  const origin = new Date(2024, 8, 15); // 90 days back approx
  return Array.from({ length: points }, (_, i) => {
    const t = i / (points - 1);
    const trend = base * (1 - bias + bias * t);
    const wave = base * (rng(i) - 0.5) * 0.06;
    const smooth = base * (rng(i * 0.3) - 0.5) * 0.025;
    const value = Math.max(stock.low52 * 0.9, +(trend + wave + smooth).toFixed(2));
    const d = new Date(origin);
    d.setDate(d.getDate() + i);
    return { date: `${d.getMonth() + 1}/${d.getDate()}`, value };
  });
}

// 30 realistic mock candles seeded from the ticker so they never flicker on re-render.
// Used whenever the live Finnhub candle call fails (rate limit, market closed, etc.).
function generateFallbackCandles(stock, points = 30) {
  const seed = stock.ticker.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 1);
  const rng = (i) => Math.abs(Math.sin(seed * 0.0013 + i * 1.618033));
  const base = stock.price ?? 100;
  const bias = (stock.changePercent ?? 0) / 100;
  const floor = (stock.low52 ?? base * 0.7) * 0.92;
  const fmt = new Intl.DateTimeFormat('en-US', { month: 'numeric', day: 'numeric' });
  const now = Date.now();
  return Array.from({ length: points }, (_, i) => {
    const t = i / (points - 1);
    const trend = base * (1 - bias * (1 - t));
    const wave   = base * (rng(i)       - 0.5) * 0.055;
    const smooth = base * (rng(i * 0.4) - 0.5) * 0.022;
    const value  = Math.max(floor, +(trend + wave + smooth).toFixed(2));
    const d = new Date(now - (points - 1 - i) * 24 * 60 * 60 * 1000);
    return { date: fmt.format(d), value };
  });
}

// Mini chart for expandable row: 150px height, gradient by gain/loss
function MiniStockChart({ stock }) {
  const isGain = (stock.changePercent ?? 0) >= 0;
  const price = stock.price ?? 0;
  const n = 24;
  const chartData = Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1);
    const drift = (stock.changePercent ?? 0) * 0.02 * (t - 0.5);
    const noise = (Math.sin(i * 0.7) * 0.02 + Math.cos(i * 0.3) * 0.015);
    return { day: i, value: price * (1 - 0.08 * (1 - t) + drift + noise) };
  });
  const color = isGain ? '#10b981' : '#ef4444';
  const colorMuted = isGain ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)';
  return (
    <div className="w-full h-[150px] rounded-lg overflow-hidden bg-slate-900/50 border border-slate-800/60">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 6, right: 6, left: 6, bottom: 6 }}>
          <defs>
            <linearGradient id={`miniFill-${stock.ticker}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.5} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="day" hide />
          <YAxis domain={['auto', 'auto']} hide />
          <Area type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} fill={`url(#miniFill-${stock.ticker})`} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// Bloomberg-style news intelligence feed with AI sentiment badges + pulse animations
function MarketIntelligence({ ticker }) {
  const items = STOCK_NEWS[ticker] ?? [];
  if (items.length === 0) return null;

  return (
    <div className={`${CARD_BASE} p-5 mb-6`}>
      <style>{`
        @keyframes intel-good-pulse {
          0%, 100% { border-color: rgba(16,185,129,0.22); box-shadow: 0 0 0 0 rgba(16,185,129,0); }
          50%       { border-color: rgba(16,185,129,0.55); box-shadow: 0 0 18px 3px rgba(16,185,129,0.08); }
        }
        @keyframes intel-bad-pulse {
          0%, 100% { border-color: rgba(239,68,68,0.22); box-shadow: 0 0 0 0 rgba(239,68,68,0); }
          50%       { border-color: rgba(239,68,68,0.55); box-shadow: 0 0 18px 3px rgba(239,68,68,0.08); }
        }
      `}</style>

      {/* ── Section header ── */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <span className="relative flex h-2 w-2 shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-blue-400" />
          </span>
          <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-[0.18em]">Market Intelligence</h3>
        </div>
        <span className="text-[10px] font-medium text-gray-600 uppercase tracking-wider">AI Sentiment · Live Feed</span>
      </div>

      {/* ── News cards grid ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {items.map((item, idx) => {
          const isGood    = item.sentiment === 'good';
          const isBad     = item.sentiment === 'bad';
          const isNeutral = !isGood && !isBad;

          const Icon        = isGood ? TrendingUp : isBad ? TrendingDown : Minus;
          const accentColor = isGood ? '#10b981'  : isBad ? '#ef4444'    : '#475569';
          const bgTint      = isGood ? 'rgba(16,185,129,0.045)' : isBad ? 'rgba(239,68,68,0.045)' : 'rgba(15,17,28,0.6)';
          const borderBase  = isGood ? 'rgba(16,185,129,0.22)'  : isBad ? 'rgba(239,68,68,0.22)'  : 'rgba(51,65,85,0.45)';

          const badgeCls = isGood
            ? 'text-emerald-400 bg-emerald-400/10 border border-emerald-500/25'
            : isBad
              ? 'text-red-400 bg-red-400/10 border border-red-500/25'
              : 'text-slate-400 bg-slate-700/30 border border-slate-600/30';

          const badgeLabel = isGood ? 'Positive' : isBad ? 'Negative' : 'Neutral';

          const pulseAnim = isGood
            ? 'intel-good-pulse 3.8s ease-in-out infinite'
            : isBad
              ? 'intel-bad-pulse 3.8s ease-in-out infinite'
              : 'none';

          return (
            <div
              key={idx}
              className="relative rounded-xl border flex flex-col gap-2.5 p-4 overflow-hidden"
              style={{ background: bgTint, borderColor: borderBase, animation: pulseAnim }}
            >
              {/* Left accent stripe */}
              <div
                className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-xl"
                style={{ background: `linear-gradient(to bottom, ${accentColor}cc, ${accentColor}44)` }}
              />

              {/* Sentiment badge */}
              <div className={`inline-flex items-center gap-1.5 self-start text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full shrink-0 ${badgeCls}`}>
                <Icon className="w-3 h-3 shrink-0" />
                {badgeLabel}
              </div>

              {/* Headline */}
              <p className="text-[13px] text-gray-100 leading-snug font-medium pl-0.5 flex-1">
                {item.headline}
              </p>

              {/* Source · Time */}
              <div className="flex items-center gap-1.5 pt-1 border-t border-slate-800/60">
                <span className="text-[11px] font-semibold text-gray-500">{item.source}</span>
                <span className="text-slate-700 text-[10px] select-none">·</span>
                <span className="text-[11px] text-gray-600 tabular-nums">{item.time}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// How-it-works copy for each AI model (shown in tooltip on info icon hover)
const MODEL_DESCRIPTIONS = {
  'DeepValue Engine': 'Analyzes FCF yields, buybacks, and intrinsic valuation vs. peers.',
  'Sentiment NLP': 'Processes analyst notes and news via NLP to score market sentiment and identify mean-reverting pessimism.',
  'Tech-Momentum Net': 'Quantitative engine tracking ADX trend strength, MACD, and relative performance vs. sector ETFs.',
};

// Multi-Model Consensus panel — replaces the arc gauge
// Renders one card per AI model, then keeps the Factor Breakdown bars below.
function AIRatingGauge({ ticker, score, models = [] }) {
  const clamped = Math.min(100, Math.max(0, score));
  const [tooltipModel, setTooltipModel] = useState(null);

  // Deterministic factor scores seeded from ticker so they never re-randomize
  const seed = ticker.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 7);
  const rng  = (i) => Math.abs(Math.sin(seed * 0.007 + i * 2.3));
  const factors = [
    { label: 'Growth',   score: Math.min(99, Math.max(10, Math.round(clamped * 0.88 + rng(1) * 22 - 11))) },
    { label: 'Value',    score: Math.min(99, Math.max(10, Math.round(clamped * 0.82 + rng(2) * 26 - 13))) },
    { label: 'Momentum', score: Math.min(99, Math.max(10, Math.round(clamped * 1.06 + rng(3) * 16 - 8)))  },
    { label: 'Quality',  score: Math.min(99, Math.max(10, Math.round(clamped * 0.98 + rng(4) * 14 - 7)))  },
  ];

  return (
    <div className={`${CARD_BASE} p-6 flex flex-col h-full`}>

      {/* ── Section header ── */}
      <div className="flex items-center justify-between mb-5">
        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em]">Multi-Model Consensus</p>
        <span className="text-[10px] text-gray-600 tabular-nums">{models.length} models</span>
      </div>

      {/* ── Model Cards ── */}
      <div className="space-y-3 flex-1">
        {models.map((model, idx) => {
          const sig = getSignal(model.score);
          return (
            <div
              key={model.name}
              className="rounded-lg border border-slate-800/70 bg-slate-900/40 overflow-visible transition-opacity duration-300"
              style={{ borderLeftWidth: '3px', borderLeftColor: sig.hex }}
            >
              <div className="px-4 pt-3 pb-2">
                {/* Name + badge row */}
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div
                    className="flex items-center gap-1.5 min-w-0 relative"
                    onMouseEnter={() => setTooltipModel(model.name)}
                    onMouseLeave={() => setTooltipModel(null)}
                  >
                    <span className="text-[11px] font-bold text-white uppercase tracking-wider truncate">
                      {model.name}
                    </span>
                    <Info className="w-3.5 h-3.5 text-gray-500 shrink-0 hover:text-gray-400 transition-colors" aria-label="How this model works" />
                    {tooltipModel === model.name && MODEL_DESCRIPTIONS[model.name] && (
                      <div
                        className="absolute bottom-full left-0 mb-1.5 z-50 w-64 px-3 py-2.5 rounded-lg text-xs text-slate-300 leading-relaxed shadow-xl"
                        style={{
                          background: 'rgba(15, 23, 42, 0.95)',
                          border: '1px solid rgba(71, 85, 105, 0.4)',
                          backdropFilter: 'blur(12px)',
                        }}
                      >
                        {MODEL_DESCRIPTIONS[model.name]}
                      </div>
                    )}
                  </div>
                  <span
                    className={`inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full border shrink-0 ${sig.textCls} ${sig.bgCls} ${sig.borderCls}`}
                  >
                    <sig.Icon className="w-3 h-3" />
                    {sig.label} · {model.score}
                  </span>
                </div>
                {/* Justification */}
                <p className="text-sm text-slate-400 leading-relaxed">{model.justification}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Factor Breakdown ── */}
      <div className="mt-5 pt-4 border-t border-slate-800/60">
        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-600 mb-3">Factor Breakdown</p>
        <div className="space-y-2.5">
          {factors.map(({ label: fl, score: fs }) => {
            const fc = fs >= 70 ? '#10b981' : fs >= 48 ? '#eab308' : '#ef4444';
            return (
              <div key={fl} className="flex items-center gap-2.5">
                <span className="text-[11px] text-gray-500 w-[4.5rem] shrink-0">{fl}</span>
                <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{ backgroundColor: fc, width: `${fs}%` }}
                  />
                </div>
                <span className="text-[11px] text-gray-400 tabular-nums w-5 text-right shrink-0">{fs}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

const CHART_TIME_RANGE_OPTIONS = [
  { value: '1d', label: '1D' },
  { value: '1mo', label: '1M' },
  { value: '3mo', label: '3M' },
  { value: '6mo', label: '6M' },
  { value: '1y', label: '1Y' },
];

function formatUsdPortfolio(n) {
  if (n == null || !Number.isFinite(Number(n))) return '—';
  return Number(n).toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Demo backtest card; fetches `/api/portfolio/simulate` for the active ticker and chart time range. */
function DemoPortfolioSimulator({ ticker, timeRange, accentColor }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const url = `${API_BASE_URL}/api/portfolio/simulate?ticker=${encodeURIComponent(ticker)}&time_range=${encodeURIComponent(timeRange)}`;
        const res = await fetch(url);
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(errText || `Request failed (${res.status})`);
        }
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) {
          setError(e?.message || 'Failed to load simulation');
          setData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [ticker, timeRange]);

  const initialBal = Number.isFinite(Number(data?.initial_balance)) ? Number(data.initial_balance) : 10_000;
  const finalBal = Number.isFinite(Number(data?.final_balance)) ? Number(data.final_balance) : null;
  const roi = Number.isFinite(Number(data?.total_roi_pct)) ? Number(data.total_roi_pct) : null;
  const trades = Array.isArray(data?.trades) ? data.trades : [];

  return (
    <div className={`${CARD_BASE} p-6 mb-6`}>
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.18em]">
          Mock Portfolio (Algo Trading)
        </h3>
        <p className="text-[11px] text-gray-600 mt-1.5 leading-relaxed">
          Hypothetical $10k all-in backtest for{' '}
          <span className="text-gray-400 font-medium tabular-nums">{ticker}</span> using the same range as the price chart (
          <span className="text-gray-400 uppercase">{timeRange}</span>). For illustration only.
        </p>
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-14 gap-4">
          <div className="relative flex items-center justify-center">
            <span
              className="absolute inline-flex h-10 w-10 rounded-full opacity-30 animate-ping"
              style={{ backgroundColor: accentColor }}
            />
            <span
              className="relative inline-flex h-5 w-5 rounded-full"
              style={{ backgroundColor: accentColor }}
            />
          </div>
          <p className="text-sm font-medium tracking-widest uppercase animate-pulse" style={{ color: accentColor }}>
            Running simulation…
          </p>
        </div>
      )}

      {!loading && error && <p className="text-sm text-red-400/90 py-6 text-center">{error}</p>}

      {!loading && !error && data && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5">
            <div className="rounded-lg border border-slate-800/70 bg-slate-900/35 px-4 py-3">
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">Initial Balance</p>
              <p className="text-lg font-bold text-white tabular-nums">{formatUsdPortfolio(initialBal)}</p>
            </div>
            <div className="rounded-lg border border-slate-800/70 bg-slate-900/35 px-4 py-3">
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">Final Balance</p>
              <p className="text-lg font-bold text-white tabular-nums">{formatUsdPortfolio(finalBal)}</p>
            </div>
            <div className="rounded-lg border border-slate-800/70 bg-slate-900/35 px-4 py-3">
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">Total ROI</p>
              <p
                className={`text-lg font-bold tabular-nums ${
                  roi == null ? 'text-gray-300' : roi > 0 ? 'text-emerald-400' : roi < 0 ? 'text-red-400' : 'text-gray-300'
                }`}
              >
                {roi == null ? '—' : `${roi > 0 ? '+' : ''}${roi.toFixed(2)}%`}
              </p>
            </div>
          </div>

          <div className="border-t border-slate-800/60 pt-4">
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em] mb-3">Trade history</p>
            <div className="rounded-lg border border-slate-800/70 overflow-hidden bg-[#0f1118] max-h-[220px] overflow-y-auto">
              <table className="w-full text-left border-collapse text-sm" aria-label="Simulated trades">
                <thead className="sticky top-0 bg-[#0f1118] border-b border-slate-800 z-[1]">
                  <tr>
                    <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Date</th>
                    <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Action</th>
                    <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">
                      Execution Price
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {trades.length === 0 ? (
                    <tr>
                      <td colSpan={3} className="py-8 px-3 text-center text-xs text-gray-500">
                        No trades in this window (signal stayed neutral or window too short).
                      </td>
                    </tr>
                  ) : (
                    trades.map((row, idx) => {
                      const act = String(row.action || '').toUpperCase();
                      const isBuy = act === 'BUY';
                      return (
                        <tr
                          key={`${row.date}-${idx}`}
                          className="border-b border-slate-800/40 last:border-0 hover:bg-white/[0.03]"
                        >
                          <td className="py-2.5 px-3 text-gray-300 tabular-nums text-xs">{row.date}</td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`inline-flex items-center text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-md border ${
                                isBuy
                                  ? 'text-emerald-400 bg-emerald-400/10 border-emerald-500/30'
                                  : 'text-red-400 bg-red-400/10 border-red-500/30'
                              }`}
                            >
                              {act || '—'}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-gray-200 tabular-nums">
                            {formatUsdPortfolio(Number(row.execution_price))}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StockDetailPage({ stock, onBack, onInitialSyncComplete, isInWatchlist, onToggleWatchlist }) {
  const [currentPrice, setCurrentPrice] = useState(stock.price);
  const [priceChange, setPriceChange] = useState(
    typeof stock.changePercent === 'number' ? ((stock.changePercent / 100) * stock.price) : 0,
  );
  const [percentageChange, setPercentageChange] = useState(stock.changePercent ?? 0);
  const [lastSynced, setLastSynced] = useState(null);
  const [fullChartData, setFullChartData] = useState([]);
  const [isChartSyncing, setIsChartSyncing] = useState(false);
  const [isFallback, setIsFallback] = useState(false);
  const [lastChartSynced, setLastChartSynced] = useState(null);
  const [chartTimeRange, setChartTimeRange] = useState('3mo');
  const [showConsensusExplainer, setShowConsensusExplainer] = useState(false);
  const initialSyncNotifiedRef = useRef(false);
  const isGain = (percentageChange ?? 0) >= 0;
  // `color` drives the price-chart gradient and the About card left-border accent (price direction)
  const color = isGain ? '#10b981' : '#ef4444';
  const gradId = `detailFill-${stock.ticker}`;
  const about = STOCK_ABOUT[stock.ticker] || `${stock.name} is a publicly traded company listed on US equity markets.`;

  const aiRating = STOCK_AI_RATINGS[stock.ticker] ?? {
    score: 70,
    summary: `${stock.name} presents a balanced risk-reward profile based on current market conditions and fundamental analysis.`,
  };

  // Derive signal purely from aiRating.score — guaranteed to match the AIRatingGauge
  const { label: signalLabel, textCls: signalText, bgCls: signalBg, borderCls: signalBorder, Icon: SignalIcon } = getSignal(aiRating.score);
  const signalBadgeCls = `${signalText} ${signalBg} ${signalBorder}`;

  // Hero header tint — keyed to AI signal, NOT price direction
  const HEADER_PALETTE = {
    'Strong Buy': { bg: 'rgba(34,197,94,0.07)',  border: 'rgba(34,197,94,0.28)',  glow: 'rgba(34,197,94,0.10)'  },
    'Buy':        { bg: 'rgba(16,185,129,0.07)', border: 'rgba(16,185,129,0.25)', glow: 'rgba(16,185,129,0.10)' },
    'Hold':       { bg: 'rgba(234,179,8,0.07)',  border: 'rgba(234,179,8,0.28)',  glow: 'rgba(234,179,8,0.09)'  },
    'Sell':       { bg: 'rgba(239,68,68,0.07)',  border: 'rgba(239,68,68,0.28)',  glow: 'rgba(239,68,68,0.10)'  },
  };
  const hp = HEADER_PALETTE[signalLabel] ?? HEADER_PALETTE['Hold'];

  const stats = [
    { label: 'Market Cap',  value: `$${stock.marketCap?.toLocaleString()}B` },
    { label: 'P/E Ratio',   value: stock.pe ? `${stock.pe.toFixed(1)}×` : '—' },
    { label: '52W High',    value: `$${stock.high52?.toFixed(2)}` },
    { label: 'Div. Yield',  value: stock.dividendYield ? `${stock.dividendYield.toFixed(2)}%` : '—' },
  ];

  useEffect(() => {
    initialSyncNotifiedRef.current = false;
  }, [stock.ticker]);

  useEffect(() => {
    let cancelled = false;
    const fetchQuote = async () => {
      try {
        const quote = await fetchLiveQuote(stock.ticker);
        if (!quote || cancelled) return;
        setCurrentPrice(quote.price);
        setPriceChange(quote.change);
        setPercentageChange(quote.changePercent);
        setLastSynced(Date.now());
      } finally {
        if (!cancelled && !initialSyncNotifiedRef.current) {
          onInitialSyncComplete?.();
          initialSyncNotifiedRef.current = true;
        }
      }
    };
    fetchQuote();
    const id = setInterval(fetchQuote, LIVE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [stock.ticker]);

  // Fetch chart data from FastAPI (/api/analyze) — historical prices only (no forecast UI)
  useEffect(() => {
    let cancelled = false;
    setFullChartData([]);
    setIsFallback(false);
    const fetchChart = async () => {
      setIsChartSyncing(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 20_000);
      try {
        const url = `${API_BASE_URL}/api/analyze?ticker=${encodeURIComponent(stock.ticker)}&time_range=${encodeURIComponent(chartTimeRange)}`;
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`API returned ${res.status} ${res.statusText}`);
        const json = await res.json();
        if (!cancelled) {
          const raw = Array.isArray(json.chart_data) ? json.chart_data : [];
          const chartData = raw
            .filter((d) => d != null && d.price != null)
            .map((d) => ({ date: d.date, price: d.price }));
          setFullChartData(chartData);
          setIsFallback(false);
          setLastChartSynced(Date.now());
        }
      } catch (err) {
        console.error('[GlobalMarketPredictor] FastAPI chart fetch error:', err);
        if (!cancelled) {
          setFullChartData(
            generateFallbackCandles(stock).map((d) => ({
              date: d.date, price: d.value,
            })),
          );
          setIsFallback(true);
        }
      } finally {
        clearTimeout(timeoutId);
        if (!cancelled) setIsChartSyncing(false);
      }
    };
    fetchChart();
    return () => { cancelled = true; };
  }, [stock.ticker, chartTimeRange]);

  const displayData = useMemo(() => fullChartData, [fullChartData]);

  return (
    <div
      key={`detail-${stock.ticker}`}
      className="w-full min-h-full py-8 px-10 box-border animate-fadeIn"
      style={{ fontFamily: 'Inter, system-ui, sans-serif' }}
    >
      {/* Back */}
      <button
        onClick={onBack}
        className="group inline-flex items-center gap-2 text-sm text-gray-500 hover:text-white transition-colors mb-8"
      >
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
        Back to Dashboard
      </button>

      {/* ── Hero header — background/border/glow all driven by AI signal ── */}
      <div
        className="rounded-2xl border p-6 mb-6 flex flex-col sm:flex-row sm:items-center gap-5 transition-all duration-500"
        style={{
          background:  hp.bg,
          borderColor: hp.border,
          boxShadow:   `0 0 48px ${hp.glow}, 0 0 96px ${hp.glow}`,
        }}
      >
        <StockLogo ticker={stock.ticker} size="lg" />
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-3 mb-1">
            <h1 className="text-4xl font-extrabold text-white tracking-tight">{stock.ticker}</h1>
            <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${signalBadgeCls}`}>
              <SignalIcon className="w-3.5 h-3.5 shrink-0" />{signalLabel}
            </span>
            {onToggleWatchlist && (
              <button
                type="button"
                onClick={() => onToggleWatchlist(stock.ticker)}
                className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border transition-colors ${
                  isInWatchlist
                    ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
                    : 'border-slate-600 bg-slate-800/40 text-gray-400 hover:text-white hover:border-slate-500'
                }`}
              >
                <Star className={`w-3.5 h-3.5 shrink-0 ${isInWatchlist ? 'fill-amber-400 text-amber-400' : ''}`} />
                {isInWatchlist ? 'Watching' : 'Add to watchlist'}
              </button>
            )}
          </div>
          <p className="text-gray-400 text-base">{stock.name}</p>
        </div>
        <div className="text-right shrink-0">
          <div className="flex items-center justify-end gap-2">
            <span className="relative flex h-2 w-2 shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            </span>
            <p className="text-4xl font-bold text-white tabular-nums">${Number(currentPrice ?? 0).toFixed(2)}</p>
          </div>
          <p className={`text-xl font-semibold mt-1 tabular-nums ${isGain ? 'text-emerald-400' : 'text-red-400'}`}>
            {isGain ? '+' : ''}{Number(priceChange ?? 0).toFixed(2)} ({isGain ? '+' : ''}{Number(percentageChange ?? 0).toFixed(2)}%)
          </p>
          <p className="text-xs text-gray-500 mt-1 uppercase tracking-wider">Today's Change</p>
        </div>
      </div>
      <div className="mt-[-1.25rem] mb-6 rounded-b-2xl border border-t-0 border-slate-800/60 px-6 py-3 bg-[#12131a]/45">
        <LiveIndicator lastSynced={lastSynced} />
      </div>

      {/* ── Market Intelligence — AI news feed, below header, above chart ── */}
      <MarketIntelligence ticker={stock.ticker} />

      {/* ── Main chart ── */}
      <div className={`${CARD_BASE} p-6 mb-6`}>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.18em]">
            Price History (Live)
          </h3>
          <div className="flex flex-wrap items-center gap-2 sm:justify-end">
            {CHART_TIME_RANGE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setChartTimeRange(opt.value)}
                className={`text-[10px] font-semibold uppercase tracking-wider px-2.5 py-1 rounded-md border transition-colors ${
                  chartTimeRange === opt.value
                    ? 'border-emerald-500/45 bg-emerald-500/10 text-emerald-300'
                    : 'border-slate-700/60 bg-slate-900/40 text-gray-500 hover:text-gray-300 hover:border-slate-600'
                }`}
              >
                {opt.label}
              </button>
            ))}
            {isChartSyncing && fullChartData.length > 0 && (
              <span className="text-[10px] text-emerald-400 animate-pulse uppercase tracking-wider">Syncing...</span>
            )}
            {isFallback && !isChartSyncing && (
              <span className="text-[10px] text-gray-600 uppercase tracking-wider border border-slate-700/60 rounded px-1.5 py-0.5">
                Simulated
              </span>
            )}
            {displayData.length > 0 && (
              <span className="text-xs text-gray-600 tabular-nums">
                {displayData[0]?.date} – {displayData[displayData.length - 1]?.date}
              </span>
            )}
          </div>
        </div>

        <div className="h-[300px]">
          {/* ── Loading state (first fetch) ── */}
          {(isChartSyncing && fullChartData.length === 0) && (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <div className="relative flex items-center justify-center">
                <span
                  className="absolute inline-flex h-10 w-10 rounded-full opacity-30 animate-ping"
                  style={{ backgroundColor: color }}
                />
                <span
                  className="relative inline-flex h-5 w-5 rounded-full"
                  style={{ backgroundColor: color }}
                />
              </div>
              <p className="text-sm font-medium tracking-widest uppercase animate-pulse" style={{ color }}>
                Fetching Market Data…
              </p>
              <p className="text-[11px] text-gray-600">Loading chart data for {stock.ticker}</p>
            </div>
          )}

          {/* ── Chart (historical) ── */}
          {displayData.length > 0 && (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={displayData} margin={{ top: 8, right: 8, left: 8, bottom: 4 }}>
                <defs>
                  <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor={color} stopOpacity={0.45} />
                    <stop offset="65%"  stopColor={color} stopOpacity={0.12} />
                    <stop offset="100%" stopColor={color} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#6b7280', fontSize: 11 }}
                  stroke="rgba(255,255,255,0.04)"
                  minTickGap={28}
                />
                <YAxis
                  domain={['auto', 'auto']}
                  tick={{ fill: '#6b7280', fontSize: 11 }}
                  stroke="rgba(255,255,255,0.04)"
                  width={64}
                  tickFormatter={(v) => `$${v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v.toFixed(0)}`}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    const hist = payload.find((p) => p.dataKey === 'price')?.value;
                    return (
                      <div style={{
                        backgroundColor: 'rgba(12,15,24,0.95)',
                        border: '1px solid rgba(71,85,105,0.5)',
                        borderRadius: '10px',
                        backdropFilter: 'blur(12px)',
                        padding: '8px 14px',
                      }}>
                        <p style={{ color: '#9ca3af', fontSize: 11, marginBottom: 4 }}>{label}</p>
                        {hist != null && (
                          <p style={{ color, fontSize: 13, fontWeight: 600 }}>{stock.ticker}: ${Number(hist).toFixed(2)}</p>
                        )}
                      </div>
                    );
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke={color}
                  strokeWidth={2}
                  fill={`url(#${gradId})`}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 0, fill: color }}
                  connectNulls={true}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
        <div className="mt-4 pt-3 border-t border-slate-800/60">
          <LiveIndicator lastSynced={lastChartSynced ?? lastSynced} />
        </div>
      </div>

      <DemoPortfolioSimulator ticker={stock.ticker} timeRange={chartTimeRange} accentColor={color} />

      {/* ── 4-column stats grid ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        {stats.map(({ label, value }) => (
          <div key={label} className={`${CARD_BASE} p-5`}>
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em] mb-2">{label}</p>
            <p className="text-xl font-bold text-white tabular-nums">{value}</p>
          </div>
        ))}
      </div>

      {/* ── AI Rating (2/5) + About the Company (3/5) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

        {/* AI Rating panel */}
        <div className="lg:col-span-2">
          <AIRatingGauge
            ticker={stock.ticker}
            score={aiRating.score}
            models={aiRating.models ?? []}
          />
        </div>

        {/* About */}
        <div
          className={`${CARD_BASE} p-6 lg:col-span-3 flex flex-col`}
          style={{ borderLeftWidth: '3px', borderLeftColor: color }}
        >
          <h3 className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em] mb-4">About the Company</h3>
          <p className="text-sm text-gray-300 leading-relaxed flex-1">{about}</p>

          {/* Quick signal row — derives from same getSignal(aiRating.score) as header badge */}
          <div className="mt-6 pt-4 border-t border-slate-800/60 flex flex-wrap items-center gap-3">
            <span className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider">AI Signal</span>
            <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${signalBadgeCls}`}>
              <SignalIcon className="w-3.5 h-3.5 shrink-0" />{signalLabel}
            </span>
            <span className="text-[10px] text-gray-500 tabular-nums">
              AI Score: <span className="text-gray-300 font-semibold tabular-nums">{aiRating.score}</span>
            </span>
          </div>
        </div>
      </div>

      {/* ── Behind the Scenes: How Our AI Models Work ── */}
      <div className="w-full mt-8">
        <button
          type="button"
          onClick={() => setShowConsensusExplainer((v) => !v)}
          className="w-full py-4 px-5 rounded-xl border border-slate-700/60 flex items-center justify-center gap-2 text-sm font-semibold text-slate-300 hover:text-white hover:border-emerald-500/40 transition-all duration-300"
          style={{
            background: 'rgba(15, 23, 42, 0.6)',
            boxShadow: showConsensusExplainer ? 'none' : '0 0 32px rgba(16, 185, 129, 0.12), 0 0 64px rgba(16, 185, 129, 0.06)',
          }}
        >
          <Info className="w-4 h-4 text-emerald-400/90 shrink-0" />
          Learn How the consensus is calculated
        </button>

        <div
          className="overflow-hidden transition-all duration-500 ease-out"
          style={{
            maxHeight: showConsensusExplainer ? '2000px' : '0',
            opacity: showConsensusExplainer ? 1 : 0,
          }}
        >
          <div
            className="mt-4 p-6 rounded-xl border border-slate-800/70"
            style={{
              background: 'rgba(15, 23, 42, 0.5)',
              backdropFilter: 'blur(12px)',
            }}
          >
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.2em] mb-6">Behind the Scenes: How Our AI Models Work</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
              {/* Column 1: DeepValue Engine */}
              <div className="rounded-lg border border-slate-800/60 p-5 bg-slate-900/30">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl" aria-hidden>⚖️</span>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider">1. DeepValue Engine</h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Analyzes intrinsic valuation. It monitors Free Cash Flow (FCF) yields, segment-level earnings catalysts, and capital return programs (buybacks). Compares current multiples vs. historical averages and industry peers.
                </p>
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Intrinsic Value vs. Price (Mock)</p>
                <div className="h-24">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={[{ n: 'T-4', iv: 82, p: 88 }, { n: 'T-3', iv: 85, p: 90 }, { n: 'T-2', iv: 88, p: 92 }, { n: 'T-1', iv: 90, p: 94 }, { n: 'T', iv: 92, p: 96 }]} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
                      <defs>
                        <linearGradient id={`behind-iv-${stock.ticker}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                          <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="n" tick={{ fontSize: 9, fill: '#64748b' }} />
                      <YAxis hide domain={['auto', 'auto']} />
                      <Area type="monotone" dataKey="iv" stroke="#10b981" fill={`url(#behind-iv-${stock.ticker})`} strokeWidth={1.5} />
                      <Line type="monotone" dataKey="p" stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Column 2: Sentiment NLP */}
              <div className="rounded-lg border border-slate-800/60 p-5 bg-slate-900/30">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl" aria-hidden>🧠</span>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider">2. Sentiment NLP</h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Uses Natural Language Processing to gauge market mood. Processes analyst notes, news headlines, and agency channel checks. Scores sentiment to detect artificial pessimism or constructive shifts.
                </p>
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Sentiment Score Flow (Positive/Negative Mock)</p>
                <div className="h-24">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[{ w: 'W1', pos: 62, neg: 38 }, { w: 'W2', pos: 58, neg: 42 }, { w: 'W3', pos: 71, neg: 29 }, { w: 'W4', pos: 68, neg: 32 }]} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
                      <XAxis dataKey="w" tick={{ fontSize: 9, fill: '#64748b' }} />
                      <YAxis hide domain={[0, 100]} />
                      <Bar dataKey="pos" fill="#10b981" fillOpacity={0.8} radius={[2, 2, 0, 0]} />
                      <Bar dataKey="neg" fill="#ef4444" fillOpacity={0.5} radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Column 3: Tech-Momentum Net */}
              <div className="rounded-lg border border-slate-800/60 p-5 bg-slate-900/30">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl" aria-hidden>📈</span>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider">3. Tech-Momentum Net</h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Quantitative technical engine. Tracks price action through ADX (trend strength), MACD (momentum), and support/resistance levels. Performs Relative Strength analysis vs. sector benchmarks (e.g., XLC ETF).
                </p>
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Momentum Indicators (MACD/ADX Mock)</p>
                <div className="h-24">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={[{ t: 'T-4', macd: 0.2, adx: 18 }, { t: 'T-3', macd: 0.5, adx: 22 }, { t: 'T-2', macd: 0.4, adx: 26 }, { t: 'T-1', macd: 0.7, adx: 28 }, { t: 'T', macd: 0.6, adx: 30 }]} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
                      <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#64748b' }} />
                      <YAxis hide domain={['auto', 'auto']} />
                      <Line type="monotone" dataKey="macd" stroke="#10b981" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="adx" stroke="#eab308" strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="pt-5 border-t border-slate-800/60">
              <p className="text-sm text-slate-400 leading-relaxed">
                <span className="font-semibold text-slate-300">The Consensus:</span> Our system compiles these three distinct signals. A &apos;BUY&apos; is generated only when a majority (2 out of 3) models align on a positive outlook, ensuring a robust, diversified investment signal.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatVolume(v) {
  if (v == null || v === undefined) return '—';
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(v);
}

function getSentimentLabelAndColor(value) {
  const v = Math.min(100, Math.max(0, value));
  if (v <= 24) return { label: 'Extreme Fear', color: '#ef4444', zone: 'extremeFear' };
  if (v <= 44) return { label: 'Fear', color: '#f97316', zone: 'fear' };
  if (v <= 55) return { label: 'Neutral', color: '#eab308', zone: 'neutral' };
  if (v <= 75) return { label: 'Greed', color: '#84cc16', zone: 'greed' };
  return { label: 'Extreme Greed', color: '#22c55e', zone: 'extremeGreed' };
}

function getSentimentDetails(score) {
  const { label, color } = getSentimentLabelAndColor(score);
  return { label, color };
}

// Spec: trader-focused actionable insight per zone (one sentence below score)
const SENTIMENT_INSIGHTS = {
  extremeFear: 'Market is oversold. Contrarian buying opportunity often identified.',
  fear: 'Risk-off sentiment. Consider defensive positioning or selective entries.',
  neutral: 'Sentiment balanced. Range-bound conditions typical.',
  greed: 'Risk-on. Stay selective with new entries; consider trimming extremes.',
  extremeGreed: 'Euphoria. Consider taking some risk off the table.',
};

// Zone-specific glow for dynamic box-shadow (spec: deep red → green by zone)
const ZONE_GLOW = {
  extremeFear: { color: '#991b1b', blur: 48, spread: 0, alpha: 0.35 },
  fear:         { color: '#c2410c', blur: 40, spread: 0, alpha: 0.28 },
  neutral:      { color: '#eab308', blur: 32, spread: 0, alpha: 0.18 },
  greed:        { color: '#65a30d', blur: 40, spread: 0, alpha: 0.28 },
  extremeGreed: { color: '#166534', blur: 48, spread: 0, alpha: 0.35 },
};

function hexToRgba(hex, alpha) {
  const match = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
  if (!match) return `rgba(0,0,0,${alpha})`;
  return `rgba(${parseInt(match[1], 16)}, ${parseInt(match[2], 16)}, ${parseInt(match[3], 16)}, ${alpha})`;
}

// Needle: pivot at arc center (50,50). rotation = (sentimentScore/100)*180 - 90 → 0=left, 50=top, 100=right.
function SentimentGauge({ value, historical = [] }) {
  const [hoveredScore, setHoveredScore] = useState(null);
  const displayScore = hoveredScore !== null ? hoveredScore : 22;
  const rotation = (displayScore / 100) * 180 - 90;

  const { label: currentLabel, color: currentColor } = getSentimentDetails(displayScore);
  const { zone } = getSentimentLabelAndColor(displayScore);
  const insightText = SENTIMENT_INSIGHTS[zone] || SENTIMENT_INSIGHTS.neutral;

  // Spec 2.3: exactly 3 rows — Current, 1 Week Ago, 1 Month Ago (dummy data if no API)
  const defaultTrend = [
    { period: 'Current', periodKey: 'Current', score: 22, value: 22, label: currentLabel },
    { period: '1 Week Ago', periodKey: '1W', score: 28, value: 28, label: 'Fear' },
    { period: '1 Month Ago', periodKey: '1M', score: 45, value: 45, label: 'Neutral' },
  ];
  const trendData = historical.length >= 3
    ? [
        { period: 'Current', periodKey: 'Current', score: 22, value: 22, label: currentLabel },
        { period: '1 Week Ago', periodKey: '1W', score: historical[1]?.value ?? 28, value: historical[1]?.value ?? 28, label: getSentimentLabelAndColor(historical[1]?.value ?? 28).label },
        { period: '1 Month Ago', periodKey: '1M', score: historical[2]?.value ?? 45, value: historical[2]?.value ?? 45, label: getSentimentLabelAndColor(historical[2]?.value ?? 45).label },
      ]
    : defaultTrend;

  const glowConfig = ZONE_GLOW[zone] || ZONE_GLOW.neutral;
  const glowRgba = hexToRgba(glowConfig.color, glowConfig.alpha);
  const glowRgbaPulse = hexToRgba(glowConfig.color, glowConfig.alpha * 0.4);
  const scoreGlow = hexToRgba(currentColor, 0.28);

  return (
    <div
      className={`relative flex flex-col items-center w-full rounded-2xl p-5 box-border overflow-hidden ${CARD_BASE}`}
      style={{
        fontFamily: 'Inter, system-ui, sans-serif',
        height: 'auto',
        boxShadow: `inset 0 0 24px rgba(0,0,0,0.25), 0 0 ${glowConfig.blur}px ${glowRgba}`,
      }}
      aria-label="US Market Sentiment gauge"
    >
      <div
        className="absolute inset-0 rounded-2xl pointer-events-none"
        style={{
          boxShadow: `inset 0 0 ${glowConfig.blur}px ${glowRgbaPulse}`,
          animation: 'sentiment-pulse 3s ease-in-out infinite',
        }}
      />
      <style>{`
        @keyframes sentiment-pulse {
          0%, 100% { opacity: 0.55; }
          50%       { opacity: 1; }
        }
      `}</style>

      <div className="w-full max-w-[280px]">
        {/*
          WHY transform-box:view-box + 50% 100% is correct:
          CSS "50px 50px" = screen pixels, NOT viewBox units. When the SVG scales
          to fill its container, 50px ≠ viewBox-50, so the pivot drifts.
          Fix: transform-box:view-box makes transform-origin use the SVG viewport.
          viewBox "0 0 100 50" → 50% of width(100) = 50, 100% of height(50) = 50.
          So "50% 100%" = viewBox point (50, 50) at any rendered size. ✓

          Needle math:
          · x1="50" y1="50" → starts at pivot (50, 50)
          · x2="50" y2="5"  → points straight UP before rotation
          · rotation = (sentimentScore / 100) * 180 − 90
            score  0 → −90° → LEFT  (Extreme Fear) ✓
            score 50 →   0° → UP    (Neutral)       ✓
            score 100 → +90° → RIGHT (Extreme Greed) ✓
            score 22  → −50.4° → far-left (Extreme Fear) ✓
        */}
        <svg viewBox="0 0 100 50" className="w-full block" preserveAspectRatio="xMidYMid meet" overflow="visible">
          <defs>
            <linearGradient id="sgEnamel" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#ef4444" stopOpacity={0.95} />
              <stop offset="50%"  stopColor="#eab308" stopOpacity={0.95} />
              <stop offset="100%" stopColor="#22c55e" stopOpacity={0.95} />
            </linearGradient>
          </defs>

          {/* Arc track and color band: semicircle center(50,50) r=50 */}
          <path d="M 0 50 A 50 50 0 0 1 100 50" fill="none" stroke="#1e293b"          strokeWidth="6"   strokeLinecap="round" />
          <path d="M 0 50 A 50 50 0 0 1 100 50" fill="none" stroke="url(#sgEnamel)" strokeWidth="4.5" strokeLinecap="round" opacity={0.92} />

          {/* LAYER 1 — NEEDLE (before pivot so hub covers the base joint)
              transform-box:view-box + origin "50% 100%" = pivot locked to (50,50) in viewBox coords */}
          <g
            style={{
              transformBox: 'view-box',
              transformOrigin: '50% 100%',
              transform: `rotate(${rotation}deg)`,
              transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          >
            <line x1="50" y1="50" x2="50" y2="5" stroke="#94a3b8" strokeWidth="2.2" strokeLinecap="butt" fill="none" />
            <circle cx="50" cy="5" r="2.5" fill={currentColor} style={{ filter: `drop-shadow(0 0 5px ${currentColor})` }} />
          </g>

          {/* LAYER 2 — PIVOT HUB on top of needle base; cx cy EXACTLY (50, 50) */}
          <circle cx="50" cy="50" r="5.5" fill="#0b0f1a" stroke="#334155" strokeWidth="1.5" />
          <circle cx="50" cy="50" r="2.2" fill="#64748b" />
        </svg>
      </div>

      <div className="w-full max-w-[320px] mt-2 px-1" aria-hidden="true">
        <div className="grid grid-cols-5 gap-x-1 text-[10px] uppercase text-gray-500/80 tracking-wider">
          {['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'].map((cat) => (
            <span key={cat} className="text-center min-w-0" title={cat}>
              {cat === 'Extreme Fear' ? 'Ext. Fear' : cat === 'Extreme Greed' ? 'Ext. Greed' : cat}
            </span>
          ))}
        </div>
      </div>

      <div
        className="mt-5 text-center rounded-2xl border border-gray-600/40 px-8 py-4 min-w-[160px] backdrop-blur-md bg-gray-900/80"
        style={{ boxShadow: `inset 0 1px 6px rgba(0,0,0,0.45), 0 0 26px ${scoreGlow}` }}
        aria-live="polite"
        aria-label={`Sentiment score ${Math.round(displayScore)}, ${currentLabel}`}
      >
        <p className="text-3xl font-bold text-white leading-none tracking-tight tabular-nums">
          {Math.round(displayScore)}
        </p>
        <p className="text-xs font-semibold mt-2 uppercase tracking-wider" style={{ color: currentColor }}>
          {currentLabel}
        </p>
      </div>

      <p className="mt-4 text-sm leading-relaxed text-center text-gray-400/90 max-w-[280px]">
        {insightText}
      </p>

      <div className="w-full max-w-[300px] mt-5 pt-5 border-t border-slate-700/50">
        <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2.5 font-semibold">Vs Current</p>
        <div className="rounded-lg border border-slate-700/50 overflow-hidden bg-gray-900/20 backdrop-blur-sm">
          <table className="w-full text-left border-collapse table-fixed" aria-label="Sentiment vs current">
            <colgroup>
              <col style={{ width: '50%' }} />
              <col style={{ width: '50%' }} />
            </colgroup>
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="py-2 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Period</th>
                <th className="py-2 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Score</th>
              </tr>
            </thead>
            <tbody>
              {trendData.map((row) => {
                const val = Number(row.value);
                return (
                    <tr
                      key={row.periodKey}
                      className="border-b border-slate-700/30 last:border-0 cursor-pointer hover:bg-slate-800/50 transition-colors rounded"
                      onMouseEnter={() => setHoveredScore(row.score)}
                      onMouseLeave={() => setHoveredScore(null)}
                    >
                    <td className="py-2 px-3 text-xs text-gray-400">{row.period}</td>
                    <td className="py-2 px-3 text-sm font-semibold text-white text-right tabular-nums">{Math.round(val)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SignalDemoPortfolioPage({ onSelectStock }) {
  const [liveQuotes, setLiveQuotes] = useState({});
  const [lastSynced, setLastSynced] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const tickers = DEMO_PORTFOLIO_HOLDINGS.map((h) => h.ticker);
    const poll = async () => {
      const entries = await Promise.all(tickers.map(async (t) => [t, await fetchLiveQuote(t)]));
      if (cancelled) return;
      const next = {};
      entries.forEach(([ticker, quote]) => {
        if (quote) next[ticker] = quote;
      });
      if (Object.keys(next).length > 0) {
        setLiveQuotes((prev) => ({ ...prev, ...next }));
        setLastSynced(Date.now());
      }
    };
    poll();
    const id = setInterval(poll, LIVE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const rows = useMemo(() => buildDemoPortfolioRows(liveQuotes), [liveQuotes]);
  const totals = useMemo(() => sumDemoPortfolioTotals(rows), [rows]);
  const buyCount = useMemo(() => rows.filter((r) => (r.aiScore ?? 0) >= 66).length, [rows]);

  const [tradeHistoryModal, setTradeHistoryModal] = useState(null);
  const closeTradeHistoryModal = useCallback(() => setTradeHistoryModal(null), []);
  const modalTradeRows = useMemo(() => {
    if (!tradeHistoryModal) return [];
    if (tradeHistoryModal.filterMode === 'ticker') {
      return filterSignalDemoTradeHistory('ticker', tradeHistoryModal.ticker);
    }
    if (tradeHistoryModal.filterMode === 'wins') return filterSignalDemoTradeHistory('wins');
    return filterSignalDemoTradeHistory('all');
  }, [tradeHistoryModal]);

  const perfStatCardClass = `${CARD_BASE} p-5 border-sky-500/20 w-full text-left transition-all duration-200 cursor-pointer hover:border-sky-400/45 hover:bg-sky-500/[0.08] hover:shadow-[0_0_28px_rgba(56,189,248,0.12)] focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0B0E14] active:scale-[0.99]`;

  return (
    <div className="w-full min-h-full py-6 md:py-8 px-6 md:px-10 box-border overflow-x-hidden" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
      <div className="max-w-6xl mx-auto flex flex-col gap-6">
        <div className={`${CARD_BASE} p-6 md:p-8 border-sky-500/20`}>
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
            <div className="flex items-start gap-3 min-w-0">
              <div className="p-2.5 rounded-xl bg-sky-500/10 border border-sky-500/25 shrink-0">
                <Briefcase className="w-6 h-6 text-sky-400" />
              </div>
              <div>
                <h2 className="text-xl md:text-2xl font-semibold text-white tracking-tight">Signal-based demo portfolio</h2>
                <p className="text-sm text-gray-400 mt-2 leading-relaxed max-w-2xl">
                  Hypothetical paper positions aligned with the same <span className="text-gray-300">AI composite score</span> and signal bands
                  (Strong Buy / Buy / Hold / Sell) used on each stock&apos;s detail page. Quotes refresh on the same cadence as the home dashboard.
                  Not investment advice.
                </p>
                <p className="text-xs text-gray-500 mt-3">
                  <span className="text-gray-400 tabular-nums">{buyCount}</span> of {rows.length} demo names are currently Buy or Strong Buy (score ≥ 66).
                </p>
              </div>
            </div>
            <LiveIndicator lastSynced={lastSynced} />
          </div>
        </div>

        {/* Performance statistics — mock until API */}
        <div>
          <div className="flex flex-col gap-0.5 mb-3 px-0.5">
            <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-[0.2em]">STATISTICS</p>
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Performance statistics</h3>
            <p className="text-[11px] text-gray-600 mt-1">Click a card to open the trade blotter — filtered to match the metric.</p>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <button
              type="button"
              className={perfStatCardClass}
              onClick={() =>
                setTradeHistoryModal({
                  filterMode: 'all',
                  title: 'All executed trades',
                  subtitle: `Full demo blotter (${SIGNAL_DEMO_TRADE_HISTORY_FULL_MOCK.length} legs). Headline count shows model lifetime; detail view is illustrative.`,
                })
              }
            >
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Total trades executed</p>
              <p className="text-2xl font-bold text-white tabular-nums tracking-tight">
                {SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.totalTradesExecuted}
              </p>
            </button>
            <button
              type="button"
              className={perfStatCardClass}
              onClick={() =>
                setTradeHistoryModal({
                  filterMode: 'wins',
                  title: 'Winning trades (realized)',
                  subtitle: 'Sell legs with positive realized P/L — drill-down for the headline win rate.',
                })
              }
            >
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Win rate</p>
              <p className="text-2xl font-bold text-white tabular-nums tracking-tight">
                {SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.winRatePct.toFixed(1)}%
              </p>
            </button>
            <button
              type="button"
              className={perfStatCardClass}
              onClick={() => {
                const tk = SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.bestTrade.ticker;
                setTradeHistoryModal({
                  filterMode: 'ticker',
                  ticker: tk,
                  title: `Trades · ${tk}`,
                  subtitle: `All executions for ${tk} in the demo book — matches “best performing” headline name.`,
                });
              }}
            >
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Best performing trade</p>
              <p className="text-lg font-bold text-white tabular-nums tracking-tight">
                {SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.bestTrade.ticker}
              </p>
              <p className="text-sm font-semibold text-emerald-400 tabular-nums mt-1">
                {SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.bestTrade.label}
              </p>
              <p className="text-[11px] text-gray-500 mt-0.5">{SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.bestTrade.sub}</p>
            </button>
            <button
              type="button"
              className={perfStatCardClass}
              onClick={() => {
                const tk = SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.worstTrade.ticker;
                setTradeHistoryModal({
                  filterMode: 'ticker',
                  ticker: tk,
                  title: `Trades · ${tk}`,
                  subtitle: `All executions for ${tk} in the demo book — matches “worst performing” headline name.`,
                });
              }}
            >
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Worst performing trade</p>
              <p className="text-lg font-bold text-white tabular-nums tracking-tight">
                {SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.worstTrade.ticker}
              </p>
              <p className="text-sm font-semibold text-red-400 tabular-nums mt-1">
                {SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.worstTrade.label}
              </p>
              <p className="text-[11px] text-gray-500 mt-0.5">{SIGNAL_DEMO_ALGO_PERFORMANCE_MOCK.worstTrade.sub}</p>
            </button>
          </div>
        </div>

        <div className={`${CARD_BASE} overflow-hidden`}>
          <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-emerald-400 shrink-0" />
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Holdings &amp; live marks</h3>
            </div>
            <div className="flex flex-wrap gap-4 text-xs sm:text-sm tabular-nums">
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Market value</p>
                <p className="text-white font-semibold">
                  ${totals.marketValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Cost basis</p>
                <p className="text-gray-300 font-medium">
                  ${totals.costBasis.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Total P/L</p>
                <p className={`font-semibold ${totals.pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {totals.pl >= 0 ? '+' : ''}
                  ${totals.pl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  <span className="text-[11px] ml-1">
                    ({totals.plPct >= 0 ? '+' : ''}
                    {totals.plPct.toFixed(2)}%)
                  </span>
                </p>
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse min-w-[720px]">
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider w-12" aria-label="Logo" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">AI signal</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Score</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider hidden md:table-cell">Thesis</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Shares</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Avg cost</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Price</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Value</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">P/L</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const stock = getStockMeta(row.ticker);
                  const SigIcon = row.signal?.Icon ?? Minus;
                  return (
                    <tr
                      key={row.ticker}
                      onClick={() => stock && onSelectStock(stock)}
                      className={`border-b border-slate-800/50 hover:bg-white/5 transition-colors ${stock ? 'cursor-pointer' : ''}`}
                    >
                      <td className="py-2.5 px-3 align-top">
                        <StockLogo ticker={row.ticker} />
                      </td>
                      <td className="py-2.5 px-3 align-top">
                        <p className="font-medium text-white text-sm">{row.ticker}</p>
                        <p className="text-[11px] text-gray-500 mt-0.5 md:hidden line-clamp-2">{row.aiSummary}</p>
                      </td>
                      <td className="py-2.5 px-3 align-top">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wide border ${row.signal?.bgCls ?? ''} ${row.signal?.textCls ?? 'text-gray-400'} ${row.signal?.borderCls ?? 'border-slate-700'}`}
                        >
                          <SigIcon className="w-3.5 h-3.5 shrink-0" />
                          {row.signal?.label ?? '—'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-200 tabular-nums align-top font-medium">{row.aiScore}</td>
                      <td
                        className="py-2.5 px-3 text-xs text-gray-400 align-top max-w-[20rem] leading-snug hidden md:table-cell"
                        title={row.aiSummary}
                      >
                        {row.aiSummary}
                      </td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums align-top">{row.shares}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums align-top">${row.avgCost.toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-white tabular-nums align-top">${row.price.toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-200 tabular-nums font-medium align-top">
                        ${row.marketValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums align-top ${row.pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}
                      >
                        {row.pl >= 0 ? '+' : ''}${row.pl.toFixed(2)}
                        <span className="text-[11px] font-normal ml-1">
                          ({row.plPct >= 0 ? '+' : ''}
                          {row.plPct.toFixed(1)}%)
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Recent model actions — mock until API */}
        <div className={`${CARD_BASE} overflow-hidden`}>
          <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1.5">
            <div>
              <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-[0.2em] mb-1">LATEST ACTIVITY</p>
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-sky-400 shrink-0" />
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Recent model actions</h3>
              </div>
            </div>
            <p className="text-[10px] text-gray-600 uppercase tracking-wider">Last 8 executions · paperbook</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse min-w-[780px]">
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Date</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider w-12" aria-label="Logo" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Action</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Shares</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Execution price</th>
                </tr>
              </thead>
              <tbody>
                {SIGNAL_DEMO_RECENT_MODEL_ACTIONS_MOCK.map((row, idx) => {
                  const stock = getStockMeta(row.ticker);
                  return (
                    <tr
                      key={`${row.date}-${row.ticker}-${idx}`}
                      onClick={() => stock && onSelectStock(stock)}
                      className={`border-b border-slate-800/50 hover:bg-white/5 transition-colors ${stock ? 'cursor-pointer' : ''}`}
                    >
                      <td className="py-2.5 px-3 text-sm text-gray-300 tabular-nums">{row.date}</td>
                      <td className="py-2.5 px-3 align-middle">
                        <StockLogo ticker={row.ticker} />
                      </td>
                      <td className="py-2.5 px-3 align-middle">
                        <span className="font-medium text-white text-sm tabular-nums">{row.ticker}</span>
                      </td>
                      <td className="py-2.5 px-3 align-middle">
                        <AlgoExecutionBadge action={row.action} />
                      </td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{row.shares}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-white tabular-nums">
                        ${Number(row.executionPrice).toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className={`${CARD_BASE} p-5 border border-slate-800/80`}>
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-gray-500 mb-2">Signal methodology</p>
          <p className="text-sm text-gray-400 leading-relaxed">
            Scores (0–100) and labels follow the same thresholds as the rest of the app: Strong Buy (85+), Buy (66–84), Hold (41–65), Sell (40 and below).
            Demo share counts and average cost are static illustrations; only marks and P/L update with live quotes.
          </p>
        </div>
      </div>

      <SignalDemoTradeHistoryModal
        open={Boolean(tradeHistoryModal)}
        title={tradeHistoryModal?.title ?? ''}
        subtitle={tradeHistoryModal?.subtitle}
        rows={modalTradeRows}
        onClose={closeTradeHistoryModal}
        onSelectStock={onSelectStock}
      />
    </div>
  );
}

function HomeDashboard({ onSelectStock, watchlistTickers, onRemoveWatchlist, onToggleWatchlist }) {
  const [sentimentValue, setSentimentValue] = useState(22);
  const [sentimentLastSynced, setSentimentLastSynced] = useState(null);
  const [liveQuotes, setLiveQuotes] = useState({});
  const [watchlistQuotes, setWatchlistQuotes] = useState({});
  const [isSyncingTables, setIsSyncingTables] = useState(false);
  const [lastTablesSynced, setLastTablesSynced] = useState(null);
  useEffect(() => {
    let cancelled = false;
    const fetchFearGreed = async () => {
      try {
        const res = await fetch(FEAR_GREED_API_URL);
        const data = await res.json();
        if (!res.ok || cancelled) return;
        const liveValue = data?.value ?? data?.fearGreed ?? data?.score;
        const parsed = Number(liveValue);
        if (!Number.isFinite(parsed)) return;
        setSentimentValue(Math.min(100, Math.max(0, parsed)));
        setSentimentLastSynced(Date.now());
      } catch (_) {
        // Keep latest known value; graceful fallback.
      }
    };
    fetchFearGreed();
    const id = setInterval(fetchFearGreed, LIVE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tickers = TOP_US_STOCKS.map((s) => s.ticker);
    const fetchTableQuotes = async () => {
      if (!cancelled) setIsSyncingTables(true);
      const entries = await Promise.all(
        tickers.map(async (ticker) => [ticker, await fetchLiveQuote(ticker)]),
      );
      if (cancelled) return;
      const next = {};
      entries.forEach(([ticker, quote]) => {
        if (quote) next[ticker] = quote;
      });
      if (Object.keys(next).length > 0) {
        setLiveQuotes((prev) => ({ ...prev, ...next }));
        setLastTablesSynced(Date.now());
      }
      setIsSyncingTables(false);
    };
    fetchTableQuotes();
    const id = setInterval(fetchTableQuotes, LIVE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tickers = Array.isArray(watchlistTickers) ? watchlistTickers.filter(Boolean) : [];
    const fetchWatchlistQuotes = async () => {
      if (tickers.length === 0) {
        if (!cancelled) setWatchlistQuotes({});
        return;
      }
      const entries = await Promise.all(
        tickers.map(async (ticker) => [ticker, await fetchLiveQuote(ticker)]),
      );
      if (cancelled) return;
      const next = {};
      entries.forEach(([ticker, quote]) => {
        if (quote) next[ticker] = quote;
      });
      if (Object.keys(next).length > 0) setWatchlistQuotes((prev) => ({ ...prev, ...next }));
    };
    fetchWatchlistQuotes();
    const id = setInterval(fetchWatchlistQuotes, LIVE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [watchlistTickers]);

  const watchlistRows = useMemo(
    () =>
      (Array.isArray(watchlistTickers) ? watchlistTickers : []).map((ticker) => {
        const meta = getStockMeta(ticker);
        const q = watchlistQuotes[ticker];
        return {
          ticker,
          name: meta?.name ?? ticker,
          price: q?.price ?? meta?.price ?? 0,
          changePercent: q?.changePercent ?? meta?.changePercent ?? 0,
        };
      }),
    [watchlistTickers, watchlistQuotes],
  );

  const stocksWithLive = useMemo(
    () => TOP_US_STOCKS.map((stock) => {
      const q = liveQuotes[stock.ticker];
      return q ? { ...stock, price: q.price, changePercent: q.changePercent } : stock;
    }),
    [liveQuotes],
  );

  const demoPortfolioRows = useMemo(() => buildDemoPortfolioRows(liveQuotes), [liveQuotes]);
  const demoPortfolioTotals = useMemo(() => sumDemoPortfolioTotals(demoPortfolioRows), [demoPortfolioRows]);

  const fearGreedTimelineData = [
    { month: 'Jan', fearGreed: 31, sp500: 4720 },
    { month: 'Feb', fearGreed: 38, sp500: 4890 },
    { month: 'Mar', fearGreed: 29, sp500: 4810 },
    { month: 'Apr', fearGreed: 45, sp500: 5020 },
    { month: 'May', fearGreed: 41, sp500: 4940 },
    { month: 'Jun', fearGreed: 56, sp500: 5110 },
    { month: 'Jul', fearGreed: 48, sp500: 4980 },
    { month: 'Aug', fearGreed: 35, sp500: 4860 },
    { month: 'Sep', fearGreed: 42, sp500: 4920 },
    { month: 'Oct', fearGreed: 26, sp500: 4780 },
    { month: 'Nov', fearGreed: 19, sp500: 4690 },
    { month: 'Dec', fearGreed: 22, sp500: 4750 },
  ];
  const sentimentNews = {
    fearDrivers: [
      { id: 'f1', headline: 'Inflation data comes in hot, rate cut hopes fade', source: 'Reuters', time: '2h ago' },
      { id: 'f2', headline: 'Geopolitical tensions weigh on risk appetite', source: 'Bloomberg', time: '5h ago' },
      { id: 'f3', headline: 'Bond yields spike on Treasury supply concerns', source: 'Fed', time: '1d ago' },
    ],
    greedDrivers: [
      { id: 'g1', headline: 'Tech earnings beat expectations, Nasdaq rallies', source: 'CNBC', time: '3h ago' },
      { id: 'g2', headline: 'Strong jobs data supports soft landing hopes', source: 'BLS', time: '1d ago' },
    ],
    aiConclusion: {
      summary: 'Fear dominates as inflation and rate concerns weigh on risk appetite. The index may stay in Fear until the next Fed signal.',
    },
  };

  return (
    <div className="w-full min-h-full py-6 md:py-8 box-border overflow-x-hidden" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
      <div className="w-full px-10 flex flex-col gap-6">
        {/* Grid 1:2 — left Gauge, right Chart/News */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full min-w-0">
          <div className={`${CARD_BASE} p-6 w-full flex flex-col min-h-[520px] lg:col-span-1`}>
            <div className="flex items-center gap-2 mb-3 w-full text-left">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.2em]">
                US Market Sentiment
              </h3>
            </div>
            <div className="flex-1 flex items-center justify-center">
              <SentimentGauge value={sentimentValue} />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/60">
              <LiveIndicator lastSynced={sentimentLastSynced} />
            </div>
          </div>

          <div className={`${CARD_BASE} p-6 w-full min-h-[520px] lg:col-span-2 flex flex-col`}>
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.2em] mb-1 w-full text-left">
              Fear & Greed Timeline
            </h3>
            <p className="text-xs text-gray-500 mb-4 uppercase tracking-wider">Fear & Greed Index vs S&amp;P 500</p>
            <div className="w-full h-[280px] flex-shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={fearGreedTimelineData} margin={{ top: 8, right: 8, left: 4, bottom: 4 }}>
                <defs>
                  <linearGradient id="fgGradTimeline" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#eab308" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#eab308" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="spGradTimeline" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#93c5fd" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#93c5fd" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" strokeOpacity={0.1} vertical={false} />
                <XAxis dataKey="month" tick={{ fill: '#6b7280', fontSize: 12 }} stroke="rgba(255,255,255,0.06)" />
                <YAxis yAxisId="left" type="number" domain={[0, 100]} padding={{ top: 20, bottom: 20 }} tick={{ fill: '#6b7280', fontSize: 12 }} stroke="rgba(255,255,255,0.06)" width={36} />
                <YAxis yAxisId="right" type="number" orientation="right" domain={['auto', 'auto']} padding={{ top: 20, bottom: 20 }} tick={{ fill: '#6b7280', fontSize: 12 }} stroke="rgba(255,255,255,0.06)" width={44} tickFormatter={(v) => `${(v / 1000).toFixed(1)}k`} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    border: '1px solid rgba(71, 85, 105, 0.5)',
                    borderRadius: '12px',
                    backdropFilter: 'blur(12px)',
                    padding: '10px 14px',
                  }}
                  itemStyle={{ color: '#e2e8f0', fontSize: 12 }}
                  formatter={(val, name) => [name === 'fearGreed' ? val : Number(val).toLocaleString(), name === 'fearGreed' ? 'Fear & Greed' : 'S&P 500']}
                />
                <Area yAxisId="left" type="linear" dataKey="fearGreed" fill="url(#fgGradTimeline)" stroke="#eab308" strokeWidth={1.5} name="Fear & Greed" />
                <Area yAxisId="right" type="linear" dataKey="sp500" fill="url(#spGradTimeline)" stroke="#93c5fd" strokeWidth={1.5} name="S&P 500" />
                <Legend
                  align="right"
                  verticalAlign="top"
                  iconType="line"
                  iconSize={8}
                  wrapperStyle={{ paddingLeft: '12px', fontSize: '12px' }}
                  formatter={(value) => <span className="text-gray-500">{value}</span>}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-6 pt-6 border-t border-slate-800/60 flex-1">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-5">
              <div className="rounded-lg border-l-[3px] border-red-500/30 bg-red-950/10 border border-slate-800/60 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.15em] text-red-400/90 mb-2.5">Fear Drivers</p>
                <ul className="space-y-3">
                  {sentimentNews.fearDrivers.map((item) => (
                    <li key={item.id} className="text-sm text-gray-400 leading-snug">
                      <span className="text-gray-300">{item.headline}</span>
                      <span className="block text-xs text-gray-500 mt-1">
                        {[item.source, item.time].filter(Boolean).join(' · ')}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="rounded-lg border-l-[3px] border-green-500/30 bg-green-950/10 border border-slate-800/60 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.15em] text-green-400/90 mb-2.5">Greed Drivers</p>
                <ul className="space-y-3">
                  {sentimentNews.greedDrivers.map((item) => (
                    <li key={item.id} className="text-sm text-gray-400 leading-snug">
                      <span className="text-gray-300">{item.headline}</span>
                      <span className="block text-xs text-gray-500 mt-1">
                        {[item.source, item.time].filter(Boolean).join(' · ')}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div
              className="rounded-lg border border-slate-800/80 bg-[#12131a] px-5 py-4"
              style={{ borderLeftWidth: '3px', borderLeftColor: 'rgba(139, 92, 246, 0.5)' }}
            >
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-300 mb-2">AI Sentiment Conclusion</p>
              <p className="text-sm text-gray-300 leading-relaxed">{sentimentNews.aiConclusion.summary}</p>
            </div>
          </div>
        </div>
      </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 w-full">
          <div className={`${CARD_BASE} p-5 flex flex-col justify-center`}>
            <h3 className="text-gray-400 text-base font-medium">SPY (S&amp;P 500)</h3>
            <p className="text-2xl md:text-3xl font-bold text-white mt-1.5">593.30 USD</p>
            <p className="text-red-400 text-base mt-1.5">-0.47%</p>
          </div>
          <div className={`${CARD_BASE} p-5 flex flex-col justify-center`}>
            <h3 className="text-gray-400 text-base font-medium">QQQ (Nasdaq)</h3>
            <p className="text-2xl md:text-3xl font-bold text-white mt-1.5">498.12 USD</p>
            <p className="text-green-400 text-base mt-1.5">+0.12%</p>
          </div>
          <div className={`${CARD_BASE} p-5 flex flex-col justify-center`}>
            <h3 className="text-gray-400 text-base font-medium">IWM (Russell 2000)</h3>
            <p className="text-2xl md:text-3xl font-bold text-white mt-1.5">215.40 USD</p>
            <p className="text-green-400 text-base mt-1.5">+1.05%</p>
          </div>
        </div>

        {/* Demo portfolio */}
        <div className={`${CARD_BASE} overflow-hidden w-full mt-6`}>
          <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-2 min-w-0">
              <Briefcase className="w-4 h-4 text-sky-400 shrink-0" />
              <div>
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Demo Portfolio</h3>
                <p className="text-[10px] text-gray-500 mt-0.5">
                  Sample holdings tied to AI signals — open <span className="text-gray-400">Signal portfolio</span> in the header for the full view.
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-4 text-xs sm:text-sm tabular-nums">
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Market value</p>
                <p className="text-white font-semibold">${demoPortfolioTotals.marketValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Cost basis</p>
                <p className="text-gray-300 font-medium">${demoPortfolioTotals.costBasis.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Total P/L</p>
                <p className={`font-semibold ${demoPortfolioTotals.pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {demoPortfolioTotals.pl >= 0 ? '+' : ''}
                  ${demoPortfolioTotals.pl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  <span className="text-[11px] ml-1">
                    ({demoPortfolioTotals.plPct >= 0 ? '+' : ''}{demoPortfolioTotals.plPct.toFixed(2)}%)
                  </span>
                </p>
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse min-w-[640px]">
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider w-12" aria-label="Logo" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider hidden lg:table-cell">AI signal</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider hidden md:table-cell">Name</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Shares</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Avg cost</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Price</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Value</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">P/L</th>
                </tr>
              </thead>
              <tbody>
                {demoPortfolioRows.map((row) => {
                  const stock = getStockMeta(row.ticker);
                  const SigIcon = row.signal?.Icon ?? Minus;
                  return (
                    <tr
                      key={row.ticker}
                      onClick={() => stock && onSelectStock(stock)}
                      className={`border-b border-slate-800/50 hover:bg-white/5 transition-colors ${stock ? 'cursor-pointer' : ''}`}
                    >
                      <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 hidden lg:table-cell">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wide border ${row.signal?.bgCls ?? ''} ${row.signal?.textCls ?? 'text-gray-400'} ${row.signal?.borderCls ?? 'border-slate-700'}`}
                          title={`Score ${row.aiScore}`}
                        >
                          <SigIcon className="w-3 h-3 shrink-0" />
                          {row.signal?.label ?? '—'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-sm text-gray-400 truncate hidden md:table-cell max-w-[12rem]" title={row.name}>
                        {row.name}
                      </td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{row.shares}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">${row.avgCost.toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-white tabular-nums">${row.price.toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-200 tabular-nums font-medium">
                        ${row.marketValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${row.pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {row.pl >= 0 ? '+' : ''}${row.pl.toFixed(2)}
                        <span className="text-[11px] font-normal ml-1">({row.plPct >= 0 ? '+' : ''}{row.plPct.toFixed(1)}%)</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Watchlist */}
        <div className={`${CARD_BASE} overflow-hidden w-full mt-6`}>
          <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 min-w-0">
              <Star className="w-4 h-4 text-amber-400 shrink-0 fill-amber-400/25" />
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider truncate">My Watchlist</h3>
              <span className="text-[10px] text-gray-500 tabular-nums shrink-0">
                {watchlistRows.length}/{WATCHLIST_MAX}
              </span>
            </div>
          </div>
          {watchlistRows.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-gray-500">
              No symbols yet. Open a stock and use <span className="text-gray-400">Add to watchlist</span>.
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider w-12" aria-label="Logo" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider hidden sm:table-cell">Name</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Price</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Chg %</th>
                  <th className="py-2.5 px-3 w-10" aria-label="Remove" />
                </tr>
              </thead>
              <tbody>
                {watchlistRows.map((row) => {
                  const stock = getStockMeta(row.ticker);
                  const canOpen = Boolean(stock);
                  return (
                    <tr
                      key={row.ticker}
                      className={`border-b border-slate-800/50 hover:bg-white/5 transition-colors ${canOpen ? 'cursor-pointer' : ''}`}
                      onClick={() => {
                        if (stock) onSelectStock(stock);
                      }}
                    >
                      <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 text-sm text-gray-400 truncate hidden sm:table-cell max-w-[10rem]" title={row.name}>
                        {row.name}
                      </td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">
                        ${Number(row.price ?? 0).toFixed(2)}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${
                          (row.changePercent ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                        }`}
                      >
                        {(row.changePercent ?? 0) >= 0 ? '+' : ''}
                        {Number(row.changePercent ?? 0).toFixed(2)}%
                      </td>
                      <td className="py-2 px-2">
                        <button
                          type="button"
                          className="p-1.5 rounded-md text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                          aria-label={`Remove ${row.ticker} from watchlist`}
                          onClick={(e) => {
                            e.stopPropagation();
                            onRemoveWatchlist?.(row.ticker);
                          }}
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        <div className="flex items-center justify-between mt-2">
          <p className="text-[10px] uppercase tracking-wider text-gray-600">Live Equity Tables</p>
          <div className="flex items-center gap-3">
            {isSyncingTables && <span className="text-[10px] text-emerald-400 animate-pulse uppercase tracking-wider">Syncing...</span>}
            <LiveIndicator lastSynced={lastTablesSynced} />
          </div>
        </div>

        {/* Three professional stock tables — no per-table scrollbars; section flows with page */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full mt-2 items-start">

          {/* TABLE 1: Top 20 Gainers — exactly 20 rows, sorted by changePercent desc */}
          <div className={`${CARD_BASE} overflow-hidden`}>
            <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Top 20 Gainers</h3>
            </div>
            <table className="w-full text-left border-collapse table-fixed">
              <colgroup>
                <col style={{ width: '50px' }} />
                <col style={{ width: '88px' }} />
                <col style={{ width: '88px' }} />
                <col style={{ width: '88px' }} />
              </colgroup>
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider" aria-label="Logo" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Price</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Chg %</th>
                </tr>
              </thead>
              <tbody>
                {[...stocksWithLive]
                  .sort((a, b) => (b.changePercent ?? 0) - (a.changePercent ?? 0))
                  .slice(0, 10)
                  .map((row) => (
                    <tr
                      key={row.ticker}
                      onClick={() => onSelectStock(row)}
                      className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer"
                    >
                      <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">${Number(row?.price ?? 0).toFixed(2)}</td>
                      <td className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${(row?.changePercent ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {(row?.changePercent ?? 0) >= 0 ? '+' : ''}{Number(row?.changePercent ?? 0).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* TABLE 2: AI Pro Predictions — Logo | Ticker | Signal | Conf */}
          <div className={`${CARD_BASE} overflow-hidden`}>
            <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">AI Pro Predictions</h3>
            </div>
            <table className="w-full text-left border-collapse table-fixed">
              <colgroup>
                <col style={{ width: '32px' }} />
                <col style={{ width: '50px' }} />
                <col style={{ width: '88px' }} />
                <col />
                <col style={{ width: '72px' }} />
              </colgroup>
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-1 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-center" aria-label="Watchlist" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider" aria-label="Logo" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Ticker</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Signal</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                {stocksWithLive
                  .slice(0, 10)
                  .map((row) => {
                  // Single source of truth: score from STOCK_AI_RATINGS, label/color from getSignal
                  const aiScore = STOCK_AI_RATINGS[row.ticker]?.score ?? 70;
                  const { label: sigLabel, textCls: sigText, Icon: SigIcon } = getSignal(aiScore);
                  const watched = (watchlistTickers || []).includes(String(row.ticker || '').toUpperCase());
                  return (
                    <tr
                      key={row.ticker}
                      onClick={() => onSelectStock(row)}
                      className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer"
                    >
                      <td className="py-2 px-1 align-middle text-center">
                        <button
                          type="button"
                          className={`inline-flex items-center justify-center p-0.5 rounded transition-colors ${
                            watched
                              ? 'text-amber-400 hover:text-amber-300'
                              : 'text-slate-600 hover:text-amber-400/90'
                          }`}
                          aria-label={watched ? `Remove ${row.ticker} from watchlist` : `Add ${row.ticker} to watchlist`}
                          onClick={(e) => {
                            e.stopPropagation();
                            onToggleWatchlist?.(row.ticker);
                          }}
                        >
                          <Star className={`w-3.5 h-3.5 shrink-0 ${watched ? 'fill-amber-400 text-amber-400' : ''}`} />
                        </button>
                      </td>
                      <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3">
                        <span className={`inline-flex items-center gap-1 text-xs font-medium ${sigText}`}>
                          <SigIcon className="w-3.5 h-3.5 shrink-0" />{sigLabel}
                        </span>
                      </td>
                      <td className={`py-2.5 px-3 text-right text-sm tabular-nums font-semibold ${sigText}`}>{aiScore}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* TABLE 3: Market Giants — Logo | Symbol | Name | Price, sorted by marketCap desc */}
          <div className={`${CARD_BASE} overflow-hidden`}>
            <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Market Giants</h3>
            </div>
            <table className="w-full text-left border-collapse table-fixed">
              <colgroup>
                <col style={{ width: '50px' }} />
                <col style={{ width: '88px' }} />
                <col />
                <col style={{ width: '88px' }} />
              </colgroup>
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider" aria-label="Logo" />
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Name</th>
                  <th className="py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Price</th>
                </tr>
              </thead>
              <tbody>
                {[...stocksWithLive]
                  .sort((a, b) => (b.marketCap ?? 0) - (a.marketCap ?? 0))
                  .slice(0, 10)
                  .map((row) => (
                    <tr
                      key={row.ticker}
                      onClick={() => onSelectStock(row)}
                      className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer"
                    >
                      <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 text-sm text-gray-400 truncate" title={row.name}>{row.name}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums font-medium">${Number(row?.price ?? 0).toFixed(2)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

        </div>
      </div>
    </div>
  );
}

// --- Professional Forecast Chart Component ---
const ProfessionalForecastChart = ({ data, currentPrice }) => {
  const safeData = Array.isArray(data) ? data : [];
  const safeCurrentPrice =
    typeof currentPrice === 'number' && Number.isFinite(currentPrice) ? currentPrice : null;

  return (
    <div className="bg-[#131722] border border-[#2B3139] rounded-xl p-4 h-[500px] flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-white font-semibold">Forecast Performance</h2>
        <div className="flex bg-[#2B3139] rounded-md p-1">
          {['1M', '3M', '6M', '1Y', 'ALL'].map((range) => (
            <button key={range} className={`px-3 py-1 text-xs rounded ${range === '3M' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}`}>
              {range}
            </button>
          ))}
        </div>
      </div>
      
      <div className="flex-grow">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={safeData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorBull" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorBase" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorBear" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#2B3139" vertical={false} />
            <XAxis dataKey="date" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 12 }} tickFormatter={(tick) => tick.substring(5)} minTickGap={30} />
            <YAxis
              stroke="#6B7280"
              tick={{ fill: '#6B7280', fontSize: 12 }}
              domain={['dataMin - 5', 'dataMax + 5']}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#1E222D', border: '1px solid #2B3139', borderRadius: '8px' }}
              itemStyle={{ color: '#E5E7EB' }}
            />
            {safeCurrentPrice !== null && (
              <ReferenceLine y={safeCurrentPrice} stroke="#eab308" strokeDasharray="3 3" />
            )}
            
            <Area type="monotone" dataKey="bull" fill="url(#colorBull)" stroke="#22c55e" strokeWidth={1} connectNulls={true} />
            <Area type="monotone" dataKey="base" fill="url(#colorBase)" stroke="#3b82f6" strokeWidth={1} connectNulls={true} />
            <Area type="monotone" dataKey="bear" fill="url(#colorBear)" stroke="#ef4444" strokeWidth={1} connectNulls={true} />
            <Line type="linear" dataKey="price" stroke="#ffffff" strokeWidth={2} dot={false} connectNulls={true} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div className="flex gap-4 mt-4 text-xs text-gray-400">
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-white rounded-sm"></div> Historical Price</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-green-500 rounded-sm"></div> Bull (90th)</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-blue-500 rounded-sm"></div> Base (50th)</div>
        <div className="flex items-center gap-1"><div className="w-3 h-3 bg-red-500 rounded-sm"></div> Bear (10th)</div>
      </div>
    </div>
  );
};

// --- Stock Analysis Screen Component ---
const StockAnalysisScreen = ({ stockData }) => {
  if (!stockData) {
    return (
      <div className="p-6">
        <p className="text-sm text-gray-400">Loading stock analysis…</p>
      </div>
    );
  }

  const price = Number(stockData.price);
  const changePct = Number(stockData.change_pct);
  const hasPrice = Number.isFinite(price);
  const hasChangePct = Number.isFinite(changePct);
  const isPositive = hasChangePct ? changePct >= 0 : true;

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white tracking-tight">{stockData.ticker || '—'}</h1>
        <p className="text-gray-400 text-sm mt-1">
          {stockData.ticker ? `${stockData.ticker} — Equity` : 'Equity'}
        </p>
        <div className="flex items-baseline gap-3 mt-4">
          <span className="text-4xl font-bold text-white">
            {hasPrice ? `${price.toFixed(2)} USD` : '--'}
          </span>
          <span className={`text-lg font-medium flex items-center ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
            {isPositive ? <TrendingUp className="w-5 h-5 mr-1" /> : <TrendingDown className="w-5 h-5 mr-1" />}
            {hasChangePct ? `${Math.abs(changePct).toFixed(2)}%` : '--'}
          </span>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart Area */}
        <div className="lg:col-span-2">
          <ProfessionalForecastChart
            data={stockData.chart_data || []}
            currentPrice={hasPrice ? price : null}
          />
        </div>
        
        {/* Sidebar Cards */}
        <div className="space-y-4">
          <div className="bg-[#131722] border border-[#2B3139] p-5 rounded-xl">
            <h3 className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-4">AI Forecast Model</h3>
            
            <div className="space-y-3">
              <div className="bg-[#1E222D] p-3 rounded-lg border border-[#2B3139]">
                <p className="text-gray-500 text-xs mb-1">Quantile Regression Target</p>
                <p className="text-white font-medium">
                  {hasPrice ? price.toFixed(2) : '--'}
                </p>
              </div>
              <div className="bg-[#1E222D] p-3 rounded-lg border border-[#2B3139]">
                <p className="text-gray-500 text-xs mb-1">Bull Case (90th)</p>
                <p className="text-green-500 font-medium">{stockData.forecast_summary?.bull || '--'}</p>
              </div>
              <div className="bg-[#1E222D] p-3 rounded-lg border border-[#2B3139]">
                <p className="text-gray-500 text-xs mb-1">Base Case (50th)</p>
                <p className="text-blue-400 font-medium">{stockData.forecast_summary?.base || '--'}</p>
              </div>
              <div className="bg-[#1E222D] p-3 rounded-lg border border-[#2B3139]">
                <p className="text-gray-500 text-xs mb-1">Bear Case (10th)</p>
                <p className="text-red-500 font-medium">{stockData.forecast_summary?.bear || '--'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Main App Component ---
function AppInner() {
  console.log('API Key Status:', !!import.meta.env?.VITE_FINNHUB_KEY);
  console.log('Key:', import.meta.env?.VITE_FINNHUB_KEY);

  const [searchInput, setSearchInput]   = useState('');
  const [currentView, setCurrentView]   = useState('home');
  const [activeStock, setActiveStock]   = useState(null);
  const [selectedStock, setSelectedStock] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [marketData] = useState(() => TOP_US_STOCKS ?? []);
  const [watchlistTickers, setWatchlistTickers] = useState(() => loadWatchlistTickers());

  useEffect(() => {
    saveWatchlistTickers(watchlistTickers);
  }, [watchlistTickers]);

  const toggleWatchlistTicker = (rawTicker) => {
    const t = String(rawTicker || '').trim().toUpperCase();
    if (!t || !getStockMeta(t)) return;
    setWatchlistTickers((prev) => {
      if (prev.includes(t)) return prev.filter((x) => x !== t);
      if (prev.length >= WATCHLIST_MAX) return prev;
      return [...prev, t];
    });
  };

  const removeWatchlistTicker = (rawTicker) => {
    const t = String(rawTicker || '').trim().toUpperCase();
    if (!t) return;
    setWatchlistTickers((prev) => prev.filter((x) => x !== t));
  };

  const popularTickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL'];
  const normalizedSearch = searchInput.trim().toUpperCase();
  const suggestions =
    normalizedSearch.length === 0
      ? []
      : popularTickers.filter((ticker) => ticker.includes(normalizedSearch)).slice(0, 5);

  const runTickerSearch = (rawInput) => {
    const ticker = rawInput.trim().toUpperCase();
    if (!ticker) return;
    setSearchInput(ticker);
    setSearchError('');

    const matched = TOP_US_STOCKS?.find((s) => s.ticker?.toUpperCase() === ticker);
    if (!matched) {
      setIsSearching(false);
      setSearchError('Ticker not found');
      return;
    }

    setIsSearching(true);
    setActiveStock(null);
    setSelectedStock(matched);
    setCurrentView('detail');
  };

  const handleSearch = (e) => {
    if (e.key === 'Enter') runTickerSearch(searchInput);
  };

  const handleSearchClick = () => runTickerSearch(searchInput);

  const handleSelectStock = (stock) => {
    setSearchError('');
    setSelectedStock(stock);
    setCurrentView('detail');
  };

  const handleBack = () => {
    setIsSearching(false);
    setSearchError('');
    setSelectedStock(null);
    setCurrentView('home');
  };

  if (!marketData || marketData.length === 0) {
    return (
      <div className="min-h-screen bg-[#0B0E14] text-white flex items-center justify-center text-xl">
        Loading QuantVision AI...
      </div>
    );
  }

  const navTitle =
    currentView === 'detail'
      ? (selectedStock ? `${selectedStock.ticker} — Detail` : 'Stock Detail')
      : 'Global Market Predictor';

  return (
    <div className="min-h-screen bg-[#0B0E14] text-gray-100 font-sans">
      <div className="w-full flex flex-col min-h-screen min-w-0">
        {/* Top Navbar */}
        <div className="min-h-16 border-b border-gray-800 flex flex-wrap items-center px-4 sm:px-6 py-2 gap-y-2 justify-between sticky top-0 bg-[#0B0E14] z-10">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
            {currentView === 'detail' ? (
              <>
                <button
                  type="button"
                  onClick={handleBack}
                  className="text-gray-400 hover:text-white flex items-center gap-1 text-sm transition-colors shrink-0"
                >
                  <ArrowLeft className="w-4 h-4" /> Back
                </button>
                <h1 className="text-base sm:text-lg font-semibold text-gray-200 truncate">{navTitle}</h1>
              </>
            ) : (
              <>
                <h1 className="text-base sm:text-lg font-semibold text-gray-200 truncate shrink-0">Global Market Predictor</h1>
                <nav className="flex items-center gap-1 ml-1 sm:ml-2 pl-2 sm:pl-3 border-l border-gray-800 shrink-0" aria-label="Primary navigation">
                  <button
                    type="button"
                    onClick={() => {
                      setSearchError('');
                      setCurrentView('home');
                    }}
                    className={`px-2.5 sm:px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      currentView === 'home'
                        ? 'bg-slate-800 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-slate-800/50'
                    }`}
                  >
                    Home
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSearchError('');
                      setCurrentView('signal-portfolio');
                    }}
                    className={`px-2.5 sm:px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      currentView === 'signal-portfolio'
                        ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
                        : 'text-gray-400 hover:text-white hover:bg-slate-800/50 border border-transparent'
                    }`}
                  >
                    Signal portfolio
                  </button>
                </nav>
              </>
            )}
          </div>
          <div className="relative w-full sm:w-72 max-w-md sm:max-w-none">
            <input
              type="text"
              placeholder="SEARCH STOCK"
              className={`w-full bg-[#131722] border rounded-md py-1.5 pl-3 pr-10 text-sm focus:outline-none focus:border-blue-500 text-white placeholder-gray-500 transition-colors uppercase ${
                searchError ? 'border-red-500/60' : 'border-[#2B3139]'
              }`}
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onKeyDown={handleSearch}
            />
            <button
              type="button"
              onClick={handleSearchClick}
              className="absolute right-2 top-1.5 p-1 rounded text-gray-500 hover:text-white transition-colors"
              aria-label="Search ticker"
            >
              <Search className="w-4 h-4" />
            </button>
            {suggestions.length > 0 && (
              <div className="absolute top-full mt-2 w-full rounded-md border border-slate-700/60 bg-[#131722] shadow-lg z-20 overflow-hidden">
                {suggestions?.map((ticker) => (
                  <button
                    key={ticker}
                    type="button"
                    className="w-full text-left px-3 py-2 text-xs text-gray-300 hover:bg-slate-800/70 hover:text-white transition-colors"
                    onClick={() => runTickerSearch(ticker)}
                  >
                    {ticker}
                  </button>
                ))}
              </div>
            )}
            {searchError && (
              <p className="absolute top-full mt-2 text-[11px] text-red-400">{searchError}</p>
            )}
          </div>
        </div>

        {/* Dynamic View Rendering */}
        <div className="overflow-y-auto flex-1 min-h-0 bg-[#0B0E14]">
          {isSearching && (
            <div className="px-6 py-2 text-xs text-emerald-400 animate-pulse uppercase tracking-wider">
              Loading live data...
            </div>
          )}
          {currentView === 'home' && (
            <div key="home" className="animate-fadeIn">
              <HomeDashboard
                onSelectStock={handleSelectStock}
                watchlistTickers={watchlistTickers}
                onRemoveWatchlist={removeWatchlistTicker}
                onToggleWatchlist={toggleWatchlistTicker}
              />
            </div>
          )}
          {currentView === 'signal-portfolio' && (
            <div key="signal-portfolio" className="animate-fadeIn">
              <SignalDemoPortfolioPage onSelectStock={handleSelectStock} />
            </div>
          )}
          {currentView === 'detail' && selectedStock && (
            <StockDetailPage
              key={`detail-${selectedStock.ticker}`}
              stock={selectedStock}
              onBack={handleBack}
              onInitialSyncComplete={() => setIsSearching(false)}
              isInWatchlist={watchlistTickers.includes(String(selectedStock.ticker || '').toUpperCase())}
              onToggleWatchlist={toggleWatchlistTicker}
            />
          )}
          {currentView === 'stock' && activeStock && (
            <div key="stock" className="animate-fadeIn">
              <StockAnalysisScreen stockData={activeStock} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppInner />
    </ErrorBoundary>
  );
}