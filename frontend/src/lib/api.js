/**
 * Backend client. Every network call in the UI goes through here.
 *
 * Base URL: VITE_API_BASE_URL at build time; otherwise localhost in dev and
 * the Render deployment in production. No API keys ever reach the browser —
 * market data providers are called server-side.
 */
const PROD_API = 'https://globalmarketpredictor.onrender.com';

function resolveApiBaseUrl() {
  const fromEnv = import.meta.env?.VITE_API_BASE_URL;
  if (fromEnv) return fromEnv;
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1') return 'http://localhost:8000';
  }
  return PROD_API;
}

export const API_BASE_URL = resolveApiBaseUrl().replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function getJson(path, { signal, timeoutMs = 25_000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort);
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, { signal: controller.signal });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (body?.detail) detail = String(body.detail);
      } catch {
        /* not JSON */
      }
      throw new ApiError(detail, res.status);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

const q = encodeURIComponent;

export const api = {
  health: (opts) => getJson('/health', opts),
  model: (opts) => getJson('/api/model', opts),
  universe: (opts) => getJson('/api/universe', opts),
  quotes: (tickers, opts) => getJson(`/api/quotes?tickers=${q(tickers.join(','))}`, opts),
  market: (opts) => getJson('/api/market', opts),
  analyze: (ticker, range, opts) => getJson(`/api/analyze?ticker=${q(ticker)}&time_range=${q(range)}`, opts),
  news: (ticker, limit = 4, opts) => getJson(`/api/news/${q(ticker)}?limit=${limit}`, opts),
  fearGreed: (opts) => getJson('/api/fear-greed', opts),
  ratings: (opts) => getJson('/api/ratings', opts),
  rating: (ticker, opts) => getJson(`/api/ratings/${q(ticker)}`, opts),
  simulate: (ticker, range, opts) =>
    getJson(`/api/portfolio/simulate?ticker=${q(ticker)}&time_range=${q(range)}`, opts),
  strategyPortfolio: (opts) => getJson('/api/strategy/portfolio', opts),
};
