const STORAGE_KEY = 'globalMarketPredictor_watchlist_v1';
export const WATCHLIST_MAX = 30;

export function loadWatchlist() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    if (!Array.isArray(parsed)) return [];
    return [...new Set(parsed.map((t) => String(t || '').trim().toUpperCase()).filter(Boolean))].slice(0, WATCHLIST_MAX);
  } catch {
    return [];
  }
}

export function saveWatchlist(tickers) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tickers));
  } catch {
    /* private mode / quota: watchlist just won't persist */
  }
}
