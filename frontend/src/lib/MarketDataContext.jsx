import { createContext, useContext, useMemo } from 'react';
import { api } from './api';
import { LIVE_POLL_MS, SLOW_POLL_MS } from './format';
import { usePolling } from './usePolling';
import { TOP_US_STOCKS } from '../data/stocks';

const MarketDataContext = createContext(null);

/**
 * App-wide market state, fetched once and shared by every page:
 * - universe: fundamentals + 52-week range (server refreshes every 15 min)
 * - quotes:   live prices for the dashboard + watchlist (every 60 s)
 * - ratings:  multi-engine consensus scores (every 5 min)
 */
export function MarketDataProvider({ watchlist = [], children }) {
  const tickers = useMemo(
    () => [...new Set([...TOP_US_STOCKS.map((s) => s.ticker), ...watchlist])],
    [watchlist],
  );
  const tickerKey = tickers.join(',');

  const universe = usePolling((signal) => api.universe({ signal }), SLOW_POLL_MS);
  const quotes = usePolling((signal) => api.quotes(tickerKey.split(','), { signal }), LIVE_POLL_MS, [tickerKey]);
  const ratings = usePolling((signal) => api.ratings({ signal }), SLOW_POLL_MS);

  const value = useMemo(() => {
    const byTicker = Object.fromEntries((universe.data || []).map((row) => [row.ticker, row]));
    const live = quotes.data || {};
    const stocks = TOP_US_STOCKS.map((meta) => {
      const row = byTicker[meta.ticker] || {};
      const q = live[meta.ticker];
      return {
        ...meta,
        ...row,
        price: q?.price ?? row.price ?? null,
        change: q?.change ?? row.change ?? null,
        changePercent: q?.changePercent ?? row.changePercent ?? null,
      };
    });
    const ratingMap = Object.fromEntries((ratings.data?.ratings || []).map((r) => [r.ticker, r]));
    return {
      stocks,
      quotes: live,
      ratings: ratingMap,
      ratingsStatus: { loading: ratings.loading, error: ratings.error },
      universeStatus: { loading: universe.loading, error: universe.error },
      lastQuoteSync: quotes.lastSynced,
      quotesSyncing: quotes.loading,
    };
  }, [universe, quotes, ratings]);

  return <MarketDataContext.Provider value={value}>{children}</MarketDataContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useMarketData() {
  const ctx = useContext(MarketDataContext);
  if (!ctx) throw new Error('useMarketData must be used inside MarketDataProvider');
  return ctx;
}
