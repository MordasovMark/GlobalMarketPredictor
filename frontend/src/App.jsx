import { useCallback, useEffect, useState } from 'react';
import { ArrowLeft, Search } from 'lucide-react';
import ErrorBoundary from './components/ErrorBoundary';
import { TOP_US_STOCKS, getStockMeta } from './data/stocks';
import { MarketDataProvider } from './lib/MarketDataContext';
import { WATCHLIST_MAX, loadWatchlist, saveWatchlist } from './lib/watchlist';
import HomeDashboard from './pages/HomeDashboard';
import SignalPortfolioPage from './pages/SignalPortfolioPage';
import StockDetailPage from './pages/StockDetailPage';

function NavButton({ active, onClick, children, accent = false }) {
  const activeCls = accent ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30' : 'bg-slate-800 text-white border border-transparent';
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-2.5 sm:px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
        active ? activeCls : 'text-gray-400 hover:text-white hover:bg-slate-800/50 border border-transparent'
      }`}
    >
      {children}
    </button>
  );
}

function AppInner() {
  const [searchInput, setSearchInput] = useState('');
  const [searchError, setSearchError] = useState('');
  const [currentView, setCurrentView] = useState('home');
  const [selectedStock, setSelectedStock] = useState(null);
  const [watchlist, setWatchlist] = useState(() => loadWatchlist());

  useEffect(() => saveWatchlist(watchlist), [watchlist]);

  const toggleWatchlist = useCallback((raw) => {
    const t = String(raw || '').trim().toUpperCase();
    if (!getStockMeta(t)) return;
    setWatchlist((prev) => {
      if (prev.includes(t)) return prev.filter((x) => x !== t);
      return prev.length >= WATCHLIST_MAX ? prev : [...prev, t];
    });
  }, []);
  const removeWatchlist = useCallback((raw) => {
    const t = String(raw || '').trim().toUpperCase();
    setWatchlist((prev) => prev.filter((x) => x !== t));
  }, []);

  const query = searchInput.trim().toUpperCase();
  const suggestions = query
    ? TOP_US_STOCKS.filter((s) => s.ticker.includes(query) || s.name.toUpperCase().includes(query)).slice(0, 6)
    : [];

  const openStock = (stock) => {
    setSearchError('');
    setSearchInput('');
    setSelectedStock(stock);
    setCurrentView('detail');
    window.scrollTo?.(0, 0);
  };

  const runSearch = (raw) => {
    const t = raw.trim().toUpperCase();
    if (!t) return;
    const match = getStockMeta(t) || suggestions[0];
    if (!match) {
      setSearchError(`${t} isn't one of the 20 tracked stocks`);
      return;
    }
    openStock(match);
  };

  const goHome = () => {
    setSearchError('');
    setSelectedStock(null);
    setCurrentView('home');
  };

  return (
    <div className="min-h-screen bg-[#0B0E14] text-gray-100 font-sans">
      <div className="w-full flex flex-col min-h-screen min-w-0">
        <div className="min-h-16 border-b border-gray-800 flex flex-wrap items-center px-4 sm:px-6 py-2 gap-y-2 justify-between sticky top-0 bg-[#0B0E14] z-10">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
            {currentView === 'detail' ? (
              <>
                <button type="button" onClick={goHome} className="text-gray-400 hover:text-white flex items-center gap-1 text-sm transition-colors shrink-0">
                  <ArrowLeft className="w-4 h-4" /> Back
                </button>
                <h1 className="text-base sm:text-lg font-semibold text-gray-200 truncate">{selectedStock?.ticker} — Detail</h1>
              </>
            ) : (
              <>
                <h1 className="text-base sm:text-lg font-semibold text-gray-200 truncate shrink-0">Global Market Predictor</h1>
                <nav className="flex items-center gap-1 ml-1 sm:ml-2 pl-2 sm:pl-3 border-l border-gray-800 shrink-0" aria-label="Primary navigation">
                  <NavButton active={currentView === 'home'} onClick={goHome}>
                    Home
                  </NavButton>
                  <NavButton accent active={currentView === 'signal-portfolio'} onClick={() => setCurrentView('signal-portfolio')}>
                    Signal portfolio
                  </NavButton>
                </nav>
              </>
            )}
          </div>
          <div className="relative w-full sm:w-72 max-w-md sm:max-w-none">
            <input
              type="text"
              placeholder="SEARCH STOCK"
              aria-label="Search stock"
              className={`w-full bg-[#131722] border rounded-md py-1.5 pl-3 pr-10 text-sm focus:outline-none focus:border-blue-500 text-white placeholder-gray-500 transition-colors uppercase ${
                searchError ? 'border-red-500/60' : 'border-[#2B3139]'
              }`}
              value={searchInput}
              onChange={(e) => {
                setSearchInput(e.target.value);
                setSearchError('');
              }}
              onKeyDown={(e) => e.key === 'Enter' && runSearch(searchInput)}
            />
            <button
              type="button"
              onClick={() => runSearch(searchInput)}
              className="absolute right-2 top-1.5 p-1 rounded text-gray-500 hover:text-white transition-colors"
              aria-label="Search ticker"
            >
              <Search className="w-4 h-4" />
            </button>
            {suggestions.length > 0 && (
              <div className="absolute top-full mt-2 w-full rounded-md border border-slate-700/60 bg-[#131722] shadow-lg z-20 overflow-hidden">
                {suggestions.map((s) => (
                  <button
                    key={s.ticker}
                    type="button"
                    className="w-full text-left px-3 py-2 text-xs text-gray-300 hover:bg-slate-800/70 hover:text-white transition-colors flex justify-between gap-3"
                    onClick={() => openStock(s)}
                  >
                    <span className="font-semibold">{s.ticker}</span>
                    <span className="text-gray-500 truncate normal-case">{s.name}</span>
                  </button>
                ))}
              </div>
            )}
            {searchError && <p className="absolute top-full mt-2 text-[11px] text-red-400">{searchError}</p>}
          </div>
        </div>

        <main className="flex-1 min-h-0 bg-[#0B0E14]">
          <MarketDataProvider watchlist={watchlist}>
            {currentView === 'home' && (
              <div key="home" className="animate-fadeIn">
                <HomeDashboard onSelectStock={openStock} watchlistTickers={watchlist} onRemoveWatchlist={removeWatchlist} onToggleWatchlist={toggleWatchlist} />
              </div>
            )}
            {currentView === 'signal-portfolio' && (
              <div key="signal-portfolio" className="animate-fadeIn">
                <SignalPortfolioPage onSelectStock={openStock} />
              </div>
            )}
            {currentView === 'detail' && selectedStock && (
              <StockDetailPage
                key={`detail-${selectedStock.ticker}`}
                stock={selectedStock}
                onBack={goHome}
                isInWatchlist={watchlist.includes(selectedStock.ticker)}
                onToggleWatchlist={toggleWatchlist}
              />
            )}
          </MarketDataProvider>
          <footer className="px-6 py-6 text-[11px] text-gray-600 border-t border-gray-800/60">
            Research demo, not investment advice. Forecasts are probabilistic and backtests are hypothetical.
          </footer>
        </main>
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
