import { useMemo } from 'react';
import { Briefcase, Minus, Star, TrendingUp, X } from 'lucide-react';
import { Area, CartesianGrid, ComposedChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import LiveIndicator from '../components/LiveIndicator';
import SentimentGauge from '../components/SentimentGauge';
import StockLogo from '../components/StockLogo';
import { getStockMeta } from '../data/stocks';
import { api } from '../lib/api';
import { CARD_BASE, LIVE_POLL_MS, NEWS_POLL_MS, SLOW_POLL_MS, fmtPct, fmtUsd, gainCls, getSignal, isNum } from '../lib/format';
import { useMarketData } from '../lib/MarketDataContext';
import { markHoldings, sumHoldings } from '../lib/portfolio';
import { usePolling } from '../lib/usePolling';
import { WATCHLIST_MAX } from '../lib/watchlist';

const MACRO_NAMES = { SPY: 'SPY (S&P 500)', QQQ: 'QQQ (Nasdaq)', IWM: 'IWM (Russell 2000)' };

function SignalChip({ signal, compact = false }) {
  const Icon = signal?.Icon ?? Minus;
  return (
    <span
      className={`inline-flex items-center gap-1 ${compact ? 'px-2 py-0.5' : 'px-2 py-1'} rounded-md text-[10px] font-semibold uppercase tracking-wide border ${signal.bgCls} ${signal.textCls} ${signal.borderCls}`}
    >
      <Icon className="w-3 h-3 shrink-0" />
      {signal.label}
    </span>
  );
}

function Headline({ item }) {
  return (
    <li className="text-sm text-gray-400 leading-snug">
      {item.link ? (
        <a href={item.link} target="_blank" rel="noreferrer" className="text-gray-300 hover:text-white">
          {item.headline}
        </a>
      ) : (
        <span className="text-gray-300">{item.headline}</span>
      )}
      <span className="block text-xs text-gray-500 mt-1">{[item.source, item.time].filter(Boolean).join(' · ')}</span>
    </li>
  );
}

/** Rank Buy-rated names by consensus score, nudged by live news tone (18%). */
function buildTopPicks(ratings) {
  return Object.values(ratings)
    .filter((r) => isNum(r.score) && r.score >= 66)
    .map((r) => ({
      ...r,
      signal: getSignal(r.score),
      pickScore: r.score * 0.82 + (r.news?.score ?? 50) * 0.18,
    }))
    .sort((a, b) => b.pickScore - a.pickScore)
    .slice(0, 5)
    .map((r, i) => ({ ...r, rank: i + 1 }));
}

export default function HomeDashboard({ onSelectStock, watchlistTickers, onRemoveWatchlist, onToggleWatchlist }) {
  const { stocks, quotes, ratings, ratingsStatus, lastQuoteSync, quotesSyncing } = useMarketData();
  const fg = usePolling((signal) => api.fearGreed({ signal }), NEWS_POLL_MS);
  const market = usePolling((signal) => api.market({ signal }), LIVE_POLL_MS);
  const portfolio = usePolling((signal) => api.strategyPortfolio({ signal }), SLOW_POLL_MS);

  const topPicks = useMemo(() => buildTopPicks(ratings), [ratings]);
  const holdings = useMemo(() => markHoldings(portfolio.data?.holdings, quotes, ratings), [portfolio.data, quotes, ratings]);
  const holdingTotals = useMemo(() => sumHoldings(holdings), [holdings]);
  const watchlistRows = useMemo(
    () =>
      (watchlistTickers || []).map((ticker) => ({
        ticker,
        name: getStockMeta(ticker)?.name ?? ticker,
        price: quotes[ticker]?.price ?? null,
        changePercent: quotes[ticker]?.changePercent ?? null,
      })),
    [watchlistTickers, quotes],
  );
  const pricedStocks = stocks.filter((s) => isNum(s.changePercent));
  const fgData = fg.data;

  return (
    <div className="w-full min-h-full py-6 md:py-8 box-border overflow-x-hidden" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
      <div className="w-full px-4 md:px-10 flex flex-col gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full min-w-0">
          <div className={`${CARD_BASE} p-6 w-full flex flex-col min-h-[520px] lg:col-span-1`}>
            <div className="flex items-center gap-2 mb-3 w-full text-left">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.2em]">US Market Sentiment</h3>
            </div>
            <div className="flex-1 flex items-center justify-center">
              {fgData ? (
                <SentimentGauge value={fgData.value} history={fgData.history} />
              ) : (
                <p className="text-sm text-gray-500">{fg.error ? `Unavailable: ${fg.error}` : 'Computing Fear & Greed…'}</p>
              )}
            </div>
            {fgData?.components?.length > 0 && (
              <div className="mt-4 space-y-1.5">
                {fgData.components.map((c) => (
                  <div key={c.key} className="flex items-center gap-2" title={c.description}>
                    <span className="text-[10px] text-gray-500 w-32 shrink-0 truncate">{c.name}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${c.score}%`, background: c.score < 45 ? '#f97316' : c.score > 55 ? '#84cc16' : '#eab308' }} />
                    </div>
                    <span className="text-[10px] text-gray-400 tabular-nums w-6 text-right">{Math.round(c.score)}</span>
                  </div>
                ))}
              </div>
            )}
            <div className="mt-4 pt-3 border-t border-slate-800/60">
              <LiveIndicator lastSynced={fg.lastSynced} />
            </div>
          </div>

          <div className={`${CARD_BASE} p-6 w-full min-h-[520px] lg:col-span-2 flex flex-col`}>
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.2em] mb-1 w-full text-left">Fear &amp; Greed Timeline</h3>
            <p className="text-xs text-gray-500 mb-4 uppercase tracking-wider">Month-end index vs S&amp;P 500, last 12 months</p>
            <div className="w-full h-[280px] flex-shrink-0">
              {fgData?.timeline?.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={fgData.timeline} margin={{ top: 8, right: 8, left: 4, bottom: 4 }}>
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
                    <YAxis yAxisId="left" domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 12 }} stroke="rgba(255,255,255,0.06)" width={36} />
                    <YAxis yAxisId="right" orientation="right" domain={['auto', 'auto']} tick={{ fill: '#6b7280', fontSize: 12 }} stroke="rgba(255,255,255,0.06)" width={48} tickFormatter={(v) => `${(v / 1000).toFixed(1)}k`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(71, 85, 105, 0.5)', borderRadius: '12px', padding: '10px 14px' }}
                      itemStyle={{ color: '#e2e8f0', fontSize: 12 }}
                      formatter={(val, name) => [name === 'Fear & Greed' ? val : Number(val).toLocaleString(), name]}
                    />
                    <Area yAxisId="left" type="linear" dataKey="fearGreed" fill="url(#fgGradTimeline)" stroke="#eab308" strokeWidth={1.5} name="Fear & Greed" isAnimationActive={false} />
                    <Area yAxisId="right" type="linear" dataKey="sp500" fill="url(#spGradTimeline)" stroke="#93c5fd" strokeWidth={1.5} name="S&P 500" isAnimationActive={false} />
                    <Legend align="right" verticalAlign="top" iconType="line" iconSize={8} wrapperStyle={{ paddingLeft: '12px', fontSize: '12px' }} />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-sm text-gray-600">{fg.loading ? 'Loading timeline…' : 'Timeline unavailable'}</div>
              )}
            </div>

            <div className="mt-6 pt-6 border-t border-slate-800/60 flex-1">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-5">
                <div className="rounded-lg border-l-[3px] border-red-500/30 bg-red-950/10 border border-slate-800/60 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-red-400/90 mb-2.5">Fear Drivers</p>
                  <ul className="space-y-3">
                    {(fgData?.drivers?.fear || []).map((item) => <Headline key={item.headline} item={item} />)}
                    {fgData && !fgData.drivers?.fear?.length && <li className="text-xs text-gray-600">No strongly negative market headlines right now.</li>}
                  </ul>
                </div>
                <div className="rounded-lg border-l-[3px] border-green-500/30 bg-green-950/10 border border-slate-800/60 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.15em] text-green-400/90 mb-2.5">Greed Drivers</p>
                  <ul className="space-y-3">
                    {(fgData?.drivers?.greed || []).map((item) => <Headline key={item.headline} item={item} />)}
                    {fgData && !fgData.drivers?.greed?.length && <li className="text-xs text-gray-600">No strongly positive market headlines right now.</li>}
                  </ul>
                </div>
              </div>
              <div className="rounded-lg border border-slate-800/80 bg-[#12131a] px-5 py-4" style={{ borderLeftWidth: '3px', borderLeftColor: 'rgba(139, 92, 246, 0.5)' }}>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-300 mb-2">AI Sentiment Conclusion</p>
                <p className="text-sm text-gray-300 leading-relaxed">{fgData?.conclusion ?? '—'}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 w-full">
          {(market.data || ['SPY', 'QQQ', 'IWM'].map((symbol) => ({ symbol }))).map((m) => (
            <div key={m.symbol} className={`${CARD_BASE} p-5 flex flex-col justify-center`}>
              <h3 className="text-gray-400 text-base font-medium">{MACRO_NAMES[m.symbol] ?? m.symbol}</h3>
              <p className="text-2xl md:text-3xl font-bold text-white mt-1.5 tabular-nums">{isNum(m.price) ? `${Number(m.price).toFixed(2)} USD` : '—'}</p>
              <p className={`text-base mt-1.5 tabular-nums ${gainCls(m.change_pct)}`}>{fmtPct(m.change_pct)}</p>
            </div>
          ))}
        </div>

        {/* Top AI stock picks */}
        <div className={`${CARD_BASE} overflow-hidden w-full`}>
          <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-2 min-w-0">
              <TrendingUp className="w-4 h-4 text-emerald-400 shrink-0" />
              <div>
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Top 5 AI Stock Picks</h3>
                <p className="text-[10px] text-gray-500 mt-0.5">Buy-rated names ranked by consensus score plus live news tone.</p>
              </div>
            </div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">{ratingsStatus.loading ? 'Computing ratings…' : `${Object.keys(ratings).length} stocks rated`}</div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse min-w-[760px]">
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  {['Symbol', 'AI signal', 'AI score', 'Why'].map((h, i) => (
                    <th key={h} className={`py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider ${i === 2 ? 'text-right' : ''}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {topPicks.length === 0 && (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-sm text-gray-500">
                      {ratingsStatus.loading
                        ? 'Ratings are being computed…'
                        : ratingsStatus.error
                          ? `Ratings unavailable: ${ratingsStatus.error}`
                          : 'No stock currently earns a Buy: fewer than 2 of 3 engines agree on any name.'}
                    </td>
                  </tr>
                )}
                {topPicks.map((row) => {
                  const stock = getStockMeta(row.ticker);
                  return (
                    <tr key={row.ticker} onClick={() => stock && onSelectStock(stock)} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer">
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-3 min-w-0">
                          <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-md border border-slate-800 bg-[#0f1118] px-1.5 text-[10px] font-semibold text-gray-500 tabular-nums">
                            #{row.rank}
                          </span>
                          <StockLogo ticker={row.ticker} />
                          <div className="min-w-0">
                            <p className="font-medium text-white text-sm">{row.ticker}</p>
                            <p className="text-[11px] text-gray-500 truncate max-w-[11rem]">{stock?.name}</p>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-3"><SignalChip signal={row.signal} compact /></td>
                      <td className="py-3 px-3 text-right text-sm font-semibold text-white tabular-nums">
                        {Math.round(row.score)}
                        <span className="text-[11px] font-normal text-gray-500">/100</span>
                      </td>
                      <td className="py-3 px-3 text-sm text-gray-300 leading-snug">
                        <div className="flex flex-col gap-1">
                          <span>{row.summary}</span>
                          <span className={`text-[10px] uppercase tracking-wider ${row.news?.is_live ? 'text-emerald-400' : 'text-gray-600'}`}>
                            {row.news?.is_live ? `${row.news.headline_count} live headlines scored` : 'No recent headlines'}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Model portfolio (summary) */}
        <div className={`${CARD_BASE} overflow-hidden w-full`}>
          <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-2 min-w-0">
              <Briefcase className="w-4 h-4 text-sky-400 shrink-0" />
              <div>
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Model Portfolio</h3>
                <p className="text-[10px] text-gray-500 mt-0.5">
                  Current positions of the AI paper portfolio. Open <span className="text-gray-400">Signal portfolio</span> for the full backtest.
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-4 text-xs sm:text-sm tabular-nums">
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Market value</p>
                <p className="text-white font-semibold">{fmtUsd(holdingTotals.marketValue)}</p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Backtest return</p>
                <p className={`font-semibold ${gainCls(portfolio.data?.total_return_pct)}`}>{fmtPct(portfolio.data?.total_return_pct)}</p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wider text-gray-500">Benchmark</p>
                <p className={`font-medium ${gainCls(portfolio.data?.benchmark_return_pct)}`}>{fmtPct(portfolio.data?.benchmark_return_pct)}</p>
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse min-w-[640px]">
              <thead className="bg-[#0f1118] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 w-12" aria-label="Logo" />
                  {['Symbol', 'AI signal', 'Shares', 'Avg cost', 'Price', 'Value', 'P/L'].map((h, i) => (
                    <th key={h} className={`py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider ${i >= 2 ? 'text-right' : ''}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {holdings.length === 0 && (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-sm text-gray-500">
                      {portfolio.loading ? 'Loading model portfolio…' : portfolio.error ? `Unavailable: ${portfolio.error}` : 'The model is in cash right now.'}
                    </td>
                  </tr>
                )}
                {holdings.map((row) => {
                  const stock = getStockMeta(row.ticker);
                  return (
                    <tr key={row.ticker} onClick={() => stock && onSelectStock(stock)} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer">
                      <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 text-right"><SignalChip signal={row.signal} compact /></td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{row.shares}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{fmtUsd(row.avgCost)}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-white tabular-nums">{fmtUsd(row.price)}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-200 tabular-nums font-medium">{fmtUsd(row.marketValue)}</td>
                      <td className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${gainCls(row.pl)}`}>
                        {row.pl >= 0 ? '+' : '−'}
                        {fmtUsd(Math.abs(row.pl))}
                        <span className="text-[11px] font-normal ml-1">({fmtPct(row.plPct, 1)})</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Watchlist */}
        <div className={`${CARD_BASE} overflow-hidden w-full`}>
          <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex items-center gap-2">
            <Star className="w-4 h-4 text-amber-400 shrink-0 fill-amber-400/25" />
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider truncate">My Watchlist</h3>
            <span className="text-[10px] text-gray-500 tabular-nums shrink-0">
              {watchlistRows.length}/{WATCHLIST_MAX}
            </span>
          </div>
          {watchlistRows.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-gray-500">
              No symbols yet. Open a stock and use <span className="text-gray-400">Add to watchlist</span>.
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <tbody>
                {watchlistRows.map((row) => {
                  const stock = getStockMeta(row.ticker);
                  return (
                    <tr key={row.ticker} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer" onClick={() => stock && onSelectStock(stock)}>
                      <td className="py-2.5 px-3 w-12"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 text-sm text-gray-400 truncate hidden sm:table-cell max-w-[10rem]">{row.name}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{fmtUsd(row.price)}</td>
                      <td className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${gainCls(row.changePercent)}`}>{fmtPct(row.changePercent)}</td>
                      <td className="py-2 px-2 w-10">
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
            {quotesSyncing && <span className="text-[10px] text-emerald-400 animate-pulse uppercase tracking-wider">Syncing...</span>}
            <LiveIndicator lastSynced={lastQuoteSync} />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full items-start">
          {/* Top gainers */}
          <div className={`${CARD_BASE} overflow-hidden`}>
            <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Top 10 Gainers Today</h3>
            </div>
            <table className="w-full text-left border-collapse">
              <tbody>
                {[...pricedStocks]
                  .sort((a, b) => b.changePercent - a.changePercent)
                  .slice(0, 10)
                  .map((row) => (
                    <tr key={row.ticker} onClick={() => onSelectStock(row)} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer">
                      <td className="py-2.5 px-3 w-12"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{fmtUsd(row.price)}</td>
                      <td className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${gainCls(row.changePercent)}`}>{fmtPct(row.changePercent)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Ratings */}
          <div className={`${CARD_BASE} overflow-hidden`}>
            <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">AI Pro Predictions</h3>
            </div>
            <table className="w-full text-left border-collapse">
              <tbody>
                {[...stocks]
                  .sort((a, b) => (ratings[b.ticker]?.score ?? -1) - (ratings[a.ticker]?.score ?? -1))
                  .slice(0, 10)
                  .map((row) => {
                    const score = ratings[row.ticker]?.score;
                    const sig = getSignal(score);
                    const SigIcon = sig.Icon;
                    const watched = (watchlistTickers || []).includes(row.ticker);
                    return (
                      <tr key={row.ticker} onClick={() => onSelectStock(row)} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer">
                        <td className="py-2 px-1 w-8 text-center">
                          <button
                            type="button"
                            className={`inline-flex p-0.5 rounded transition-colors ${watched ? 'text-amber-400' : 'text-slate-600 hover:text-amber-400/90'}`}
                            aria-label={watched ? `Remove ${row.ticker} from watchlist` : `Add ${row.ticker} to watchlist`}
                            onClick={(e) => {
                              e.stopPropagation();
                              onToggleWatchlist?.(row.ticker);
                            }}
                          >
                            <Star className={`w-3.5 h-3.5 ${watched ? 'fill-amber-400' : ''}`} />
                          </button>
                        </td>
                        <td className="py-2.5 px-2 w-10"><StockLogo ticker={row.ticker} /></td>
                        <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                        <td className="py-2.5 px-3">
                          <span className={`inline-flex items-center gap-1 text-xs font-medium ${sig.textCls}`}>
                            <SigIcon className="w-3.5 h-3.5 shrink-0" />
                            {sig.label}
                          </span>
                        </td>
                        <td className={`py-2.5 px-3 text-right text-sm tabular-nums font-semibold ${sig.textCls}`}>{isNum(score) ? Math.round(score) : '—'}</td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>

          {/* Market giants */}
          <div className={`${CARD_BASE} overflow-hidden`}>
            <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Market Giants</h3>
            </div>
            <table className="w-full text-left border-collapse">
              <tbody>
                {[...stocks]
                  .sort((a, b) => (b.marketCap ?? 0) - (a.marketCap ?? 0))
                  .slice(0, 10)
                  .map((row) => (
                    <tr key={row.ticker} onClick={() => onSelectStock(row)} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer">
                      <td className="py-2.5 px-3 w-12"><StockLogo ticker={row.ticker} /></td>
                      <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                      <td className="py-2.5 px-3 text-sm text-gray-400 truncate max-w-[9rem]" title={row.name}>{row.name}</td>
                      <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums font-medium">
                        {isNum(row.marketCap) ? `$${(row.marketCap / 1000).toFixed(2)}T` : fmtUsd(row.price)}
                      </td>
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
