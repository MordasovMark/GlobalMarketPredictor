import { useCallback, useMemo, useState } from 'react';
import { Briefcase, Minus, TrendingUp } from 'lucide-react';
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import AlgoExecutionBadge from '../components/AlgoExecutionBadge';
import LiveIndicator from '../components/LiveIndicator';
import StockLogo from '../components/StockLogo';
import TradeHistoryModal from '../components/TradeHistoryModal';
import { getStockMeta } from '../data/stocks';
import { api } from '../lib/api';
import { CARD_BASE, SLOW_POLL_MS, TOOLTIP_STYLE, fmtPct, fmtUsd, gainCls, isNum } from '../lib/format';
import { useMarketData } from '../lib/MarketDataContext';
import { usePolling } from '../lib/usePolling';
import { markHoldings, sumHoldings } from '../lib/portfolio';

const perfStatCardClass = `${CARD_BASE} p-5 border-sky-500/20 w-full text-left transition-all duration-200 cursor-pointer hover:border-sky-400/45 hover:bg-sky-500/[0.08] hover:shadow-[0_0_28px_rgba(56,189,248,0.12)] focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/45 active:scale-[0.99]`;

export default function SignalPortfolioPage({ onSelectStock }) {
  const { quotes, ratings, lastQuoteSync } = useMarketData();
  const { data, loading, error } = usePolling((signal) => api.strategyPortfolio({ signal }), SLOW_POLL_MS);
  const [modal, setModal] = useState(null);
  const closeModal = useCallback(() => setModal(null), []);

  const rows = useMemo(() => markHoldings(data?.holdings, quotes, ratings), [data, quotes, ratings]);
  const totals = useMemo(() => sumHoldings(rows), [rows]);
  const trades = useMemo(() => data?.trades ?? [], [data]);
  const modalRows = useMemo(() => {
    if (!modal) return [];
    if (modal.filterMode === 'ticker') return trades.filter((t) => t.ticker === modal.ticker);
    if (modal.filterMode === 'wins') return trades.filter((t) => t.isWin);
    return trades;
  }, [modal, trades]);

  const best = data?.best_trade;
  const worst = data?.worst_trade;
  const liveEquity = data ? data.cash + totals.marketValue : null;

  return (
    <div className="w-full min-h-full py-6 md:py-8 px-4 md:px-10 box-border overflow-x-hidden" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
      <div className="max-w-6xl mx-auto flex flex-col gap-6">
        <div className={`${CARD_BASE} p-6 md:p-8 border-sky-500/20`}>
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
            <div className="flex items-start gap-3 min-w-0">
              <div className="p-2.5 rounded-xl bg-sky-500/10 border border-sky-500/25 shrink-0">
                <Briefcase className="w-6 h-6 text-sky-400" />
              </div>
              <div>
                <h2 className="text-xl md:text-2xl font-semibold text-white tracking-tight">Signal-based model portfolio</h2>
                <p className="text-sm text-gray-400 mt-2 leading-relaxed max-w-2xl">
                  $100k paper portfolio run by the forecaster on <span className="text-gray-300">out-of-sample</span> signals. Every week it holds the
                  {' '}{data?.rules?.top_k ?? 5} stocks (of 20) with the best risk-adjusted 5-day forecast, equal weight, and only if the forecast beats trading costs.
                  Orders fill at the next session&apos;s open with {data?.rules?.cost_bps_per_side ?? 10} bps costs. Not investment advice.
                </p>
                {data && (
                  <p className="text-xs text-gray-500 mt-3">
                    Backtest window {data.start} → {data.end}. Model {fmtPct(data.total_return_pct)} vs equal-weight benchmark {fmtPct(data.benchmark_return_pct)}.
                  </p>
                )}
              </div>
            </div>
            <LiveIndicator lastSynced={lastQuoteSync} />
          </div>
        </div>

        {loading && <div className={`${CARD_BASE} p-10 text-center text-sm text-sky-300 animate-pulse`}>Running the model portfolio backtest…</div>}
        {!loading && error && <div className={`${CARD_BASE} p-6 text-sm text-amber-300/90`}>Model portfolio unavailable: {error}</div>}

        {data && (
          <>
            <div>
              <div className="flex flex-col gap-0.5 mb-3 px-0.5">
                <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-[0.2em]">STATISTICS</p>
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Performance statistics</h3>
                <p className="text-[11px] text-gray-600 mt-1">Click a card to open the trade blotter, filtered to match the metric.</p>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <button type="button" className={perfStatCardClass} onClick={() => setModal({ filterMode: 'all', title: 'All executed trades', subtitle: `${trades.length} legs since ${data.start}.` })}>
                  <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Total trades executed</p>
                  <p className="text-2xl font-bold text-white tabular-nums tracking-tight">{data.total_trades}</p>
                  <p className="text-[11px] text-gray-500 mt-0.5">{data.closed_trades} closed</p>
                </button>
                <button type="button" className={perfStatCardClass} onClick={() => setModal({ filterMode: 'wins', title: 'Winning trades (realized)', subtitle: 'Sells with positive realized P/L after costs.' })}>
                  <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Win rate</p>
                  <p className="text-2xl font-bold text-white tabular-nums tracking-tight">{isNum(data.win_rate_pct) ? `${data.win_rate_pct.toFixed(1)}%` : '—'}</p>
                  <p className="text-[11px] text-gray-500 mt-0.5">Sharpe {isNum(data.sharpe) ? data.sharpe.toFixed(2) : '—'} · max DD {fmtPct(data.max_drawdown_pct, 1)}</p>
                </button>
                {[
                  ['Best performing trade', best, 'text-emerald-400'],
                  ['Worst performing trade', worst, 'text-red-400'],
                ].map(([label, t, cls]) => (
                  <button
                    key={label}
                    type="button"
                    className={perfStatCardClass}
                    disabled={!t}
                    onClick={() => t && setModal({ filterMode: 'ticker', ticker: t.ticker, title: `Trades · ${t.ticker}`, subtitle: `All executions for ${t.ticker}.` })}
                  >
                    <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">{label}</p>
                    <p className="text-lg font-bold text-white tabular-nums tracking-tight">{t?.ticker ?? '—'}</p>
                    {t && (
                      <>
                        <p className={`text-sm font-semibold tabular-nums mt-1 ${cls}`}>
                          {t.pl >= 0 ? '+' : '−'}
                          {fmtUsd(Math.abs(t.pl))}
                        </p>
                        <p className="text-[11px] text-gray-500 mt-0.5">{fmtPct(t.pl_pct, 1)} round-trip · {t.date}</p>
                      </>
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div className={`${CARD_BASE} p-5`}>
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-1">Equity curve</h3>
              <p className="text-[11px] text-gray-500 mb-3">Model portfolio vs buying all 20 stocks equally on day one.</p>
              <div className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={data.equity_curve} margin={{ top: 8, right: 8, left: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 11 }} minTickGap={40} />
                    <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} width={56} domain={['auto', 'auto']} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: '#e2e8f0' }} formatter={(v, n) => [fmtUsd(v, 0), n]} labelStyle={{ color: '#9ca3af' }} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Area type="monotone" dataKey="model" name="Model portfolio" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.12} strokeWidth={2} isAnimationActive={false} />
                    <Line type="monotone" dataKey="benchmark" name="Equal-weight benchmark" stroke="#94a3b8" strokeDasharray="4 3" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
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
                    <p className="text-[10px] uppercase tracking-wider text-gray-500">Account value</p>
                    <p className="text-white font-semibold">{fmtUsd(liveEquity)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-gray-500">Cash</p>
                    <p className="text-gray-300 font-medium">{fmtUsd(data.cash)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-gray-500">Open P/L</p>
                    <p className={`font-semibold ${gainCls(totals.pl)}`}>
                      {totals.pl >= 0 ? '+' : '−'}
                      {fmtUsd(Math.abs(totals.pl))}
                      <span className="text-[11px] ml-1">({fmtPct(totals.plPct)})</span>
                    </p>
                  </div>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[720px]">
                  <thead className="bg-[#0f1118] border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3 w-12" aria-label="Logo" />
                      {['Symbol', 'AI signal', 'Score', 'Shares', 'Avg cost', 'Price', 'Value', 'P/L'].map((h, i) => (
                        <th key={h} className={`py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider ${i >= 2 ? 'text-right' : ''}`}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.length === 0 && (
                      <tr>
                        <td colSpan={9} className="py-8 text-center text-sm text-gray-500">
                          The model is fully in cash: no stock currently clears its bar.
                        </td>
                      </tr>
                    )}
                    {rows.map((row) => {
                      const stock = getStockMeta(row.ticker);
                      const SigIcon = row.signal?.Icon ?? Minus;
                      return (
                        <tr key={row.ticker} onClick={() => stock && onSelectStock(stock)} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer" title={row.aiSummary}>
                          <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                          <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                          <td className="py-2.5 px-3 text-right">
                            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wide border ${row.signal.bgCls} ${row.signal.textCls} ${row.signal.borderCls}`}>
                              <SigIcon className="w-3.5 h-3.5 shrink-0" />
                              {row.signal.label}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-sm text-gray-200 tabular-nums">{isNum(row.aiScore) ? Math.round(row.aiScore) : '—'}</td>
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

            <div className={`${CARD_BASE} overflow-hidden`}>
              <div className="bg-[#12131a] border-b border-slate-800 px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1.5">
                <div>
                  <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-[0.2em] mb-1">LATEST ACTIVITY</p>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-sky-400 shrink-0" />
                    <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Recent model actions</h3>
                  </div>
                </div>
                <p className="text-[10px] text-gray-600 uppercase tracking-wider">Last 8 executions</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[680px]">
                  <thead className="bg-[#0f1118] border-b border-slate-800">
                    <tr>
                      {['Date', '', 'Symbol', 'Action', 'Shares', 'Execution price'].map((h, i) => (
                        <th key={`${h}-${i}`} className={`py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider ${i >= 4 ? 'text-right' : ''}`}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {trades.slice(0, 8).map((row, idx) => {
                      const stock = getStockMeta(row.ticker);
                      return (
                        <tr key={`${row.date}-${row.ticker}-${idx}`} onClick={() => stock && onSelectStock(stock)} className="border-b border-slate-800/50 hover:bg-white/5 transition-colors cursor-pointer">
                          <td className="py-2.5 px-3 text-sm text-gray-300 tabular-nums">{row.date}</td>
                          <td className="py-2.5 px-3"><StockLogo ticker={row.ticker} /></td>
                          <td className="py-2.5 px-3 font-medium text-white text-sm">{row.ticker}</td>
                          <td className="py-2.5 px-3"><AlgoExecutionBadge action={row.action} /></td>
                          <td className="py-2.5 px-3 text-right text-sm text-gray-300 tabular-nums">{row.shares}</td>
                          <td className="py-2.5 px-3 text-right text-sm text-white tabular-nums">{fmtUsd(row.execution_price)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        <div className={`${CARD_BASE} p-5 border border-slate-800/80`}>
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-gray-500 mb-2">Signal methodology</p>
          <p className="text-sm text-gray-400 leading-relaxed">
            Ratings (0–100) follow the same bands across the app: Strong Buy (85+), Buy (66–84), Hold (41–65), Sell (40 and below). The portfolio
            trades only on forecasts made by models that had not seen that day (walk-forward), so these results are not in-sample. Past performance
            of a backtest does not predict future returns.
          </p>
        </div>
      </div>

      <TradeHistoryModal open={Boolean(modal)} title={modal?.title ?? ''} subtitle={modal?.subtitle} rows={modalRows} onClose={closeModal} onSelectStock={onSelectStock} />
    </div>
  );
}
