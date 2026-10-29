import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../lib/api';
import { CARD_BASE, TOOLTIP_STYLE, fmtPct, fmtUsd, gainCls, isNum } from '../lib/format';
import { usePolling } from '../lib/usePolling';

function Stat({ label, value, cls = 'text-white', sub }) {
  return (
    <div className="rounded-lg border border-slate-800/70 bg-slate-900/35 px-4 py-3">
      <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-lg font-bold tabular-nums ${cls}`}>{value}</p>
      {sub ? <p className="text-[10px] text-gray-600 mt-0.5">{sub}</p> : null}
    </div>
  );
}

/** $10k paper trade of one stock, driven by the model's out-of-sample signals (/api/portfolio/simulate). */
export default function PortfolioSimulator({ ticker, timeRange, accentColor }) {
  const { data, loading, error } = usePolling(
    (signal) => api.simulate(ticker, timeRange, { signal }),
    0,
    [ticker, timeRange],
  );
  const trades = Array.isArray(data?.trades) ? data.trades : [];
  const roi = data?.total_roi_pct;
  const bh = data?.buy_hold_roi_pct;

  return (
    <div className={`${CARD_BASE} p-6 mb-6`}>
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.18em]">Mock Portfolio (Algo Trading)</h3>
        <p className="text-[11px] text-gray-600 mt-1.5 leading-relaxed">
          $10k paper trade of <span className="text-gray-400 font-medium">{ticker}</span> over{' '}
          <span className="text-gray-400 uppercase">{data?.window ?? timeRange}</span> using the model&apos;s{' '}
          <span className="text-gray-400">out-of-sample</span> 5-day signals. Fills at the next day&apos;s open with{' '}
          {data?.cost_bps_per_side ?? 10} bps costs per trade. Compared with simply buying and holding.
        </p>
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-14 gap-4">
          <span className="relative inline-flex h-5 w-5 rounded-full animate-ping" style={{ backgroundColor: accentColor }} />
          <p className="text-sm font-medium tracking-widest uppercase animate-pulse" style={{ color: accentColor }}>
            Running simulation…
          </p>
        </div>
      )}

      {!loading && error && <p className="text-sm text-amber-300/90 py-6 text-center">{error}</p>}

      {!loading && !error && data && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <Stat label="Initial Balance" value={fmtUsd(data.initial_balance)} />
            <Stat label="Final Balance" value={fmtUsd(data.final_balance)} />
            <Stat label="Model ROI" value={fmtPct(roi)} cls={gainCls(roi)} />
            <Stat
              label="Buy & Hold ROI"
              value={fmtPct(bh)}
              cls={gainCls(bh)}
              sub={isNum(roi) && isNum(bh) ? `${roi >= bh ? 'Model beat' : 'Model trailed'} by ${Math.abs(roi - bh).toFixed(2)} pts` : null}
            />
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5 text-xs">
            {[
              ['Sharpe', isNum(data.sharpe) ? data.sharpe.toFixed(2) : '—'],
              ['Max drawdown', fmtPct(data.max_drawdown_pct)],
              ['Win rate', isNum(data.win_rate_pct) ? `${data.win_rate_pct.toFixed(0)}%` : '—'],
              ['Time in market', isNum(data.exposure_pct) ? `${data.exposure_pct.toFixed(0)}%` : '—'],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between rounded-md border border-slate-800/60 px-3 py-2">
                <span className="text-gray-500 uppercase tracking-wider text-[10px]">{k}</span>
                <span className="text-gray-200 tabular-nums font-semibold">{v}</span>
              </div>
            ))}
          </div>

          {data.equity_curve?.length > 1 && (
            <div className="h-28 mb-5">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.equity_curve} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
                  <XAxis dataKey="date" hide />
                  <YAxis hide domain={['auto', 'auto']} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [fmtUsd(v), 'Equity']} labelStyle={{ color: '#9ca3af' }} />
                  <Area type="monotone" dataKey="equity" stroke={accentColor} fill={accentColor} fillOpacity={0.12} strokeWidth={1.5} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="border-t border-slate-800/60 pt-4">
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em] mb-3">Trade history</p>
            <div className="rounded-lg border border-slate-800/70 overflow-hidden bg-[#0f1118] max-h-[220px] overflow-y-auto">
              <table className="w-full text-left border-collapse text-sm" aria-label="Simulated trades">
                <thead className="sticky top-0 bg-[#0f1118] border-b border-slate-800 z-[1]">
                  <tr>
                    {['Date', 'Action', 'Shares', 'Execution Price', 'P/L'].map((h, i) => (
                      <th key={h} className={`py-2.5 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider ${i >= 2 ? 'text-right' : ''}`}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {trades.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="py-8 px-3 text-center text-xs text-gray-500">
                        No trades in this window: the model never cleared its BUY bar.
                      </td>
                    </tr>
                  ) : (
                    trades.map((row, idx) => {
                      const isBuy = row.action === 'BUY';
                      return (
                        <tr key={`${row.date}-${idx}`} className="border-b border-slate-800/40 last:border-0 hover:bg-white/[0.03]">
                          <td className="py-2.5 px-3 text-gray-300 tabular-nums text-xs">{row.date}</td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`inline-flex items-center text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-md border ${
                                isBuy ? 'text-emerald-400 bg-emerald-400/10 border-emerald-500/30' : 'text-red-400 bg-red-400/10 border-red-500/30'
                              }`}
                            >
                              {row.action}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-gray-300 tabular-nums">{row.shares}</td>
                          <td className="py-2.5 px-3 text-right text-gray-200 tabular-nums">{fmtUsd(row.execution_price)}</td>
                          <td className={`py-2.5 px-3 text-right tabular-nums ${gainCls(row.pnl)}`}>
                            {isNum(row.pnl) ? `${row.pnl >= 0 ? '+' : '−'}${fmtUsd(Math.abs(row.pnl))}` : '—'}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
            {data.open_position && (
              <p className="text-[11px] text-gray-500 mt-2">
                Still holding {data.open_position.shares} shares bought at {fmtUsd(data.open_position.entry_price)}.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
