import { useEffect } from 'react';
import { X } from 'lucide-react';
import { CARD_BASE, fmtUsd } from '../lib/format';
import { getStockMeta } from '../data/stocks';
import StockLogo from './StockLogo';
import AlgoExecutionBadge from './AlgoExecutionBadge';

export default function TradeHistoryModal({ open, title, subtitle, rows, onClose, onSelectStock }) {
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
                        {fmtUsd(row.execution_price)}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right text-sm font-semibold tabular-nums ${
                          plShow == null ? 'text-gray-600' : plShow >= 0 ? 'text-emerald-400' : 'text-red-400'
                        }`}
                      >
                        {plShow == null ? '—' : `${plShow >= 0 ? '+' : '−'}${fmtUsd(Math.abs(plShow))}`}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        <div className="shrink-0 px-5 py-3 border-t border-slate-800 bg-[#12131a]/95 text-[11px] text-gray-500">
          {rows.length} row{rows.length !== 1 ? 's' : ''} in this view · paper trading, out-of-sample model signals
        </div>
      </div>
    </div>
  );
}

