import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { CARD_BASE } from '../lib/format';

function MarketIntelligence({ ticker, items = [], isLoading = false, error = '' }) {
  if (!isLoading && items.length === 0 && !error) return null;

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
        <span className="text-[10px] font-medium text-gray-600 uppercase tracking-wider">
          AI Sentiment · Live RSS · {ticker}
        </span>
      </div>

      {/* ── News cards grid ── */}
      {error && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-200 mb-3">
          Live news is temporarily unavailable: {error}
        </div>
      )}
      {isLoading && items.length === 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {[0, 1, 2, 3].map((idx) => (
            <div key={idx} className="rounded-xl border border-slate-800 bg-[#0f1118] p-4">
              <div className="h-4 w-20 rounded bg-slate-800 animate-pulse mb-4" />
              <div className="h-3 w-full rounded bg-slate-800 animate-pulse mb-2" />
              <div className="h-3 w-4/5 rounded bg-slate-800 animate-pulse" />
            </div>
          ))}
        </div>
      )}
      {items.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {items.map((item, idx) => {
          const isGood    = item.sentiment === 'good';
          const isBad     = item.sentiment === 'bad';

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
      )}
    </div>
  );
}

export default MarketIntelligence;
