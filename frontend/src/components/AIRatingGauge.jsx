import { useState } from 'react';
import { Info } from 'lucide-react';
import { CARD_BASE, getSignal, isNum } from '../lib/format';

// How each engine works (tooltip on the info icon). Matches gmp/ratings.py.
const MODEL_DESCRIPTIONS = {
  'DeepValue Engine': 'Ranks earnings, free-cash-flow and book yield (value), ROE, margins and leverage (quality), and revenue/EPS growth against the 20 tracked peers.',
  'Sentiment NLP': 'Scores the last 7 days of Yahoo Finance and Google News headlines, weighting recent stories more. Few headlines pull the score toward neutral.',
  'Tech-Momentum Net': 'Trend and relative-strength factors (200/50-day averages, MACD, 3-month return vs S&P 500, ADX), blended with the ML forecaster in proportion to its measured out-of-sample skill.',
};

// Multi-Model Consensus panel: one card per engine, then the factor percentiles.
// Every number comes from /api/ratings; an engine without data shows "No data".
export default function AIRatingGauge({ models = [], factors = [], loading = false }) {
  const [tooltipModel, setTooltipModel] = useState(null);

  return (
    <div className={`${CARD_BASE} p-6 flex flex-col h-full`}>

      {/* ── Section header ── */}
      <div className="flex items-center justify-between mb-5">
        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em]">Multi-Model Consensus</p>
        <span className="text-[10px] text-gray-600 tabular-nums">
          {loading ? 'Computing…' : `${models.filter((m) => isNum(m.score)).length} of ${models.length || 3} engines reporting`}
        </span>
      </div>

      {/* ── Model Cards ── */}
      <div className="space-y-3 flex-1">
        {models.map((model) => {
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
                    {isNum(model.score) ? `${sig.label} · ${model.score}` : 'No data'}
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
        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-600 mb-3">Factor Breakdown · percentile vs peers</p>
        <div className="space-y-2.5">
          {factors.map(({ label: fl, score: fs }) => {
            const has = isNum(fs);
            const fc = !has ? '#334155' : fs >= 70 ? '#10b981' : fs >= 48 ? '#eab308' : '#ef4444';
            return (
              <div key={fl} className="flex items-center gap-2.5">
                <span className="text-[11px] text-gray-500 w-[4.5rem] shrink-0">{fl}</span>
                <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{ backgroundColor: fc, width: `${has ? fs : 0}%` }}
                  />
                </div>
                <span className="text-[11px] text-gray-400 tabular-nums w-6 text-right shrink-0">{has ? fs : '—'}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
