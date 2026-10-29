import { ShieldCheck } from 'lucide-react';
import { CARD_BASE, isNum } from '../lib/format';

const pct = (v) => (isNum(v) ? `${(v * 100).toFixed(1)}%` : '—');

/**
 * Honest scorecard: how the forecaster did on data it never saw
 * (walk-forward), next to the naive baseline it has to beat.
 */
export default function ModelTrackRecord({ model }) {
  if (!model || model.status !== 'ready') {
    return (
      <div className={`${CARD_BASE} p-5 mb-6 text-sm text-gray-500`}>
        Forecast model status: <span className="text-gray-300">{model?.status ?? 'loading'}</span>
      </div>
    );
  }
  const m = model.oos_5d || {};
  const conf = model.skill_confidence ?? 0;
  const rows = [
    ['Direction hit rate (5d)', pct(m.hit_rate), `"Always up" baseline: ${pct(m.always_up_hit_rate)}`],
    ['80% band coverage', pct(m.interval_coverage_80), 'Target: 80.0%. Closer is better calibrated.'],
    ['Rank IC (5d)', isNum(m.rank_ic_mean) ? m.rank_ic_mean.toFixed(3) : '—', `t-stat ${isNum(m.rank_ic_tstat) ? m.rank_ic_tstat.toFixed(1) : '—'} (|t| > 2 ≈ significant)`],
  ];
  return (
    <div className={`${CARD_BASE} p-5 mb-6`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-sky-400" />
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-[0.18em]">Model track record · out-of-sample</h3>
        </div>
        <span className="text-[10px] text-gray-600 uppercase tracking-wider">
          Trained {model.trained_at?.slice(0, 10)} · data to {model.data_end} · {model.n_tickers} stocks
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {rows.map(([label, value, sub]) => (
          <div key={label} className="rounded-lg border border-slate-800/70 bg-slate-900/35 px-4 py-3">
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">{label}</p>
            <p className="text-lg font-bold text-white tabular-nums">{value}</p>
            <p className="text-[10px] text-gray-600 mt-0.5">{sub}</p>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-gray-500 mt-3 leading-relaxed">
        {conf > 0
          ? `The model's direction calls carry ${Math.round(conf * 100)}% weight in the Tech-Momentum engine, set by its measured skill.`
          : 'Direction skill is not statistically significant yet, so the rating ignores the model’s up/down call and uses it only for the risk band. The weekly retrain updates this automatically.'}
      </p>
    </div>
  );
}
