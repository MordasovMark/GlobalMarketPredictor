import { useState } from 'react';
import { CARD_BASE } from '../lib/format';

// eslint-disable-next-line react-refresh/only-export-components
export function getSentimentLabelAndColor(value) {
  const v = Math.min(100, Math.max(0, value));
  if (v <= 24) return { label: 'Extreme Fear', color: '#ef4444', zone: 'extremeFear' };
  if (v <= 44) return { label: 'Fear', color: '#f97316', zone: 'fear' };
  if (v <= 55) return { label: 'Neutral', color: '#eab308', zone: 'neutral' };
  if (v <= 75) return { label: 'Greed', color: '#84cc16', zone: 'greed' };
  return { label: 'Extreme Greed', color: '#22c55e', zone: 'extremeGreed' };
}

function getSentimentDetails(score) {
  const { label, color } = getSentimentLabelAndColor(score);
  return { label, color };
}

// Spec: trader-focused actionable insight per zone (one sentence below score)
const SENTIMENT_INSIGHTS = {
  extremeFear: 'Market is oversold. Contrarian buying opportunity often identified.',
  fear: 'Risk-off sentiment. Consider defensive positioning or selective entries.',
  neutral: 'Sentiment balanced. Range-bound conditions typical.',
  greed: 'Risk-on. Stay selective with new entries; consider trimming extremes.',
  extremeGreed: 'Euphoria. Consider taking some risk off the table.',
};

// Zone-specific glow for dynamic box-shadow (spec: deep red → green by zone)
const ZONE_GLOW = {
  extremeFear: { color: '#991b1b', blur: 48, spread: 0, alpha: 0.35 },
  fear:         { color: '#c2410c', blur: 40, spread: 0, alpha: 0.28 },
  neutral:      { color: '#eab308', blur: 32, spread: 0, alpha: 0.18 },
  greed:        { color: '#65a30d', blur: 40, spread: 0, alpha: 0.28 },
  extremeGreed: { color: '#166534', blur: 48, spread: 0, alpha: 0.35 },
};

function hexToRgba(hex, alpha) {
  const match = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
  if (!match) return `rgba(0,0,0,${alpha})`;
  return `rgba(${parseInt(match[1], 16)}, ${parseInt(match[2], 16)}, ${parseInt(match[3], 16)}, ${alpha})`;
}

// Needle: pivot at arc center (50,50). rotation = (score/100)*180 - 90 → 0=left, 50=top, 100=right.
// `history` comes from /api/fear-greed: [{ period: 'Current' | '1 Week Ago' | ..., value }].
export default function SentimentGauge({ value = 50, history = [] }) {
  const [hoveredScore, setHoveredScore] = useState(null);
  const displayScore = hoveredScore !== null ? hoveredScore : value;
  const rotation = (displayScore / 100) * 180 - 90;

  const { label: currentLabel, color: currentColor } = getSentimentDetails(displayScore);
  const { zone } = getSentimentLabelAndColor(displayScore);
  const insightText = SENTIMENT_INSIGHTS[zone] || SENTIMENT_INSIGHTS.neutral;

  const trendData = history
    .filter((row) => Number.isFinite(Number(row?.value)))
    .map((row) => ({ period: row.period, periodKey: row.period, score: Number(row.value), value: Number(row.value) }));

  const glowConfig = ZONE_GLOW[zone] || ZONE_GLOW.neutral;
  const glowRgba = hexToRgba(glowConfig.color, glowConfig.alpha);
  const glowRgbaPulse = hexToRgba(glowConfig.color, glowConfig.alpha * 0.4);
  const scoreGlow = hexToRgba(currentColor, 0.28);

  return (
    <div
      className={`relative flex flex-col items-center w-full rounded-2xl p-5 box-border overflow-hidden ${CARD_BASE}`}
      style={{
        fontFamily: 'Inter, system-ui, sans-serif',
        height: 'auto',
        boxShadow: `inset 0 0 24px rgba(0,0,0,0.25), 0 0 ${glowConfig.blur}px ${glowRgba}`,
      }}
      aria-label="US Market Sentiment gauge"
    >
      <div
        className="absolute inset-0 rounded-2xl pointer-events-none"
        style={{
          boxShadow: `inset 0 0 ${glowConfig.blur}px ${glowRgbaPulse}`,
          animation: 'sentiment-pulse 3s ease-in-out infinite',
        }}
      />
      <style>{`
        @keyframes sentiment-pulse {
          0%, 100% { opacity: 0.55; }
          50%       { opacity: 1; }
        }
      `}</style>

      <div className="w-full max-w-[280px]">
        {/*
          WHY transform-box:view-box + 50% 100% is correct:
          CSS "50px 50px" = screen pixels, NOT viewBox units. When the SVG scales
          to fill its container, 50px ≠ viewBox-50, so the pivot drifts.
          Fix: transform-box:view-box makes transform-origin use the SVG viewport.
          viewBox "0 0 100 50" → 50% of width(100) = 50, 100% of height(50) = 50.
          So "50% 100%" = viewBox point (50, 50) at any rendered size. ✓

          Needle math:
          · x1="50" y1="50" → starts at pivot (50, 50)
          · x2="50" y2="5"  → points straight UP before rotation
          · rotation = (sentimentScore / 100) * 180 − 90
            score  0 → −90° → LEFT  (Extreme Fear) ✓
            score 50 →   0° → UP    (Neutral)       ✓
            score 100 → +90° → RIGHT (Extreme Greed) ✓
            score 22  → −50.4° → far-left (Extreme Fear) ✓
        */}
        <svg viewBox="0 0 100 50" className="w-full block" preserveAspectRatio="xMidYMid meet" overflow="visible">
          <defs>
            <linearGradient id="sgEnamel" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#ef4444" stopOpacity={0.95} />
              <stop offset="50%"  stopColor="#eab308" stopOpacity={0.95} />
              <stop offset="100%" stopColor="#22c55e" stopOpacity={0.95} />
            </linearGradient>
          </defs>

          {/* Arc track and color band: semicircle center(50,50) r=50 */}
          <path d="M 0 50 A 50 50 0 0 1 100 50" fill="none" stroke="#1e293b"          strokeWidth="6"   strokeLinecap="round" />
          <path d="M 0 50 A 50 50 0 0 1 100 50" fill="none" stroke="url(#sgEnamel)" strokeWidth="4.5" strokeLinecap="round" opacity={0.92} />

          {/* LAYER 1 — NEEDLE (before pivot so hub covers the base joint)
              transform-box:view-box + origin "50% 100%" = pivot locked to (50,50) in viewBox coords */}
          <g
            style={{
              transformBox: 'view-box',
              transformOrigin: '50% 100%',
              transform: `rotate(${rotation}deg)`,
              transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          >
            <line x1="50" y1="50" x2="50" y2="5" stroke="#94a3b8" strokeWidth="2.2" strokeLinecap="butt" fill="none" />
            <circle cx="50" cy="5" r="2.5" fill={currentColor} style={{ filter: `drop-shadow(0 0 5px ${currentColor})` }} />
          </g>

          {/* LAYER 2 — PIVOT HUB on top of needle base; cx cy EXACTLY (50, 50) */}
          <circle cx="50" cy="50" r="5.5" fill="#0b0f1a" stroke="#334155" strokeWidth="1.5" />
          <circle cx="50" cy="50" r="2.2" fill="#64748b" />
        </svg>
      </div>

      <div className="w-full max-w-[320px] mt-2 px-1" aria-hidden="true">
        <div className="grid grid-cols-5 gap-x-1 text-[10px] uppercase text-gray-500/80 tracking-wider">
          {['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'].map((cat) => (
            <span key={cat} className="text-center min-w-0" title={cat}>
              {cat === 'Extreme Fear' ? 'Ext. Fear' : cat === 'Extreme Greed' ? 'Ext. Greed' : cat}
            </span>
          ))}
        </div>
      </div>

      <div
        className="mt-5 text-center rounded-2xl border border-gray-600/40 px-8 py-4 min-w-[160px] backdrop-blur-md bg-gray-900/80"
        style={{ boxShadow: `inset 0 1px 6px rgba(0,0,0,0.45), 0 0 26px ${scoreGlow}` }}
        aria-live="polite"
        aria-label={`Sentiment score ${Math.round(displayScore)}, ${currentLabel}`}
      >
        <p className="text-3xl font-bold text-white leading-none tracking-tight tabular-nums">
          {Math.round(displayScore)}
        </p>
        <p className="text-xs font-semibold mt-2 uppercase tracking-wider" style={{ color: currentColor }}>
          {currentLabel}
        </p>
      </div>

      <p className="mt-4 text-sm leading-relaxed text-center text-gray-400/90 max-w-[280px]">
        {insightText}
      </p>

      <div className="w-full max-w-[300px] mt-5 pt-5 border-t border-slate-700/50">
        <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2.5 font-semibold">Vs Current</p>
        <div className="rounded-lg border border-slate-700/50 overflow-hidden bg-gray-900/20 backdrop-blur-sm">
          <table className="w-full text-left border-collapse table-fixed" aria-label="Sentiment vs current">
            <colgroup>
              <col style={{ width: '50%' }} />
              <col style={{ width: '50%' }} />
            </colgroup>
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="py-2 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Period</th>
                <th className="py-2 px-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider text-right">Score</th>
              </tr>
            </thead>
            <tbody>
              {trendData.map((row) => {
                const val = Number(row.value);
                return (
                    <tr
                      key={row.periodKey}
                      className="border-b border-slate-700/30 last:border-0 cursor-pointer hover:bg-slate-800/50 transition-colors rounded"
                      onMouseEnter={() => setHoveredScore(row.score)}
                      onMouseLeave={() => setHoveredScore(null)}
                    >
                    <td className="py-2 px-3 text-xs text-gray-400">{row.period}</td>
                    <td className="py-2 px-3 text-sm font-semibold text-white text-right tabular-nums">{Math.round(val)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}


