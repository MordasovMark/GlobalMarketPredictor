import { CheckCircle, AlertTriangle, Minus } from 'lucide-react';

export const CARD_BASE = 'bg-[#12131a] border border-slate-800/80 rounded-xl';
export const LIVE_POLL_MS = 60 * 1000;
export const SLOW_POLL_MS = 5 * 60 * 1000;
export const NEWS_POLL_MS = 15 * 60 * 1000;

export const isNum = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));

export function fmtUsd(n, digits = 2) {
  if (!isNum(n)) return '—';
  return Number(n).toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPct(n, digits = 2, { sign = true } = {}) {
  if (!isNum(n)) return '—';
  const v = Number(n);
  return `${sign && v > 0 ? '+' : ''}${v.toFixed(digits)}%`;
}

export function fmtNum(n, digits = 2) {
  return isNum(n) ? Number(n).toFixed(digits) : '—';
}

export const gainCls = (n) => (!isNum(n) ? 'text-gray-400' : Number(n) >= 0 ? 'text-emerald-400' : 'text-red-400');

export function formatSyncedTime(ts) {
  if (!ts) return '—';
  return new Date(ts).toLocaleTimeString('en-GB');
}

export function sampleChartData(data, maxPoints) {
  if (!Array.isArray(data)) return [];
  const len = data.length;
  if (len <= maxPoints || maxPoints < 2) return data;
  const step = (len - 1) / (maxPoints - 1);
  return Array.from({ length: maxPoints }, (_, i) => data[Math.round(i * step)]);
}

/**
 * SINGLE SOURCE OF TRUTH for rating bands — must match gmp/signals.py.
 * 0-40 Sell | 41-65 Hold | 66-84 Buy | 85-100 Strong Buy
 */
export function getSignal(score) {
  if (!isNum(score)) {
    return { label: 'Pending', hex: '#64748b', textCls: 'text-slate-400', bgCls: 'bg-slate-500/10', borderCls: 'border-slate-500/25', Icon: Minus };
  }
  const s = Math.min(100, Math.max(0, Number(score)));
  if (s >= 85) return { label: 'Strong Buy', hex: '#22c55e', textCls: 'text-green-400', bgCls: 'bg-green-400/10', borderCls: 'border-green-400/30', Icon: CheckCircle };
  if (s >= 66) return { label: 'Buy', hex: '#10b981', textCls: 'text-emerald-400', bgCls: 'bg-emerald-400/10', borderCls: 'border-emerald-400/25', Icon: CheckCircle };
  if (s >= 41) return { label: 'Hold', hex: '#eab308', textCls: 'text-amber-400', bgCls: 'bg-amber-400/10', borderCls: 'border-amber-400/25', Icon: Minus };
  return { label: 'Sell', hex: '#ef4444', textCls: 'text-red-400', bgCls: 'bg-red-400/10', borderCls: 'border-red-400/25', Icon: AlertTriangle };
}

export const TOOLTIP_STYLE = {
  backgroundColor: 'rgba(15, 23, 42, 0.92)',
  border: '1px solid rgba(71, 85, 105, 0.5)',
  borderRadius: '10px',
  padding: '8px 12px',
};
