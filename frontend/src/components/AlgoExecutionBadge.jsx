import { CheckCircle, AlertTriangle } from 'lucide-react';

function AlgoExecutionBadge({ action }) {
  const isBuy = String(action || '').toUpperCase() === 'BUY';
  const Icon = isBuy ? CheckCircle : AlertTriangle;
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wide border ${
        isBuy
          ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25'
          : 'text-red-400 bg-red-400/10 border-red-400/25'
      }`}
    >
      <Icon className="w-3.5 h-3.5 shrink-0" />
      {isBuy ? 'BUY' : 'SELL'}
    </span>
  );
}

export default AlgoExecutionBadge;
