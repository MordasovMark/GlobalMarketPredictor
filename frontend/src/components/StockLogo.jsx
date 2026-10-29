import { useState } from 'react';

function StockLogo({ ticker, size = 'sm' }) {
  const [failed, setFailed] = useState(false);
  const symbol = (ticker || '').replace('.', '-');
  const sizeClass = size === 'lg' ? 'w-14 h-14 text-xl' : size === 'md' ? 'w-10 h-10 text-sm' : 'w-[26px] h-[26px] text-[11px]';
  if (failed || !ticker) {
    return (
      <span className={`flex items-center justify-center rounded-full bg-slate-800 font-bold text-slate-300 shrink-0 select-none uppercase ${sizeClass}`}>
        {ticker ? ticker.replace('.', '').slice(0, 1) : '?'}
      </span>
    );
  }
  return (
    <img
      src={`https://assets.parqet.com/logos/symbol/${encodeURIComponent(symbol)}?format=png`}
      alt=""
      aria-hidden="true"
      loading="lazy"
      className={`rounded-full object-cover bg-slate-800 shrink-0 ${sizeClass}`}
      onError={() => setFailed(true)}
    />
  );
}

export default StockLogo;
