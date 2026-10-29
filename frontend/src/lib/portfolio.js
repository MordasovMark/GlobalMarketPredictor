import { getSignal } from './format';

/** Re-mark the model's open positions with live quotes. */
export function markHoldings(holdings = [], quotes = {}, ratings = {}) {
  return holdings.map((h) => {
    const price = quotes[h.ticker]?.price ?? h.price;
    const marketValue = h.shares * price;
    const costBasis = h.shares * h.avgCost;
    const pl = marketValue - costBasis;
    const rating = ratings[h.ticker];
    return {
      ...h,
      price,
      marketValue,
      costBasis,
      pl,
      plPct: costBasis > 0 ? (pl / costBasis) * 100 : 0,
      aiScore: rating?.score ?? null,
      signal: getSignal(rating?.score),
      aiSummary: rating?.summary ?? '',
    };
  });
}

export function sumHoldings(rows) {
  const costBasis = rows.reduce((s, r) => s + r.costBasis, 0);
  const marketValue = rows.reduce((s, r) => s + r.marketValue, 0);
  const pl = marketValue - costBasis;
  return { costBasis, marketValue, pl, plPct: costBasis > 0 ? (pl / costBasis) * 100 : 0 };
}

