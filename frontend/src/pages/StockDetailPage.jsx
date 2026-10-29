import { useState } from 'react';
import { ArrowLeft, Info, Star } from 'lucide-react';
import { Bar, BarChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import AIRatingGauge from '../components/AIRatingGauge';
import ForecastChart from '../components/ForecastChart';
import LiveIndicator from '../components/LiveIndicator';
import MarketIntelligence from '../components/MarketIntelligence';
import ModelTrackRecord from '../components/ModelTrackRecord';
import PortfolioSimulator from '../components/PortfolioSimulator';
import StockLogo from '../components/StockLogo';
import { STOCK_ABOUT } from '../data/stocks';
import { api } from '../lib/api';
import { CARD_BASE, LIVE_POLL_MS, NEWS_POLL_MS, SLOW_POLL_MS, TOOLTIP_STYLE, fmtPct, fmtUsd, getSignal, isNum } from '../lib/format';
import { useMarketData } from '../lib/MarketDataContext';
import { usePolling } from '../lib/usePolling';

const CHART_TIME_RANGE_OPTIONS = [
  { value: '1d', label: '1D' },
  { value: '1mo', label: '1M' },
  { value: '3mo', label: '3M' },
  { value: '6mo', label: '6M' },
  { value: '1y', label: '1Y' },
];
const SMALL_MODEL_CHART_MARGIN = { top: 4, right: 4, left: 4, bottom: 0 };
const HEADER_PALETTE = {
  'Strong Buy': { bg: 'rgba(34,197,94,0.07)', border: 'rgba(34,197,94,0.28)', glow: 'rgba(34,197,94,0.10)' },
  Buy: { bg: 'rgba(16,185,129,0.07)', border: 'rgba(16,185,129,0.25)', glow: 'rgba(16,185,129,0.10)' },
  Hold: { bg: 'rgba(234,179,8,0.07)', border: 'rgba(234,179,8,0.28)', glow: 'rgba(234,179,8,0.09)' },
  Sell: { bg: 'rgba(239,68,68,0.07)', border: 'rgba(239,68,68,0.28)', glow: 'rgba(239,68,68,0.10)' },
  Pending: { bg: 'rgba(100,116,139,0.06)', border: 'rgba(100,116,139,0.25)', glow: 'rgba(100,116,139,0.06)' },
};
const MODEL_SIGNAL_CLS = {
  BUY: 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10',
  SELL: 'text-red-300 border-red-500/30 bg-red-500/10',
  HOLD: 'text-amber-300 border-amber-500/30 bg-amber-500/10',
};

function EmptyMini({ children }) {
  return <div className="h-full flex items-center justify-center text-[11px] text-gray-600">{children}</div>;
}

export default function StockDetailPage({ stock, onBack, isInWatchlist, onToggleWatchlist }) {
  const ticker = stock.ticker;
  const { stocks } = useMarketData();
  const live = stocks.find((s) => s.ticker === ticker) || stock;
  const [chartTimeRange, setChartTimeRange] = useState('3mo');
  const [showConsensusExplainer, setShowConsensusExplainer] = useState(false);

  const quote = usePolling((signal) => api.quotes([ticker], { signal }).then((r) => r[ticker] || null), LIVE_POLL_MS, [ticker]);
  const news = usePolling((signal) => api.news(ticker, 4, { signal }), NEWS_POLL_MS, [ticker]);
  const chart = usePolling((signal) => api.analyze(ticker, chartTimeRange, { signal }), LIVE_POLL_MS * 5, [ticker, chartTimeRange]);
  const rating = usePolling((signal) => api.rating(ticker, { signal }), SLOW_POLL_MS, [ticker]);

  const price = quote.data?.price ?? live.price;
  const change = quote.data?.change ?? live.change;
  const changePct = quote.data?.changePercent ?? live.changePercent;
  const isGain = (changePct ?? 0) >= 0;
  const color = isGain ? '#10b981' : '#ef4444';

  const r = rating.data;
  const sig = getSignal(r?.score);
  const SignalIcon = sig.Icon;
  const signalBadgeCls = `${sig.textCls} ${sig.bgCls} ${sig.borderCls}`;
  const hp = HEADER_PALETTE[sig.label] ?? HEADER_PALETTE.Hold;
  const about = live.about || STOCK_ABOUT[ticker] || `${stock.name} is a publicly traded company listed on US equity markets.`;

  const c = chart.data;
  const chartData = c?.chart_data || [];
  const stats = [
    { label: 'Market Cap', value: isNum(live.marketCap) ? `$${Number(live.marketCap).toLocaleString()}B` : '—' },
    { label: 'P/E Ratio', value: isNum(live.pe) ? `${Number(live.pe).toFixed(1)}×` : '—' },
    { label: '52W High', value: fmtUsd(live.high52) },
    { label: 'Div. Yield', value: isNum(live.dividendYield) ? `${Number(live.dividendYield).toFixed(2)}%` : '—' },
  ];
  const charts = r?.charts || {};

  return (
    <div key={`detail-${ticker}`} className="w-full min-h-full py-8 px-4 sm:px-10 box-border animate-fadeIn" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
      <button onClick={onBack} className="group inline-flex items-center gap-2 text-sm text-gray-500 hover:text-white transition-colors mb-8">
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
        Back to Dashboard
      </button>

      {/* Hero header — tint follows the consensus rating */}
      <div
        className="rounded-2xl border p-6 mb-6 flex flex-col sm:flex-row sm:items-center gap-5 transition-all duration-500"
        style={{ background: hp.bg, borderColor: hp.border, boxShadow: `0 0 48px ${hp.glow}, 0 0 96px ${hp.glow}` }}
      >
        <StockLogo ticker={ticker} size="lg" />
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-3 mb-1">
            <h1 className="text-4xl font-extrabold text-white tracking-tight">{ticker}</h1>
            <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${signalBadgeCls}`}>
              <SignalIcon className="w-3.5 h-3.5 shrink-0" />
              {rating.loading ? 'Rating…' : sig.label}
            </span>
            {onToggleWatchlist && (
              <button
                type="button"
                onClick={() => onToggleWatchlist(ticker)}
                className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border transition-colors ${
                  isInWatchlist
                    ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
                    : 'border-slate-600 bg-slate-800/40 text-gray-400 hover:text-white hover:border-slate-500'
                }`}
              >
                <Star className={`w-3.5 h-3.5 shrink-0 ${isInWatchlist ? 'fill-amber-400 text-amber-400' : ''}`} />
                {isInWatchlist ? 'Watching' : 'Add to watchlist'}
              </button>
            )}
          </div>
          <p className="text-gray-400 text-base">{stock.name}</p>
        </div>
        <div className="text-right shrink-0">
          <p className="text-4xl font-bold text-white tabular-nums">{fmtUsd(price)}</p>
          <p className={`text-xl font-semibold mt-1 tabular-nums ${isGain ? 'text-emerald-400' : 'text-red-400'}`}>
            {isNum(change) ? `${change >= 0 ? '+' : ''}${Number(change).toFixed(2)}` : '—'} ({fmtPct(changePct)})
          </p>
          <p className="text-xs text-gray-500 mt-1 uppercase tracking-wider">Today&apos;s Change</p>
        </div>
      </div>
      <div className="mt-[-1.25rem] mb-6 rounded-b-2xl border border-t-0 border-slate-800/60 px-6 py-3 bg-[#12131a]/45">
        <LiveIndicator lastSynced={quote.lastSynced} />
      </div>

      <MarketIntelligence ticker={ticker} items={news.data?.items ?? []} isLoading={news.loading} error={news.error || ''} />

      {/* Price + forecast cone */}
      <div className={`${CARD_BASE} p-6 mb-6`}>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
          <div>
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-[0.18em]">Price History &amp; AI Forecast</h3>
            {c?.forecast_available && (
              <p className="text-[11px] text-gray-600 mt-1">
                Dashed: median forecast. Shaded: 80% range (10th–90th percentile), next {chartTimeRange === '1d' ? 'close' : '10 sessions'}.
              </p>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 sm:justify-end">
            {CHART_TIME_RANGE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setChartTimeRange(opt.value)}
                className={`text-[10px] font-semibold uppercase tracking-wider px-2.5 py-1 rounded-md border transition-colors ${
                  chartTimeRange === opt.value
                    ? 'border-emerald-500/45 bg-emerald-500/10 text-emerald-300'
                    : 'border-slate-700/60 bg-slate-900/40 text-gray-500 hover:text-gray-300 hover:border-slate-600'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div className="h-[320px]">
          {chart.loading && chartData.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <span className="relative inline-flex h-5 w-5 rounded-full animate-ping" style={{ backgroundColor: color }} />
              <p className="text-sm font-medium tracking-widest uppercase animate-pulse" style={{ color }}>
                Fetching Market Data…
              </p>
            </div>
          )}
          {!chart.loading && chart.error && chartData.length === 0 && (
            <div className="flex items-center justify-center h-full text-sm text-amber-300/90">Chart unavailable: {chart.error}</div>
          )}
          {chartData.length > 0 && <ForecastChart data={chartData} ticker={ticker} color={color} gradId={`detailFill-${ticker}`} />}
        </div>

        {c?.forecast_summary && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
            <div className="rounded-lg border border-slate-800/70 bg-slate-900/35 px-4 py-3">
              <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">Model signal (5d)</p>
              <span className={`inline-flex text-xs font-bold px-2 py-0.5 rounded-md border ${MODEL_SIGNAL_CLS[c.signal] || ''}`}>{c.signal}</span>
              <p className="text-[10px] text-gray-600 mt-1">{isNum(c.prob_up_5d) ? `${Math.round(c.prob_up_5d * 100)}% chance of a gain` : ''}</p>
            </div>
            {[
              ['Bull case (90th)', c.forecast_summary.bull, 'text-emerald-400'],
              ['Base case (median)', c.forecast_summary.base, 'text-sky-300'],
              ['Bear case (10th)', c.forecast_summary.bear, 'text-red-400'],
            ].map(([label, v, cls]) => (
              <div key={label} className="rounded-lg border border-slate-800/70 bg-slate-900/35 px-4 py-3">
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">{label} · 10d</p>
                <p className={`text-lg font-bold tabular-nums ${cls}`}>{fmtUsd(v)}</p>
                <p className="text-[10px] text-gray-600">{isNum(c.price) ? fmtPct(((v / c.price) - 1) * 100) : ''}</p>
              </div>
            ))}
          </div>
        )}
        {c && !c.forecast_available && (
          <p className="text-[11px] text-amber-300/80 mt-3">The forecast model is still loading on the server; showing price history only.</p>
        )}
        <div className="mt-4 pt-3 border-t border-slate-800/60">
          <LiveIndicator lastSynced={chart.lastSynced} />
        </div>
      </div>

      <ModelTrackRecord model={c?.model} />
      <PortfolioSimulator ticker={ticker} timeRange={chartTimeRange} accentColor={color} />

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        {stats.map(({ label, value }) => (
          <div key={label} className={`${CARD_BASE} p-5`}>
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em] mb-2">{label}</p>
            <p className="text-xl font-bold text-white tabular-nums">{value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2">
          <AIRatingGauge models={r?.models ?? []} factors={r?.factors ?? []} loading={rating.loading} />
        </div>
        <div className={`${CARD_BASE} p-6 lg:col-span-3 flex flex-col`} style={{ borderLeftWidth: '3px', borderLeftColor: color }}>
          <h3 className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.18em] mb-4">About the Company</h3>
          <p className="text-sm text-gray-300 leading-relaxed flex-1 line-clamp-[12]">{about}</p>
          {r?.summary && <p className="text-sm text-slate-400 leading-relaxed mt-4 pt-4 border-t border-slate-800/60">{r.summary}</p>}
          <div className="mt-6 pt-4 border-t border-slate-800/60 flex flex-wrap items-center gap-3">
            <span className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider">AI Signal</span>
            <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${signalBadgeCls}`}>
              <SignalIcon className="w-3.5 h-3.5 shrink-0" />
              {sig.label}
            </span>
            <span className="text-[10px] text-gray-500 tabular-nums">
              AI Score: <span className="text-gray-300 font-semibold tabular-nums">{isNum(r?.score) ? Math.round(r.score) : '—'}</span>
            </span>
            {rating.error && <span className="text-[10px] text-amber-400/80">{rating.error}</span>}
          </div>
        </div>
      </div>

      {/* How the engines work — every mini-chart is this stock's live data */}
      <div className="w-full mt-8">
        <button
          type="button"
          onClick={() => setShowConsensusExplainer((v) => !v)}
          className="w-full py-4 px-5 rounded-xl border border-slate-700/60 flex items-center justify-center gap-2 text-sm font-semibold text-slate-300 hover:text-white hover:border-emerald-500/40 transition-all duration-300"
          style={{
            background: 'rgba(15, 23, 42, 0.6)',
            boxShadow: showConsensusExplainer ? 'none' : '0 0 32px rgba(16, 185, 129, 0.12), 0 0 64px rgba(16, 185, 129, 0.06)',
          }}
        >
          <Info className="w-4 h-4 text-emerald-400/90 shrink-0" />
          Learn how the consensus is calculated
        </button>

        <div
          className="overflow-hidden transition-all duration-500 ease-out"
          style={{ maxHeight: showConsensusExplainer ? '2400px' : '0', opacity: showConsensusExplainer ? 1 : 0 }}
        >
          <div className="mt-4 p-6 rounded-xl border border-slate-800/70" style={{ background: 'rgba(15, 23, 42, 0.5)', backdropFilter: 'blur(12px)' }}>
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-[0.2em] mb-6">Behind the Scenes: How Our AI Models Work</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
              <div className="rounded-lg border border-slate-800/60 p-5 bg-slate-900/30">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl" aria-hidden>⚖️</span>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider">1. DeepValue Engine</h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Ranks {ticker} against the 20 tracked peers on earnings, free-cash-flow and book yield (value), ROE, margins and debt (quality), and revenue and EPS growth.
                </p>
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Factor percentile vs peers (50 = median)</p>
                <div className="h-24">
                  {charts.value_factors?.length ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={charts.value_factors} margin={SMALL_MODEL_CHART_MARGIN}>
                        <XAxis dataKey="factor" tick={{ fontSize: 9, fill: '#64748b' }} />
                        <YAxis hide domain={[0, 100]} />
                        <ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="3 3" />
                        <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: '#e2e8f0' }} />
                        <Bar dataKey="percentile" fill="#10b981" fillOpacity={0.8} radius={[2, 2, 0, 0]} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <EmptyMini>Fundamentals unavailable</EmptyMini>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-slate-800/60 p-5 bg-slate-900/30">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl" aria-hidden>🧠</span>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider">2. Sentiment NLP</h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Scores every headline from the last 7 days ({r?.news?.backend === 'finbert' ? 'FinBERT' : 'finance lexicon'}). Recent stories count more; few headlines pull the score toward neutral.
                </p>
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Headlines per day: positive vs negative</p>
                <div className="h-24">
                  {charts.sentiment_flow?.some((d) => d.pos || d.neg) ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={charts.sentiment_flow} margin={SMALL_MODEL_CHART_MARGIN}>
                        <XAxis dataKey="w" tick={{ fontSize: 9, fill: '#64748b' }} />
                        <YAxis hide allowDecimals={false} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: '#e2e8f0' }} />
                        <Bar dataKey="pos" name="Positive" fill="#10b981" fillOpacity={0.8} radius={[2, 2, 0, 0]} isAnimationActive={false} />
                        <Bar dataKey="neg" name="Negative" fill="#ef4444" fillOpacity={0.6} radius={[2, 2, 0, 0]} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <EmptyMini>No scored headlines this week</EmptyMini>
                  )}
                </div>
              </div>

              <div className="rounded-lg border border-slate-800/60 p-5 bg-slate-900/30">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl" aria-hidden>📈</span>
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider">3. Tech-Momentum Net</h4>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Trend and relative strength vs the S&amp;P 500, blended with the gradient-boosting forecaster. The ML weight follows the model&apos;s measured out-of-sample skill.
                </p>
                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">MACD vs signal, ADX (last 20 sessions)</p>
                <div className="h-24">
                  {charts.momentum?.length ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={charts.momentum} margin={SMALL_MODEL_CHART_MARGIN}>
                        <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#64748b' }} minTickGap={16} />
                        <YAxis yAxisId="m" hide domain={['auto', 'auto']} />
                        <YAxis yAxisId="a" hide domain={[0, 60]} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: '#e2e8f0' }} />
                        <Line yAxisId="m" type="monotone" dataKey="macd" name="MACD" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line yAxisId="m" type="monotone" dataKey="signal" name="Signal" stroke="#94a3b8" strokeWidth={1} dot={false} isAnimationActive={false} />
                        <Line yAxisId="a" type="monotone" dataKey="adx" name="ADX" stroke="#eab308" strokeWidth={1.5} strokeDasharray="4 2" dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <EmptyMini>Indicators unavailable</EmptyMini>
                  )}
                </div>
              </div>
            </div>

            <div className="pt-5 border-t border-slate-800/60">
              <p className="text-sm text-slate-400 leading-relaxed">
                <span className="font-semibold text-slate-300">The Consensus:</span> A weighted blend (value 35%, sentiment 25%, momentum 40%) of the engines that have data. A Buy needs at least 2 of 3 engines bullish (score ≥ 60) and a Sell at least 2 bearish (≤ 40); otherwise the rating is held in the Hold band.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
