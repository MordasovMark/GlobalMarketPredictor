import { useMemo } from 'react';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { sampleChartData } from '../lib/format';

const MAX_POINTS = 260;

function ChartTooltip({ active, payload, label, ticker, color }) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload || {};
  return (
    <div
      style={{
        backgroundColor: 'rgba(12,15,24,0.95)',
        border: '1px solid rgba(71,85,105,0.5)',
        borderRadius: 10,
        padding: '8px 14px',
      }}
    >
      <p style={{ color: '#9ca3af', fontSize: 11, marginBottom: 4 }}>{label}</p>
      {row.price != null && (
        <p style={{ color, fontSize: 13, fontWeight: 600 }}>
          {ticker}: ${Number(row.price).toFixed(2)}
        </p>
      )}
      {row.price == null && row.predictedPrice != null && (
        <>
          <p style={{ color: '#93c5fd', fontSize: 13, fontWeight: 600 }}>Median: ${row.predictedPrice.toFixed(2)}</p>
          <p style={{ color: '#9ca3af', fontSize: 11 }}>
            80% range: ${row.predictedLow.toFixed(2)} – ${row.predictedHigh.toFixed(2)}
          </p>
        </>
      )}
    </div>
  );
}

/**
 * Price history + model forecast cone.
 * The shaded band is the model's 10th–90th percentile: 8 in 10 outcomes
 * should land inside it (the model card reports how often they actually do).
 */
export default function ForecastChart({ data = [], ticker, color, gradId }) {
  const chartData = useMemo(() => {
    const hist = data.filter((d) => d.price != null);
    const future = data.filter((d) => d.price == null);
    const sampled = sampleChartData(hist, MAX_POINTS - future.length);
    // Make sure the stitch point (last history bar, which carries predictedPrice) survives sampling.
    if (hist.length && sampled[sampled.length - 1] !== hist[hist.length - 1]) sampled.push(hist[hist.length - 1]);
    return [...sampled, ...future].map((d) => ({
      ...d,
      band: d.predictedLow != null && d.predictedHigh != null ? [d.predictedLow, d.predictedHigh] : null,
    }));
  }, [data]);
  const lastHist = useMemo(() => [...chartData].reverse().find((d) => d.price != null), [chartData]);
  const hasForecast = chartData.some((d) => d.price == null);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 4 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.45} />
            <stop offset="65%" stopColor={color} stopOpacity={0.12} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
        <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 11 }} stroke="rgba(255,255,255,0.04)" minTickGap={28} />
        <YAxis
          domain={['auto', 'auto']}
          tick={{ fill: '#6b7280', fontSize: 11 }}
          stroke="rgba(255,255,255,0.04)"
          width={64}
          tickFormatter={(v) => `$${v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)}`}
        />
        <Tooltip content={<ChartTooltip ticker={ticker} color={color} />} />
        {hasForecast && lastHist && (
          <ReferenceLine x={lastHist.date} stroke="rgba(148,163,184,0.35)" strokeDasharray="2 4" label={{ value: 'Today', fill: '#64748b', fontSize: 10, position: 'insideTopRight' }} />
        )}
        <Area type="linear" dataKey="price" stroke={color} strokeWidth={2} fill={`url(#${gradId})`} dot={false} connectNulls={false} isAnimationActive={false} />
        <Area type="monotone" dataKey="band" stroke="none" fill="#3b82f6" fillOpacity={0.16} connectNulls isAnimationActive={false} />
        <Line type="monotone" dataKey="predictedPrice" stroke="#93c5fd" strokeWidth={2} strokeDasharray="5 4" dot={false} connectNulls isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
