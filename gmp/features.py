"""Feature engineering shared by training and live inference.

Design rules (the old pipeline broke all of them):

1. **Point-in-time.** A row dated *t* only uses data known at the close of *t*.
   ``tests/test_features.py`` checks this by changing future prices and
   confirming past features don't move.
2. **Scale-free.** Raw prices and raw SMA levels can't be pooled across tickers
   (a $900 stock and a $40 stock look like different worlds). Every feature is a
   return, a ratio, a z-score or a rank.
3. **Same code path** for training and serving, so there is no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS: tuple[int, ...] = (1, 5, 10)

FEATURES: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "ret_5d_z",
    "vol_21d",
    "vol_ratio",
    "rsi_14",
    "macd_hist_norm",
    "adx_14",
    "dist_sma50",
    "dist_sma200",
    "range_pos_252",
    "volume_z_21",
    "beta_63",
    "rel_ret_21d",
    "rel_ret_63d",
    "mkt_ret_5d",
    "mkt_ret_21d",
    "mkt_vol_21d",
    "xs_rank_ret_21d",
    "xs_rank_ret_63d",
    "xs_rank_vol_21d",
]

# Minimum daily-vol floor so vol-scaled targets don't explode on quiet names.
MIN_DAILY_VOL = 0.004


# ---------------------------------------------------------------------------
# Classic indicators (also used for display)
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses at all in the window -> RSI 100; no movement -> 50.
    out = out.where(~(avg_loss == 0), 100.0)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (trend strength, 0-100)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Per-ticker features
# ---------------------------------------------------------------------------
def ticker_features(df: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    """Features for one ticker. ``df`` is clean daily OHLCV; ``bench_close`` the market."""
    close = df["Close"].astype(float)
    logp = np.log(close)
    r1 = logp.diff()
    bench = bench_close.reindex(close.index).ffill()
    blogp = np.log(bench)
    br1 = blogp.diff()

    out = pd.DataFrame(index=close.index)
    out["ret_1d"] = r1
    for n in (5, 21, 63, 126):
        out[f"ret_{n}d"] = logp - logp.shift(n)

    vol_21 = r1.rolling(21, min_periods=15).std()
    vol_63 = r1.rolling(63, min_periods=40).std()
    out["vol_21d"] = vol_21 * np.sqrt(TRADING_DAYS)
    out["vol_ratio"] = vol_21 / vol_63
    out["ret_5d_z"] = out["ret_5d"] / (vol_21.clip(lower=MIN_DAILY_VOL) * np.sqrt(5))

    out["rsi_14"] = rsi(close, 14)
    out["macd_hist_norm"] = macd(close)["hist"] / close
    out["adx_14"] = adx(df, 14)

    out["dist_sma50"] = close / close.rolling(50, min_periods=40).mean() - 1.0
    out["dist_sma200"] = close / close.rolling(200, min_periods=150).mean() - 1.0
    hi = close.rolling(252, min_periods=126).max()
    lo = close.rolling(252, min_periods=126).min()
    out["range_pos_252"] = (close - lo) / (hi - lo).replace(0.0, np.nan)

    logv = np.log(df["Volume"].astype(float).replace(0.0, np.nan))
    out["volume_z_21"] = (logv - logv.rolling(21, min_periods=15).mean()) / logv.rolling(21, min_periods=15).std()

    cov = r1.rolling(63, min_periods=40).cov(br1)
    var = br1.rolling(63, min_periods=40).var()
    out["beta_63"] = cov / var
    out["mkt_ret_5d"] = blogp - blogp.shift(5)
    out["mkt_ret_21d"] = blogp - blogp.shift(21)
    out["mkt_vol_21d"] = br1.rolling(21, min_periods=15).std() * np.sqrt(TRADING_DAYS)
    out["rel_ret_21d"] = out["ret_21d"] - out["mkt_ret_21d"]
    out["rel_ret_63d"] = out["ret_63d"] - (blogp - blogp.shift(63))

    # Kept for target scaling; not a model input by itself.
    out["daily_vol"] = vol_21.clip(lower=MIN_DAILY_VOL)
    out["close"] = close
    return out.replace([np.inf, -np.inf], np.nan)


def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks across tickers on each date (needs a (date, ticker) index)."""
    grouped = panel.groupby(level="date")
    panel["xs_rank_ret_21d"] = grouped["ret_21d"].rank(pct=True)
    panel["xs_rank_ret_63d"] = grouped["ret_63d"].rank(pct=True)
    panel["xs_rank_vol_21d"] = grouped["vol_21d"].rank(pct=True)
    return panel


def add_targets(panel: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Forward log returns and vol-scaled versions. Uses future data on purpose."""
    logp = np.log(panel["close"])
    by_ticker = logp.groupby(level="ticker")
    for h in horizons:
        fwd = by_ticker.shift(-h) - logp
        panel[f"fwd_ret_{h}d"] = fwd
        panel[f"target_z_{h}d"] = fwd / (panel["daily_vol"] * np.sqrt(h))
    return panel


def build_panel(
    histories: dict[str, pd.DataFrame],
    bench_close: pd.Series,
    with_targets: bool = True,
) -> pd.DataFrame:
    """Stack per-ticker features into a (date, ticker) panel."""
    frames = []
    for ticker, df in histories.items():
        if df is None or len(df) < 60:
            continue
        feats = ticker_features(df, bench_close)
        feats["ticker"] = ticker
        frames.append(feats)
    if not frames:
        return pd.DataFrame(columns=FEATURES)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("ticker", append=True).sort_index()
    panel = add_cross_sectional(panel)
    if with_targets:
        panel = add_targets(panel)
    return panel


def indicator_series(df: pd.DataFrame, last_n: int = 20) -> list[dict]:
    """MACD + ADX for the last ``last_n`` sessions (for the momentum mini-chart)."""
    m = macd(df["Close"])
    a = adx(df)
    frame = pd.DataFrame({"macd": m["macd"], "signal": m["signal"], "adx": a}).dropna().tail(last_n)
    return [
        {
            "t": idx.strftime("%m-%d"),
            "macd": round(float(row.macd), 3),
            "signal": round(float(row.signal), 3),
            "adx": round(float(row.adx), 1),
        }
        for idx, row in frame.iterrows()
    ]
