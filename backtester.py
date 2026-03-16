"""
Backtester: Simulate the trained AI model's trading performance vs SPY benchmark.
Signal for today is based on YESTERDAY's features (no look-ahead). Strategy_Return = Signal.shift(1) * Daily_Return
so we decide today but only get tomorrow's profit. Saves timeline to data/backtest_results.csv.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

# Align with train_model constants
MODEL_PATH = Path("models") / "trained_ai_model.pkl"
RESULTS_PATH = Path("data") / "backtest_results.csv"
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
}
MACRO_TICKERS = {"vix": "^VIX", "tnx": "^TNX"}
TEST_YEARS = 2
LOOKBACK_DAYS = 60  # extra history for RSI/SMA/MACD warmup
RSI_PERIOD = 14
SMA_PERIOD = 20
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
SECTOR_MOMENTUM_DAYS = 5
PROBA_THRESHOLD = 0.55


def _download_series(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download daily Close for a ticker; return Series with DatetimeIndex."""
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
    if data is None or data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0).str.strip()
    close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
    return close.astype(float).sort_index()


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _macd_histogram(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line).fillna(0.0)


def _get_sector(ticker: str) -> str | None:
    try:
        sector = yf.Ticker(ticker).info.get("sector")
        if not sector or not isinstance(sector, str):
            return None
        sector = sector.strip()
        return sector if sector in SECTOR_ETFS else None
    except Exception:
        return None


def _build_feature_row(
    date: pd.Timestamp,
    close: pd.Series,
    macro: pd.DataFrame,
    sector_etf_close: pd.Series,
    sector_name: str,
    feature_names: list[str],
) -> pd.Series | None:
    """Build a single row of features in the order expected by the model."""
    if date not in close.index or date not in macro.index:
        return None
    dates = close.index
    idx = dates.get_loc(date)
    min_lookback = max(SMA_PERIOD, RSI_PERIOD, MACD_SLOW + MACD_SIGNAL)
    if idx < min_lookback:
        return None
    close_slice = close.iloc[: idx + 1]
    rsi = _rsi(close_slice, RSI_PERIOD).iloc[-1]
    sma_20 = close_slice.rolling(window=SMA_PERIOD, min_periods=1).mean().iloc[-1]
    macd = _macd_histogram(close_slice, MACD_FAST, MACD_SLOW, MACD_SIGNAL).iloc[-1]
    vix = macro.loc[date, "vix"]
    tnx = macro.loc[date, "tnx"]
    # Sector momentum: 5-day pct change of sector ETF aligned to this date
    etf_aligned = sector_etf_close.reindex(dates).ffill().bfill()
    if date in etf_aligned.index and idx >= SECTOR_MOMENTUM_DAYS:
        pct = etf_aligned.pct_change(SECTOR_MOMENTUM_DAYS).loc[date]
        sector_momentum = pct if pd.notna(pct) else 0.0
    else:
        sector_momentum = 0.0
    row = {
        "rsi": rsi,
        "sma_20": sma_20,
        "macd": macd,
        "vix": vix,
        "tnx": tnx,
        "Sector_Momentum": sector_momentum,
    }
    for name in feature_names:
        if name.startswith("Sector_"):
            sector_col = name.replace("Sector_", "")
            row[name] = 1 if sector_col == sector_name else 0
    return pd.Series({k: row[k] for k in feature_names if k in row})


def run_backtest(
    test_stock: str = "AAPL",
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    """Load model, download data, reconstruct features, run signals, compute returns and metrics."""
    model = joblib.load(MODEL_PATH)
    feature_names = list(model.feature_names_in_)

    end = pd.Timestamp.now().normalize()
    test_start = end - pd.Timedelta(days=TEST_YEARS * 365)
    lookback_start = test_start - pd.Timedelta(days=LOOKBACK_DAYS)

    print(f"Loading data ({lookback_start.date()} to {end.date()})...")
    # Benchmark and stock
    spy_close = _download_series(benchmark_ticker, lookback_start, end)
    stock_close = _download_series(test_stock, lookback_start, end)
    if spy_close.empty or stock_close.empty:
        raise RuntimeError(f"Could not download {benchmark_ticker} or {test_stock}.")

    # Macro
    macro_series = {}
    for name, ticker in MACRO_TICKERS.items():
        s = _download_series(ticker, lookback_start, end)
        if not s.empty:
            macro_series[name] = s
    if len(macro_series) != 2:
        raise RuntimeError("Could not download VIX and TNX.")
    macro = pd.DataFrame(macro_series).dropna(how="any")

    # Sector and sector ETF
    sector_name = _get_sector(test_stock)
    if sector_name not in SECTOR_ETFS:
        sector_name = "Unknown"
        sector_etf_close = pd.Series(index=stock_close.index, data=0.0)
    else:
        sector_etf_close = _download_series(SECTOR_ETFS[sector_name], lookback_start, end)
        if sector_etf_close.empty:
            sector_etf_close = pd.Series(index=stock_close.index, data=0.0)

    # Common dates (test period only): intersection of stock, SPY, macro
    common = stock_close.index.union(spy_close.index).union(macro.index)
    common = common[common >= test_start].sort_values()
    common = common.intersection(stock_close.index).intersection(spy_close.index).intersection(macro.index)

    rows = []
    for date in common:
        ser = _build_feature_row(
            date, stock_close, macro, sector_etf_close, sector_name or "Unknown", feature_names
        )
        if ser is None:
            continue
        row = ser.to_dict()
        row["date"] = date
        row["stock_close"] = stock_close.loc[date]
        row["spy_close"] = spy_close.reindex([date]).ffill().bfill().iloc[0]
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid feature rows in test period.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Calculate features normally (no shifting). Signal[T] = model(features T) = prediction for T+1.
    X = df[feature_names]
    df["Signal"] = model.predict(X)

    # Daily returns from close-to-close
    df["Daily_Return"] = df["stock_close"].pct_change().fillna(0.0)
    df["spy_return"] = df["spy_close"].pct_change().fillna(0.0)

    # THE FINAL FIX: Signal from yesterday applies to today's return — decide today, profit tomorrow
    df["Strategy_Return"] = df["Signal"].shift(1).fillna(0).astype(int) * df["Daily_Return"]

    print("DEBUG: Checking signal alignment:\n", df[["Daily_Return", "Signal", "Strategy_Return"]].head(10))

    # Recalculate Cumulative and Annualized from scratch
    df["spy_cumulative"] = (1 + df["spy_return"]).cumprod()
    df["ai_cumulative"] = (1 + df["Strategy_Return"]).cumprod()

    return df


def _annualized_return(cumulative: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return (cumulative ** (252.0 / n_days)) - 1.0


def _max_drawdown(cumulative_series: pd.Series) -> float:
    peak = cumulative_series.expanding().max()
    dd = (peak - cumulative_series) / peak
    return dd.max()


def print_report(df: pd.DataFrame, test_stock: str = "AAPL") -> None:
    """Print formatted comparison of AI Strategy vs SPY."""
    n = len(df)
    spy_final = df["spy_cumulative"].iloc[-1]
    ai_final = df["ai_cumulative"].iloc[-1]
    spy_cum = spy_final - 1.0
    ai_cum = ai_final - 1.0
    spy_ann = _annualized_return(spy_final, n)
    ai_ann = _annualized_return(ai_final, n)
    spy_dd = _max_drawdown(df["spy_cumulative"])
    ai_dd = _max_drawdown(df["ai_cumulative"])

    print()
    print("=" * 60)
    print("  BACKTEST REPORT: AI Strategy vs SPY Benchmark")
    print("=" * 60)
    print(f"  Test stock: {test_stock}  |  Benchmark: SPY  |  Period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Trading days: {n}  |  Strategy_Return = Signal.shift(1) * Daily_Return")
    print("-" * 60)
    print(f"  {'Metric':<28}  {'AI Strategy':>14}  {'SPY Buy & Hold':>14}")
    print("-" * 60)
    print(f"  {'Cumulative Return':<28}  {ai_cum:>13.2%}  {spy_cum:>13.2%}")
    print(f"  {'Annualized Return':<28}  {ai_ann:>13.2%}  {spy_ann:>13.2%}")
    print(f"  {'Max Drawdown':<28}  {ai_dd:>13.2%}  {spy_dd:>13.2%}")
    print("-" * 60)
    print("=" * 60)
    print()


def main() -> None:
    test_stock = "AAPL"
    benchmark_ticker = "SPY"

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    df = run_backtest(test_stock=test_stock, benchmark_ticker=benchmark_ticker)
    print_report(df, test_stock=test_stock)

    out = df[
        [
            "date",
            "spy_return",
            "spy_cumulative",
            "Strategy_Return",
            "ai_cumulative",
            "Signal",
            "Daily_Return",
        ]
    ].copy()
    out.columns = ["date", "spy_return", "spy_cumulative", "ai_daily_return", "ai_cumulative", "signal", "stock_return"]
    out.to_csv(RESULTS_PATH, index=False)
    print(f"Cumulative timeline saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
