"""Business logic behind the API, plus the background dashboard snapshot.

The dashboard needs ratings for 20 stocks (fundamentals + news + model) and a
model-portfolio backtest. Computing that per request would take tens of
seconds, so a background thread rebuilds a snapshot every 15 minutes and the
endpoints serve the latest one.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .backtest import model_portfolio, simulate_single
from .forecast import cone, next_sessions
from .live import finite_or_none, latest_forecast, prediction_history, skill_confidence
from .market_data import DataUnavailable, get_fundamentals, get_histories, get_history, get_intraday, get_quotes
from .model import registry
from .ratings import compute_ratings
from .settings import settings
from .signals import trade_signal
from .universe import DASHBOARD, MACRO_SYMBOLS, to_display, to_yahoo

log = logging.getLogger(__name__)

RANGES = {"1d", "1mo", "3mo", "6mo", "1y"}
RANGE_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366}
REFRESH_SECONDS = 15 * 60


def normalize_range(time_range: str | None) -> str:
    r = (time_range or "3mo").strip().lower()
    return r if r in RANGES else "3mo"


def _window(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    days = RANGE_DAYS.get(time_range, 92)
    return df[df.index >= df.index[-1] - pd.Timedelta(days=days)]


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
def model_summary() -> dict:
    model = registry.get()
    if model is None:
        return {"status": registry.status, "error": registry.error}
    meta = model.meta
    m5 = (meta.get("metrics") or {}).get("5d", {})
    return {
        "status": "ready",
        "trained_at": meta.get("trained_at"),
        "data_source": meta.get("data_source"),
        "data_end": meta.get("data_end"),
        "n_tickers": meta.get("n_tickers"),
        "skill_confidence": round(skill_confidence(), 3),
        "oos_5d": {
            k: m5.get(k)
            for k in ("hit_rate", "always_up_hit_rate", "interval_coverage_80", "rank_ic_mean", "rank_ic_tstat")
        },
    }


def model_card() -> dict:
    model = registry.get()
    if model is None:
        raise DataUnavailable(f"Model not available ({registry.status})")
    return model.meta


# ---------------------------------------------------------------------------
# Quotes & universe
# ---------------------------------------------------------------------------
def quotes(tickers: list[str]) -> dict[str, dict]:
    raw = get_quotes(tickers)
    return {
        to_display(sym): {
            "price": round(q["price"], 4),
            "change": round(q["change"], 4),
            "changePercent": round(q["change_pct"], 4),
            "source": q.get("source"),
        }
        for sym, q in raw.items()
    }


def macro(symbols: list[str] | None = None) -> list[dict]:
    syms = symbols or list(MACRO_SYMBOLS)
    q = get_quotes(syms)
    out = []
    for s in syms:
        item = q.get(to_yahoo(s))
        out.append(
            {
                "symbol": s,
                "price": round(item["price"], 2) if item else None,
                "change_pct": round(item["change_pct"], 2) if item else None,
            }
        )
    return out


def universe_rows() -> list[dict]:
    syms = [to_yahoo(s.ticker) for s in DASHBOARD]
    hist = get_histories(syms, "1y")
    live = get_quotes(syms)
    with ThreadPoolExecutor(max_workers=8) as pool:
        funds = dict(zip(syms, pool.map(lambda s: get_fundamentals(s) or {}, syms)))
    rows = []
    for stock, sym in zip(DASHBOARD, syms):
        h = hist.get(sym)
        q = live.get(sym)
        f = funds.get(sym, {})
        last_year = h if h is not None else pd.DataFrame()
        mcap = f.get("market_cap")
        rows.append(
            {
                "ticker": stock.ticker,
                "name": stock.name,
                "sector": stock.sector,
                "domain": stock.domain,
                "price": finite_or_none(q["price"] if q else None, 2),
                "change": finite_or_none(q["change"] if q else None, 2),
                "changePercent": finite_or_none(q["change_pct"] if q else None, 2),
                "high52": finite_or_none(last_year["High"].max() if len(last_year) else None, 2),
                "low52": finite_or_none(last_year["Low"].min() if len(last_year) else None, 2),
                "avgVolume": finite_or_none(last_year["Volume"].tail(63).mean() if len(last_year) else None, 0),
                "pe": finite_or_none(f.get("trailing_pe"), 1),
                "dividendYield": finite_or_none(
                    f["dividend_yield"] * 100 if f.get("dividend_yield") is not None else None, 2
                ),
                "marketCap": finite_or_none(mcap / 1e9 if mcap else None, 0),
                "about": f.get("summary"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Chart + forecast
# ---------------------------------------------------------------------------
def analyze(ticker: str, time_range: str) -> dict:
    sym = to_yahoo(ticker)
    time_range = normalize_range(time_range)
    intraday = time_range == "1d"
    daily = get_history(sym, "2y")
    if intraday:
        try:
            hist = get_intraday(sym)
        except DataUnavailable:
            hist, intraday = _window(daily, "1mo"), False
    else:
        hist = _window(daily, time_range)
    if len(hist) < 2:
        raise DataUnavailable(f"No price data for {ticker}")

    fmt = "%H:%M" if intraday else "%Y-%m-%d"
    chart = [{"date": d.strftime(fmt), "price": round(float(p), 2)} for d, p in hist["Close"].items()]
    current = float(hist["Close"].iloc[-1])
    # Day change is always vs the previous session's close.
    prior = daily.loc[daily.index.normalize() < hist.index[-1].normalize(), "Close"]
    prev = float(prior.iloc[-1]) if len(prior) else float(hist["Close"].iloc[0])
    change_pct = (current / prev - 1) * 100 if prev else 0.0

    fc = latest_forecast(sym)
    payload: dict = {
        "ticker": to_display(sym),
        "price": round(current, 2),
        "change_pct": round(change_pct, 2),
        "time_range": time_range,
        "chart_data": chart,
        "forecast_available": fc is not None,
        "is_simulated": False,
        "model": model_summary(),
    }
    if fc is None:
        payload.update({"signal": None, "forecast_summary": None})
        return payload

    chart[-1].update(
        {"predictedPrice": chart[-1]["price"], "predictedLow": chart[-1]["price"], "predictedHigh": chart[-1]["price"]}
    )
    path = cone(fc, days=1 if intraday else 10)
    sessions = next_sessions(daily.index[-1], len(path))
    for step, day in zip(path, sessions):
        chart.append(
            {
                "date": "Next close" if intraday else day.strftime(fmt),
                "price": None,
                "predictedPrice": round(current * float(np.exp(step["q50"])), 2),
                "predictedLow": round(current * float(np.exp(step["q10"])), 2),
                "predictedHigh": round(current * float(np.exp(step["q90"])), 2),
            }
        )
    h5, h10 = fc["horizons"]["5d"], fc["horizons"]["10d"]
    payload.update(
        {
            "signal": trade_signal(h5["q50"], h5["prob_up"]),
            "prob_up_5d": round(h5["prob_up"], 3),
            "forecast_as_of": fc["as_of"],
            "forecast_summary": {
                "horizon_days": 10,
                "bull": round(current * float(np.exp(h10["q90"])), 2),
                "base": round(current * float(np.exp(h10["q50"])), 2),
                "bear": round(current * float(np.exp(h10["q10"])), 2),
            },
            "horizons": {
                k: {
                    "low": round(current * float(np.exp(v["q10"])), 2),
                    "median": round(current * float(np.exp(v["q50"])), 2),
                    "high": round(current * float(np.exp(v["q90"])), 2),
                    "prob_up": round(v["prob_up"], 3),
                }
                for k, v in fc["horizons"].items()
            },
        }
    )
    return payload


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------
def simulate(ticker: str, time_range: str) -> dict:
    sym = to_yahoo(ticker)
    time_range = normalize_range(time_range)
    window = "1mo" if time_range == "1d" else time_range
    prices = _window(get_history(sym, "2y"), window)
    preds = prediction_history(sym)
    if preds.empty:
        raise DataUnavailable("Model is not ready yet; try again shortly")
    result = simulate_single(prices, preds)
    return {
        "ticker": to_display(sym),
        "time_range": time_range,
        "window": window,
        "strategy": "Long when the 5-day forecast beats costs with >=55% odds; flat on a SELL signal. Fills at next open.",
        **result,
    }


def portfolio() -> dict:
    syms = [to_yahoo(s.ticker) for s in DASHBOARD]
    histories = get_histories(syms, "2y")
    preds = {s: prediction_history(s) for s in histories}
    preds = {s: p for s, p in preds.items() if not p.empty}
    if not preds:
        raise DataUnavailable("Model is not ready yet")
    first = min(p.index.min() for p in preds.values())
    end = max(h.index.max() for h in histories.values())
    start = max(first, end - pd.Timedelta(days=365))
    result = model_portfolio(histories, preds, start)
    for leg in result.get("trades", []):
        leg["ticker"] = to_display(leg["ticker"])
    for key in ("best_trade", "worst_trade"):
        if result.get(key):
            result[key] = {**result[key], "ticker": to_display(result[key]["ticker"])}
    for h in result.get("holdings", []):
        h["ticker"] = to_display(h["ticker"])
    return result


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
class Snapshot:
    def __init__(self) -> None:
        self.data: dict = {}
        self.updated_at: float | None = None
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def get(self, key: str):
        with self._lock:
            return self.data.get(key)

    def refresh(self) -> None:
        jobs = {"ratings": compute_ratings, "universe": universe_rows, "portfolio": portfolio}
        for key, fn in jobs.items():
            try:
                value = fn()
            except Exception as exc:  # noqa: BLE001
                log.warning("Snapshot job %s failed: %s", key, exc)
                with self._lock:
                    self.errors[key] = str(exc)
                continue
            with self._lock:
                self.data[key] = value
                self.errors.pop(key, None)
        with self._lock:
            self.updated_at = time.time()

    def status(self) -> dict:
        with self._lock:
            return {
                "updated_at": dt.datetime.fromtimestamp(self.updated_at, dt.UTC).isoformat(timespec="seconds")
                if self.updated_at
                else None,
                "ready": sorted(self.data),
                "errors": dict(self.errors),
            }

    def _loop(self) -> None:
        ensure_model()
        while not self._stop.is_set():
            started = time.time()
            self.refresh()
            log.info("Snapshot refreshed in %.1fs", time.time() - started)
            self._stop.wait(REFRESH_SECONDS)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="snapshot", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()


snapshot = Snapshot()


def ensure_model() -> None:
    """Load the shipped model, or fit a quick one if none exists."""
    if registry.get() is not None or not settings.train_on_boot:
        return
    from .training import train

    log.warning("No model artifact found; fitting a quick model (no walk-forward metrics)")
    registry.status = "training"
    try:
        train(quick=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("Boot training failed")
        registry.status, registry.error = "error", str(exc)
        return
    registry.reload()
