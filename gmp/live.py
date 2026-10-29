"""Live inference: today's features and forecasts for any tracked ticker."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .cache import TTLCache
from .features import HORIZONS, build_panel
from .market_data import DataUnavailable, get_benchmark, get_histories
from .model import QUANTILES, qcol, registry
from .universe import to_yahoo, training_universe

log = logging.getLogger(__name__)

_panel_cache = TTLCache(ttl_seconds=15 * 60, max_entries=8)
_pred_cache = TTLCache(ttl_seconds=15 * 60, max_entries=8)


def live_panel(extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Feature panel over the research universe (+ extra tickers), no targets.

    Cross-sectional rank features need the whole universe, the same way the
    model saw it in training.
    """
    tickers = tuple(dict.fromkeys(training_universe() + [to_yahoo(t) for t in extra]))

    def load() -> pd.DataFrame:
        histories = get_histories(tickers, "2y")
        if not histories:
            raise DataUnavailable("No price history available")
        bench = get_benchmark("2y")["Close"]
        return build_panel(histories, bench, with_targets=False)

    return _panel_cache.get_or_load(tickers, load)


def model_predictions(extra: tuple[str, ...] = ()) -> pd.DataFrame | None:
    """Model output (return quantiles + z) for every row of the live panel."""
    model = registry.get()
    if model is None:
        return None
    key = (tuple(extra), model.meta.get("trained_at"))

    def load() -> pd.DataFrame:
        panel = live_panel(extra)
        rows = panel.dropna(subset=["vol_21d", "ret_21d"])
        preds = model.predict_returns(rows)
        return preds.join(rows[["close", "daily_vol"]])

    return _pred_cache.get_or_load(key, load)


def skill_confidence() -> float:
    """0..1: how much to trust the median's *direction*, from out-of-sample rank IC.

    A t-stat of 3 or more counts as full confidence. No measured skill, or
    negative skill, means 0 and the UI shows the forecast as low-confidence.
    """
    model = registry.get()
    if model is None:
        return 0.0
    m5 = (model.meta.get("metrics") or {}).get("5d") or {}
    ic, t = m5.get("rank_ic_mean"), m5.get("rank_ic_tstat")
    if ic is None or t is None or ic <= 0:
        return 0.0
    return float(np.clip(t / 3.0, 0.0, 1.0))


def prob_up(q10: float, q50: float, q90: float) -> float:
    """P(return > 0) from three quantiles, using a piecewise-linear CDF."""
    xs, ps = np.array([q10, q50, q90]), np.array([0.1, 0.5, 0.9])
    if not np.all(np.diff(xs) > 1e-12):
        return 0.5
    if 0 <= xs[0]:
        lower_slope = 0.4 / (xs[1] - xs[0])
        cdf0 = 0.1 - (xs[0] - 0) * lower_slope
    elif 0 >= xs[2]:
        upper_slope = 0.4 / (xs[2] - xs[1])
        cdf0 = 0.9 + (0 - xs[2]) * upper_slope
    else:
        cdf0 = float(np.interp(0.0, xs, ps))
    return float(np.clip(1.0 - cdf0, 0.02, 0.98))


def latest_forecast(ticker: str) -> dict | None:
    """Most recent forecast for one ticker, or None if the model isn't ready."""
    sym = to_yahoo(ticker)
    preds = model_predictions((sym,))
    if preds is None:
        return None
    try:
        rows = preds.xs(sym, level="ticker")
    except KeyError:
        return None
    if rows.empty:
        return None
    last = rows.iloc[-1]
    out: dict = {
        "as_of": rows.index[-1].strftime("%Y-%m-%d"),
        "close": float(last["close"]),
        "daily_vol": float(last["daily_vol"]),
        "horizons": {},
    }
    for h in HORIZONS:
        q = {f"q{int(p * 100)}": float(last[qcol(p, h)]) for p in QUANTILES}
        out["horizons"][f"{h}d"] = {
            **q,
            "z50": float(last[f"z_{qcol(0.5, h)}"]),
            "prob_up": prob_up(q["q10"], q["q50"], q["q90"]),
        }
    return out


def prediction_history(ticker: str) -> pd.DataFrame:
    """Daily 5-day predictions for one ticker, never in-sample.

    Dates up to the end of the training data come from the walk-forward
    out-of-sample file; later dates come from the shipped model, which never
    saw them. Columns: q10_5d, q50_5d, q90_5d, z_q50_5d.
    """
    sym = to_yahoo(ticker)
    model = registry.get()
    if model is None:
        return pd.DataFrame()
    cols = [qcol(q, 5) for q in QUANTILES] + [f"z_{qcol(0.5, 5)}"]
    cutoff = pd.Timestamp(model.meta.get("data_end", "1900-01-01"))
    parts = []
    oos = registry.oos_predictions()
    if not oos.empty:
        try:
            parts.append(oos.xs(sym, level="ticker")[cols])
        except KeyError:
            pass
    live = model_predictions((sym,))
    if live is not None:
        try:
            recent = live.xs(sym, level="ticker")
            parts.append(recent.loc[recent.index > cutoff, cols])
        except KeyError:
            pass
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")]


def finite_or_none(x: float | None, digits: int = 4) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, digits) if math.isfinite(xf) else None
