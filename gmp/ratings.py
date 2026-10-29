"""Multi-model consensus rating (0-100) for each dashboard stock.

Three independent engines, each scored 0-100 against the 20 peer stocks:

* **DeepValue Engine** – fundamentals: earnings / free-cash-flow / book yield
  (value), ROE, margins and leverage (quality), revenue and EPS growth.
* **Sentiment NLP** – recency-weighted sentiment of the last week of headlines.
* **Tech-Momentum Net** – trend and relative-strength factors, blended with
  the ML forecaster's probability of a 5-day gain. The ML part's weight is
  set by the model's measured out-of-sample skill, so an unproven model can't
  dominate the rating.

Consensus rule (as described in the UI): Buy needs at least 2 of 3 engines
bullish (>= 60); Sell needs at least 2 bearish (<= 40). Otherwise the score is
clamped into the Hold band.

News sentiment is a separate engine rather than a model feature on purpose:
there is no point-in-time news archive to train on, and back-filling one
would leak future information into the backtest.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .features import indicator_series
from .live import latest_forecast, live_panel, skill_confidence
from .market_data import DataUnavailable, get_fundamentals, get_history
from .news import ticker_news
from .signals import score_to_label
from .universe import DASHBOARD, Stock, to_yahoo

log = logging.getLogger(__name__)

WEIGHTS = {"value": 0.35, "sentiment": 0.25, "momentum": 0.40}
ENGINE_NAMES = {"value": "DeepValue Engine", "sentiment": "Sentiment NLP", "momentum": "Tech-Momentum Net"}
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ratings")


def pct_rank(values: dict[str, float | None]) -> dict[str, float | None]:
    """Percentile (0-100) of each value among the non-missing ones."""
    s = pd.Series({k: v for k, v in values.items() if v is not None and np.isfinite(v)}, dtype=float)
    if len(s) < 3:
        return {k: None for k in values}
    ranks = s.rank(pct=True) * 100
    return {k: (float(ranks[k]) if k in ranks else None) for k in values}


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _fmt_pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{digits}f}%"


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
def value_engine(fund: dict[str, dict]) -> dict[str, dict]:
    def ratio(f, num, den):
        a, b = f.get(num), f.get(den)
        return a / b if a is not None and b not in (None, 0) else None

    def inv(f, key):
        v = f.get(key)
        return 1 / v if v not in (None, 0) and v > 0 else None

    raw = {
        "earnings_yield": {t: inv(f, "forward_pe") or inv(f, "trailing_pe") for t, f in fund.items()},
        "fcf_yield": {t: ratio(f, "free_cashflow", "market_cap") for t, f in fund.items()},
        "book_yield": {t: inv(f, "price_to_book") for t, f in fund.items()},
        "roe": {t: f.get("roe") for t, f in fund.items()},
        "margin": {t: f.get("profit_margin") for t, f in fund.items()},
        "low_leverage": {
            t: (-f["debt_to_equity"] if f.get("debt_to_equity") is not None else None) for t, f in fund.items()
        },
        "rev_growth": {t: f.get("revenue_growth") for t, f in fund.items()},
        "eps_growth": {t: f.get("earnings_growth") for t, f in fund.items()},
    }
    ranks = {k: pct_rank(v) for k, v in raw.items()}
    out = {}
    for t, f in fund.items():
        value = _mean([ranks[k][t] for k in ("earnings_yield", "fcf_yield", "book_yield")])
        quality = _mean([ranks[k][t] for k in ("roe", "margin", "low_leverage")])
        growth = _mean([ranks[k][t] for k in ("rev_growth", "eps_growth")])
        parts = [(0.45, value), (0.35, quality), (0.20, growth)]
        avail = [(w, s) for w, s in parts if s is not None]
        score = sum(w * s for w, s in avail) / sum(w for w, _ in avail) if avail else None
        pe = f.get("forward_pe") or f.get("trailing_pe")
        fcf = raw["fcf_yield"][t]
        bits = []
        if pe:
            bits.append(f"P/E {pe:.1f}x")
        if fcf is not None:
            bits.append(f"FCF yield {fcf * 100:.1f}%")
        if f.get("roe") is not None:
            bits.append(f"ROE {f['roe'] * 100:.0f}%")
        if f.get("profit_margin") is not None:
            bits.append(f"net margin {f['profit_margin'] * 100:.0f}%")
        if f.get("revenue_growth") is not None:
            bits.append(f"revenue growth {_fmt_pct(f['revenue_growth'], 0)}")
        if f.get("debt_to_equity") is not None:
            bits.append(f"debt/equity {f['debt_to_equity'] / 100:.2f}x")
        if score is None:
            text = "Fundamental data is unavailable right now, so this engine abstains."
        else:
            text = (
                f"{', '.join(bits)}. Versus the 20 tracked peers: value {value:.0f}th, quality "
                f"{quality if quality is not None else 0:.0f}th, growth {growth if growth is not None else 0:.0f}th percentile."
            )
        out[t] = {"score": score, "value": value, "quality": quality, "growth": growth, "justification": text}
    return out


def momentum_engine(latest: pd.DataFrame, forecasts: dict[str, dict | None], confidence: float) -> dict[str, dict]:
    cols = ["rel_ret_63d", "ret_126d", "dist_sma200", "macd_hist_norm"]
    ranks = {c: pct_rank(latest[c].to_dict()) for c in cols if c in latest}
    out = {}
    for t, row in latest.iterrows():
        relative = _mean([ranks[c][t] for c in ranks])
        checks = [
            row.get("dist_sma200", np.nan) > 0,
            row.get("dist_sma50", np.nan) > 0,
            row.get("macd_hist_norm", np.nan) > 0,
            30 <= row.get("rsi_14", 50) <= 70,
        ]
        absolute = 100 * float(np.mean(checks))
        tech = 0.5 * absolute + 0.5 * (relative if relative is not None else 50)
        adx_v = float(row.get("adx_14", np.nan))
        if np.isfinite(adx_v) and adx_v > 25:  # strong trend: trust the direction more
            tech = 50 + (tech - 50) * 1.15
        fc = forecasts.get(t)
        ml_score = None
        if fc:
            p = fc["horizons"]["5d"]["prob_up"]
            ml_score = 50 + 50 * float(np.clip((p - 0.5) / 0.15, -1, 1))
        w_ml = 0.5 * confidence if ml_score is not None else 0.0
        score = float(np.clip((1 - w_ml) * tech + w_ml * (ml_score or 50), 0, 100))

        text = (
            f"Price is {_fmt_pct(row.get('dist_sma200'))} vs its 200-day and {_fmt_pct(row.get('dist_sma50'))} vs its "
            f"50-day average; 3-month return vs S&P 500 {_fmt_pct(row.get('rel_ret_63d'))}; RSI {row.get('rsi_14', float('nan')):.0f}, "
            f"ADX {adx_v:.0f}."
        )
        if fc:
            h5 = fc["horizons"]["5d"]
            text += (
                f" ML forecast: median 5-day move {_fmt_pct(h5['q50'])} (80% range {_fmt_pct(h5['q10'])} to "
                f"{_fmt_pct(h5['q90'])}), {h5['prob_up'] * 100:.0f}% chance of a gain"
                + (
                    f"; weighted {w_ml * 100:.0f}% by measured skill."
                    if w_ml > 0
                    else "; not yet weighted because out-of-sample skill isn't statistically significant."
                )
            )
        out[t] = {"score": score, "tech": tech, "ml": ml_score, "ml_weight": w_ml, "justification": text}
    return out


def consensus(engine_scores: dict[str, float | None]) -> float:
    avail = {k: v for k, v in engine_scores.items() if v is not None}
    if not avail:
        return 50.0
    score = sum(WEIGHTS[k] * v for k, v in avail.items()) / sum(WEIGHTS[k] for k in avail)
    bulls = sum(v >= 60 for v in avail.values())
    bears = sum(v <= 40 for v in avail.values())
    need = 2 if len(avail) >= 2 else 1
    if score >= 66 and bulls < need:
        score = 65.0
    if score <= 40 and bears < need:
        score = 41.0
    return round(float(score), 1)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - one bad ticker must not sink the page
        log.info("%s failed for %s: %s", getattr(fn, "__name__", fn), args, exc)
        return default


def compute_ratings(stocks: tuple[Stock, ...] = DASHBOARD) -> dict[str, dict]:
    syms = {s.ticker: to_yahoo(s.ticker) for s in stocks}
    panel = live_panel(tuple(syms.values()))
    last_date = panel.index.get_level_values("date").max()
    latest_rows = {}
    for disp, sym in syms.items():
        try:
            latest_rows[disp] = panel.xs(sym, level="ticker").iloc[-1]
        except (KeyError, IndexError):
            continue
    if not latest_rows:
        raise DataUnavailable("No live features for the dashboard universe")
    latest = pd.DataFrame(latest_rows).T.astype(float)

    tickers = list(latest.index)
    fund = dict(zip(tickers, _pool.map(lambda t: _safe(get_fundamentals, t, default={}) or {}, tickers)))
    news = dict(zip(tickers, _pool.map(lambda t: _safe(ticker_news, t, 4), tickers)))
    forecasts = {t: _safe(latest_forecast, t) for t in tickers}
    confidence = skill_confidence()

    value = value_engine(fund)
    momentum = momentum_engine(latest, forecasts, confidence)
    out = {}
    for t in tickers:
        n = news.get(t)
        n_count = (n or {}).get("headline_count", 0)
        senti = n["news_sentiment_score"] if n and n_count else None
        engines = {"value": value[t]["score"], "sentiment": senti, "momentum": momentum[t]["score"]}
        score = consensus(engines)
        label = score_to_label(score)
        fc = forecasts.get(t)
        senti_text = n["thesis"] if n and n_count else "No recent headlines, so this engine abstains."
        models = [
            {
                "name": ENGINE_NAMES["value"],
                "key": "value",
                "score": _round(engines["value"]),
                "justification": value[t]["justification"],
            },
            {
                "name": ENGINE_NAMES["sentiment"],
                "key": "sentiment",
                "score": _round(engines["sentiment"]),
                "justification": senti_text,
            },
            {
                "name": ENGINE_NAMES["momentum"],
                "key": "momentum",
                "score": _round(engines["momentum"]),
                "justification": momentum[t]["justification"],
            },
        ]
        rated = [m for m in models if m["score"] is not None]
        best = max(rated, key=lambda m: m["score"]) if rated else None
        worst = min(rated, key=lambda m: m["score"]) if rated else None
        summary = f"{label} at {score:.0f}/100 from {len(rated)} of 3 engines."
        if best and worst and best is not worst:
            summary += (
                f" Strongest: {best['name']} ({best['score']:.0f}); weakest: {worst['name']} ({worst['score']:.0f})."
            )
        if fc:
            h5 = fc["horizons"]["5d"]
            summary += f" Model's 5-day median {_fmt_pct(h5['q50'])}, 80% range {_fmt_pct(h5['q10'])} to {_fmt_pct(h5['q90'])}."

        out[t] = {
            "ticker": t,
            "score": score,
            "label": label,
            "summary": summary,
            "as_of": last_date.strftime("%Y-%m-%d"),
            "models": models,
            "factors": [
                {"label": "Growth", "score": _round(value[t]["growth"])},
                {"label": "Value", "score": _round(value[t]["value"])},
                {"label": "Momentum", "score": _round(momentum[t]["tech"])},
                {"label": "Quality", "score": _round(value[t]["quality"])},
            ],
            "news": {
                "score": senti,
                "headline_count": n_count,
                "thesis": (n or {}).get("thesis", ""),
                "is_live": bool(n and n.get("is_live")),
                "daily_flow": (n or {}).get("daily_flow", []),
                "backend": (n or {}).get("sentiment_backend"),
            },
            "forecast_5d": fc["horizons"]["5d"] if fc else None,
            "ml_weight": round(momentum[t]["ml_weight"], 3),
        }
    return out


def rating_detail(rating: dict) -> dict:
    """Add chart series for the 'how the engines work' panel."""
    t = rating["ticker"]
    try:
        hist = get_history(t, "1y")
        momentum_series = indicator_series(hist, 20)
    except DataUnavailable:
        momentum_series = []
    return {
        **rating,
        "charts": {
            "value_factors": [
                {"factor": f["label"], "percentile": f["score"], "peer_median": 50}
                for f in rating["factors"]
                if f["label"] != "Momentum" and f["score"] is not None
            ],
            "sentiment_flow": rating["news"]["daily_flow"],
            "momentum": momentum_series,
        },
    }


def _round(x: float | None, n: int = 0) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), n) if n else float(round(x))
