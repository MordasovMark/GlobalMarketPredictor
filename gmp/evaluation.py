"""Out-of-sample evaluation metrics for the forecaster.

Everything here is computed on walk-forward predictions, i.e. each prediction
was made by a model that never saw that date (or the 10 days before it).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model import QUANTILES, qcol


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _safe(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


def daily_rank_ic(scores: pd.Series, realized: pd.Series) -> pd.Series:
    """Spearman correlation across tickers, one value per date."""
    frame = pd.DataFrame({"s": scores, "r": realized}).dropna()

    def one(day: pd.DataFrame) -> float:
        if len(day) < 5:
            return np.nan
        return day["s"].rank().corr(day["r"].rank())

    return frame.groupby(level="date").apply(one).dropna()


def horizon_metrics(oos: pd.DataFrame, panel: pd.DataFrame, h: int, train_z: pd.Series) -> dict:
    """Accuracy, calibration and skill vs a no-information baseline for one horizon."""
    target = f"target_z_{h}d"
    realized_ret = panel.loc[oos.index, f"fwd_ret_{h}d"]
    realized_z = panel.loc[oos.index, target].clip(-6, 6)
    mask = realized_ret.notna()
    oos, realized_ret, realized_z = oos[mask], realized_ret[mask], realized_z[mask]
    if oos.empty:
        return {"n": 0}

    med = oos[qcol(0.5, h)]
    lo, hi = oos[qcol(0.1, h)], oos[qcol(0.9, h)]
    nonzero = realized_ret != 0
    hit = float((np.sign(med[nonzero]) == np.sign(realized_ret[nonzero])).mean())
    # Baseline 1: always predict "up" (stocks drift up; a fair bar to beat).
    always_up = float((realized_ret[nonzero] > 0).mean())
    coverage = float(((realized_ret >= lo) & (realized_ret <= hi)).mean())

    # Pinball skill vs "climatology": the unconditional training quantiles of z.
    skill = {}
    for q in QUANTILES:
        model_loss = pinball(realized_z.to_numpy(), oos[f"z_{qcol(q, h)}"].to_numpy(), q)
        base_loss = pinball(realized_z.to_numpy(), np.full(len(realized_z), train_z.quantile(q)), q)
        skill[f"q{int(q * 100)}"] = _safe(1 - model_loss / base_loss) if base_loss > 0 else None

    ic = daily_rank_ic(oos[f"z_{qcol(0.5, h)}"], realized_ret)
    ic_mean = float(ic.mean()) if len(ic) else float("nan")
    ic_t = float(ic_mean / ic.std() * math.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else float("nan")

    return {
        "n": int(len(oos)),
        "hit_rate": _safe(hit),
        "always_up_hit_rate": _safe(always_up),
        "interval_coverage_80": _safe(coverage),
        "pinball_skill": skill,
        "rank_ic_mean": _safe(ic_mean),
        "rank_ic_tstat": _safe(ic_t),
        "mae_median_vs_zero": _safe(
            1 - float(np.mean(np.abs(realized_ret - med))) / float(np.mean(np.abs(realized_ret)))
        ),
    }


def quintile_backtest(
    oos: pd.DataFrame,
    panel: pd.DataFrame,
    h: int = 5,
    cost_bps: float = 10.0,
    top_frac: float = 0.2,
) -> dict:
    """Non-overlapping h-day rebalances: long top quintile vs bottom, with costs.

    Costs are charged on actual turnover (sum of absolute weight changes).
    """
    score = oos[f"z_{qcol(0.5, h)}"]
    fwd = np.expm1(panel.loc[oos.index, f"fwd_ret_{h}d"])
    frame = pd.DataFrame({"score": score, "fwd": fwd}).dropna()
    dates = frame.index.get_level_values("date").unique().sort_values()
    rebal = dates[::h]

    prev_long: pd.Series | None = None
    prev_ls: pd.Series | None = None
    rows = []
    for d in rebal:
        day = frame.xs(d, level="date")
        if len(day) < 10:
            continue
        k = max(1, int(len(day) * top_frac))
        ranked = day.sort_values("score")
        longs, shorts = ranked.index[-k:], ranked.index[:k]
        w_long = pd.Series(1.0 / k, index=longs)
        w_ls = pd.concat([w_long, pd.Series(-1.0 / k, index=shorts)])

        def turnover(new: pd.Series, old: pd.Series | None) -> float:
            if old is None:
                return float(new.abs().sum())
            return float(new.sub(old, fill_value=0.0).abs().sum())

        c = cost_bps / 1e4
        long_ret = float(day.loc[longs, "fwd"].mean()) - c * turnover(w_long, prev_long)
        ls_ret = float((w_ls * day.loc[w_ls.index, "fwd"]).sum()) - c * turnover(w_ls, prev_ls)
        rows.append({"date": d, "long": long_ret, "long_short": ls_ret, "universe": float(day["fwd"].mean())})
        prev_long, prev_ls = w_long, w_ls

    if not rows:
        return {}
    res = pd.DataFrame(rows).set_index("date")
    periods = 252 / h

    def stats(r: pd.Series) -> dict:
        equity = (1 + r).cumprod()
        years = len(r) / periods
        cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else float("nan")
        sharpe = r.mean() / r.std() * math.sqrt(periods) if r.std() > 0 else float("nan")
        dd = float((equity / equity.cummax() - 1).min())
        return {
            "cagr": _safe(cagr),
            "sharpe": _safe(sharpe),
            "max_drawdown": _safe(dd),
            "total_return": _safe(equity.iloc[-1] - 1),
        }

    curve = (1 + res).cumprod()
    step = max(1, len(curve) // 120)
    return {
        "horizon_days": h,
        "cost_bps_per_side": cost_bps,
        "rebalances": int(len(res)),
        "long_top_quintile": stats(res["long"]),
        "long_short_quintiles": stats(res["long_short"]),
        "equal_weight_universe": stats(res["universe"]),
        "equity_curve": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "long": round(float(row["long"]), 4),
                "long_short": round(float(row["long_short"]), 4),
                "universe": round(float(row["universe"]), 4),
            }
            for d, row in curve.iloc[::step].iterrows()
        ],
    }
