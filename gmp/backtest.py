"""Paper-trading simulations driven by out-of-sample model forecasts.

Execution model (kept honest on purpose):

* Signals are computed from data up to the close of day *t*.
* Orders fill at the **open of day t+1**, never at the close that produced
  the signal.
* Every fill pays ``cost_bps`` (commission + spread + slippage).
* Whole shares only; no leverage, no shorting.
"""

from __future__ import annotations

import math

import pandas as pd

from .live import prob_up
from .signals import ROUND_TRIP_COST, trade_signal

DEFAULT_COST_BPS = 10.0


def _fmt(d: pd.Timestamp, intraday: bool = False) -> str:
    return d.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")


def _r(x: float, n: int = 2) -> float | None:
    return round(float(x), n) if x is not None and math.isfinite(float(x)) else None


def _risk_stats(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * math.sqrt(252) if len(rets) > 2 and rets.std() > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")
    return {"sharpe": _r(sharpe), "max_drawdown_pct": _r(dd * 100)}


def _signal_for(preds: pd.DataFrame, d: pd.Timestamp) -> str | None:
    if d not in preds.index:
        return None
    row = preds.loc[d]
    return trade_signal(float(row["q50_5d"]), prob_up(row["q10_5d"], row["q50_5d"], row["q90_5d"]))


# ---------------------------------------------------------------------------
# Single stock: $10k, all-in long or flat
# ---------------------------------------------------------------------------
def simulate_single(
    prices: pd.DataFrame,
    preds: pd.DataFrame,
    initial: float = 10_000.0,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict:
    c = cost_bps / 1e4
    cash, shares, entry_px, entry_cash = initial, 0, 0.0, 0.0
    trades: list[dict] = []
    equity = []
    days_invested = 0
    opens = prices["Open"].where(prices["Open"] > 0, prices["Close"])
    closes = prices["Close"]
    dates = prices.index
    covered = int(sum(d in preds.index for d in dates))

    equity.append(initial)
    for i in range(len(dates) - 1):
        sig = _signal_for(preds, dates[i])
        px = float(opens.iloc[i + 1])
        when = _fmt(dates[i + 1])
        if sig == "BUY" and shares == 0:
            qty = int(cash // (px * (1 + c)))
            if qty > 0:
                entry_cash = qty * px * (1 + c)
                cash -= entry_cash
                shares, entry_px = qty, px
                trades.append({"date": when, "action": "BUY", "shares": qty, "execution_price": _r(px)})
        elif sig == "SELL" and shares > 0:
            proceeds = shares * px * (1 - c)
            pnl = proceeds - entry_cash
            trades.append(
                {
                    "date": when,
                    "action": "SELL",
                    "shares": shares,
                    "execution_price": _r(px),
                    "pnl": _r(pnl),
                    "pnl_pct": _r(pnl / entry_cash * 100),
                }
            )
            cash += proceeds
            shares = 0
        days_invested += shares > 0
        equity.append(cash + shares * float(closes.iloc[i + 1]))

    eq = pd.Series(equity, index=dates)
    final = float(eq.iloc[-1])
    first_open = float(opens.iloc[0])
    bh_shares = int(initial // (first_open * (1 + c)))
    bh_final = initial - bh_shares * first_open * (1 + c) + bh_shares * float(closes.iloc[-1]) * (1 - c)
    closed = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in closed if (t.get("pnl") or 0) > 0]
    in_market = days_invested / max(1, len(dates) - 1)
    step = max(1, len(eq) // 150)
    return {
        "initial_balance": _r(initial),
        "final_balance": _r(final),
        "total_roi_pct": _r((final / initial - 1) * 100),
        "buy_hold_roi_pct": _r((bh_final / initial - 1) * 100),
        "open_position": {"shares": shares, "entry_price": _r(entry_px)} if shares else None,
        "trades": trades,
        "n_trades": len(trades),
        "win_rate_pct": _r(len(wins) / len(closed) * 100) if closed else None,
        "exposure_pct": _r(in_market * 100),
        "signal_coverage_pct": _r(covered / len(dates) * 100) if len(dates) else 0.0,
        "cost_bps_per_side": cost_bps,
        **_risk_stats(eq),
        "equity_curve": [{"date": _fmt(d), "equity": _r(v)} for d, v in eq.iloc[::step].items()],
    }


# ---------------------------------------------------------------------------
# Multi-stock model portfolio
# ---------------------------------------------------------------------------
def model_portfolio(
    histories: dict[str, pd.DataFrame],
    preds: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    capital: float = 100_000.0,
    top_k: int = 5,
    rebalance_every: int = 5,
    cost_bps: float = DEFAULT_COST_BPS,
    min_trade_value: float = 250.0,
) -> dict:
    """Weekly: hold the ``top_k`` names with the best risk-adjusted 5-day forecast.

    Only names whose median forecast beats trading costs qualify; empty slots
    stay in cash. Positions are equal weight at each rebalance.
    """
    c = cost_bps / 1e4
    syms = sorted(histories)
    opens = pd.DataFrame({s: histories[s]["Open"].where(histories[s]["Open"] > 0, histories[s]["Close"]) for s in syms})
    closes = pd.DataFrame({s: histories[s]["Close"] for s in syms})
    closes = closes[closes.index >= start].ffill()
    opens = opens.reindex(closes.index).fillna(closes)
    dates = closes.index
    if len(dates) < rebalance_every + 2:
        return {"error": "Not enough history for the model portfolio"}

    cash = capital
    pos: dict[str, int] = {}
    avg_cost: dict[str, float] = {}
    legs: list[dict] = []
    equity = [capital]

    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        if i % rebalance_every == 0:
            scored = []
            for s in syms:
                p = preds.get(s)
                if p is None or d not in p.index:
                    continue
                row = p.loc[d]
                if float(row["q50_5d"]) > ROUND_TRIP_COST / 2:
                    scored.append((float(row["z_q50_5d"]), s))
            picks = [s for _, s in sorted(scored, reverse=True)[:top_k]]
            px = opens.loc[nxt]
            nav = cash + sum(q * float(px[s]) for s, q in pos.items())
            target = {s: int((nav / top_k) // (float(px[s]) * (1 + c))) for s in picks}
            # Sells first so the cash is there for buys.
            for s in sorted(set(pos) | set(target), key=lambda k: target.get(k, 0) - pos.get(k, 0)):
                delta = target.get(s, 0) - pos.get(s, 0)
                price = float(px[s])
                is_resize = target.get(s, 0) > 0 and pos.get(s, 0) > 0
                if delta == 0 or (is_resize and abs(delta) * price < min_trade_value):
                    continue  # don't churn tiny weight drifts
                if delta < 0:
                    qty = -delta
                    proceeds = qty * price * (1 - c)
                    pnl = proceeds - qty * avg_cost[s]
                    cash += proceeds
                    pos[s] -= qty
                    legs.append(
                        {
                            "date": _fmt(nxt),
                            "ticker": s,
                            "action": "SELL",
                            "shares": qty,
                            "execution_price": _r(price),
                            "pl": _r(pnl),
                            "pl_pct": _r(pnl / (qty * avg_cost[s]) * 100),
                            "isWin": pnl > 0,
                        }
                    )
                    if pos[s] == 0:
                        del pos[s], avg_cost[s]
                else:
                    qty = min(delta, int(cash // (price * (1 + c))))
                    if qty <= 0:
                        continue
                    spend = qty * price * (1 + c)
                    held = pos.get(s, 0)
                    avg_cost[s] = (avg_cost.get(s, 0.0) * held + spend) / (held + qty)
                    pos[s] = held + qty
                    cash -= spend
                    legs.append(
                        {
                            "date": _fmt(nxt),
                            "ticker": s,
                            "action": "BUY",
                            "shares": qty,
                            "execution_price": _r(price),
                            "pl": None,
                            "isWin": None,
                        }
                    )
        equity.append(cash + sum(q * float(closes.loc[nxt, s]) for s, q in pos.items()))

    eq = pd.Series(equity, index=dates)
    # Benchmark: equal weight in every name on day one, then hold.
    first_open = opens.iloc[0]
    alloc = capital / len(syms)
    bench_shares = {s: alloc * (1 - c) / float(first_open[s]) for s in syms}
    bench = closes.apply(lambda row: sum(bench_shares[s] * float(row[s]) for s in syms), axis=1)

    last_close = closes.iloc[-1]
    holdings = [
        {
            "ticker": s,
            "shares": q,
            "avgCost": _r(avg_cost[s]),
            "price": _r(float(last_close[s])),
            "marketValue": _r(q * float(last_close[s])),
            "costBasis": _r(q * avg_cost[s]),
            "pl": _r(q * (float(last_close[s]) - avg_cost[s])),
            "plPct": _r((float(last_close[s]) / avg_cost[s] - 1) * 100),
        }
        for s, q in sorted(pos.items(), key=lambda kv: -kv[1] * float(last_close[kv[0]]))
    ]
    closed = [leg for leg in legs if leg["action"] == "SELL"]
    wins = [leg for leg in closed if leg["isWin"]]
    best = max(closed, key=lambda leg: leg["pl"], default=None)
    worst = min(closed, key=lambda leg: leg["pl"], default=None)
    step = max(1, len(eq) // 150)
    return {
        "start": _fmt(dates[0]),
        "end": _fmt(dates[-1]),
        "capital": _r(capital),
        "cash": _r(cash),
        "equity": _r(float(eq.iloc[-1])),
        "total_return_pct": _r((eq.iloc[-1] / capital - 1) * 100),
        "benchmark_return_pct": _r((bench.iloc[-1] / capital - 1) * 100),
        **_risk_stats(eq),
        "rules": {
            "top_k": top_k,
            "rebalance_every_days": rebalance_every,
            "cost_bps_per_side": cost_bps,
            "execution": "next session open",
        },
        "total_trades": len(legs),
        "closed_trades": len(closed),
        "win_rate_pct": _r(len(wins) / len(closed) * 100) if closed else None,
        "best_trade": best,
        "worst_trade": worst,
        "holdings": holdings,
        "trades": list(reversed(legs)),
        "equity_curve": [
            {"date": _fmt(d), "model": _r(eq.loc[d]), "benchmark": _r(bench.loc[d])} for d in eq.index[::step]
        ],
    }
