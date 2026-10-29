"""Turning a return forecast into BUY / SELL / HOLD.

The rule is deliberately simple and explainable:

* BUY  when the median 5-day return beats the round-trip trading cost *and*
  the model gives at least a 55% chance of the price being higher.
* SELL when the mirror image holds.
* HOLD otherwise.
"""

from __future__ import annotations

ROUND_TRIP_COST = 0.002  # 20 bps: commission + spread + slippage, both sides
MIN_PROB_EDGE = 0.55


def trade_signal(q50: float, p_up: float) -> str:
    if q50 > ROUND_TRIP_COST and p_up >= MIN_PROB_EDGE:
        return "BUY"
    if q50 < -ROUND_TRIP_COST and p_up <= 1 - MIN_PROB_EDGE:
        return "SELL"
    return "HOLD"


def score_to_label(score: float) -> str:
    """Same bands as the UI's getSignal()."""
    if score >= 85:
        return "Strong Buy"
    if score >= 66:
        return "Buy"
    if score >= 41:
        return "Hold"
    return "Sell"
