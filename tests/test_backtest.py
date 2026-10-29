import pandas as pd
import pytest

from gmp.backtest import model_portfolio, simulate_single


def _prices(closes, opens=None):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    opens = opens or closes
    return pd.DataFrame({"Open": opens, "High": closes, "Low": closes, "Close": closes, "Volume": 1.0}, index=idx)


def _preds(index, q50s):
    rows = [{"q10_5d": q - 0.03, "q50_5d": q, "q90_5d": q + 0.03, "z_q50_5d": q * 10} for q in q50s]
    return pd.DataFrame(rows, index=index)


def test_fills_at_next_open_not_signal_close():
    prices = _prices([100, 100, 100, 100], opens=[100, 110, 120, 130])
    preds = _preds(prices.index, [0.02, 0.0, 0.0, 0.0])  # BUY on day 0 only
    out = simulate_single(prices, preds, initial=10_000, cost_bps=0)
    assert out["trades"][0]["action"] == "BUY"
    assert out["trades"][0]["execution_price"] == 110  # day 1 open, not day 0 close
    assert out["trades"][0]["date"] == prices.index[1].strftime("%Y-%m-%d")


def test_costs_are_charged():
    prices = _prices([100.0] * 6)
    preds = _preds(prices.index, [0.02, 0.0, -0.02, 0.0, 0.0, 0.0])  # buy, then sell
    free = simulate_single(prices, preds, cost_bps=0)
    costly = simulate_single(prices, preds, cost_bps=50)
    assert free["final_balance"] == pytest.approx(10_000)
    assert costly["final_balance"] < free["final_balance"]
    assert costly["trades"][-1]["pnl"] < 0


def test_no_signal_means_no_trades():
    prices = _prices([100, 101, 102, 103])
    out = simulate_single(prices, _preds(prices.index, [0.0] * 4))
    assert out["n_trades"] == 0 and out["final_balance"] == pytest.approx(10_000)


def test_model_portfolio_tracks_cash_and_positions():
    idx = pd.bdate_range("2024-01-01", periods=30)
    histories = {
        t: pd.DataFrame({"Open": p, "High": p, "Low": p, "Close": p, "Volume": 1.0}, index=idx)
        for t, p in {
            "UP": [100 + i for i in range(30)],
            "FLAT": [50.0] * 30,
            "DOWN": [80 - i for i in range(30)],
        }.items()
    }
    preds = {
        "UP": _preds(idx, [0.03] * 30),
        "FLAT": _preds(idx, [0.0] * 30),
        "DOWN": _preds(idx, [-0.03] * 30),
    }
    out = model_portfolio(histories, preds, idx[0], capital=10_000, top_k=1, cost_bps=0)
    assert [h["ticker"] for h in out["holdings"]] == ["UP"]
    assert all(t["ticker"] == "UP" for t in out["trades"])
    assert out["total_return_pct"] > 0
    held = out["holdings"][0]
    assert out["cash"] + held["marketValue"] == pytest.approx(out["equity"], rel=1e-6)
