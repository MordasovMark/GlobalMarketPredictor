import numpy as np
import pandas as pd
import pytest

from gmp.features import FEATURES, add_targets, build_panel, rsi, ticker_features
from tests.conftest import synthetic_ohlcv


def test_features_are_point_in_time():
    """Changing prices after day t must not change any feature dated <= t."""
    df = synthetic_ohlcv(1, n=400)
    bench = synthetic_ohlcv(2, n=400)["Close"]
    cut = 300
    base = ticker_features(df, bench)

    shocked = df.copy()
    shocked.iloc[cut + 1 :, :4] *= 1.5  # big jump in the future
    shocked.iloc[cut + 1 :, 4] *= 3
    bench_shocked = bench.copy()
    bench_shocked.iloc[cut + 1 :] *= 0.7
    after = ticker_features(shocked, bench_shocked)

    cols = [c for c in FEATURES if c in base.columns]
    pd.testing.assert_frame_equal(base.iloc[: cut + 1][cols], after.iloc[: cut + 1][cols])


def test_targets_look_forward_exactly_h_days():
    df = synthetic_ohlcv(3, n=120)
    bench = synthetic_ohlcv(4, n=120)["Close"]
    panel = build_panel({"AAA": df, "BBB": synthetic_ohlcv(5, n=120)}, bench, with_targets=False)
    panel = add_targets(panel)
    rows = panel.xs("AAA", level="ticker")
    expected = np.log(df["Close"].shift(-5) / df["Close"])
    np.testing.assert_allclose(rows["fwd_ret_5d"].dropna(), expected.dropna().loc[rows["fwd_ret_5d"].dropna().index])
    assert rows["fwd_ret_5d"].iloc[-5:].isna().all()


def test_features_are_scale_free():
    """Multiplying all prices by 10 leaves features unchanged (no raw levels)."""
    df = synthetic_ohlcv(6, n=300)
    bench = synthetic_ohlcv(7, n=300)["Close"]
    a = ticker_features(df, bench)[[c for c in FEATURES if not c.startswith("xs_")]]
    scaled = df.copy()
    scaled.iloc[:, :4] *= 10
    b = ticker_features(scaled, bench)[a.columns]
    pd.testing.assert_frame_equal(a, b, atol=1e-9, rtol=1e-7)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(1, 2, 60))
    assert rsi(up).dropna().iloc[-1] == pytest.approx(100.0)
    noisy = pd.Series(synthetic_ohlcv(8, n=200)["Close"].to_numpy())
    vals = rsi(noisy).dropna()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_cross_sectional_ranks_in_unit_interval():
    bench = synthetic_ohlcv(9, n=200)["Close"]
    panel = build_panel({f"T{i}": synthetic_ohlcv(10 + i, n=200) for i in range(6)}, bench, with_targets=False)
    ranks = panel["xs_rank_ret_21d"].dropna()
    assert ranks.between(0, 1).all()
