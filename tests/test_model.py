import numpy as np
import pytest

from gmp.features import build_panel
from gmp.live import prob_up
from gmp.model import QuantileForecaster, qcol
from tests.conftest import synthetic_ohlcv


@pytest.fixture(scope="module")
def panel():
    bench = synthetic_ohlcv(100, n=500)["Close"]
    histories = {f"S{i}": synthetic_ohlcv(200 + i, n=500) for i in range(8)}
    return build_panel(histories, bench).dropna(subset=["ret_63d", "vol_21d"])


def test_fit_predict_quantiles_are_ordered(panel):
    model = QuantileForecaster().fit(panel)
    preds = model.predict_returns(panel.tail(200))
    for h in model.horizons:
        lo, mid, hi = (preds[qcol(q, h)] for q in model.quantiles)
        assert (lo <= mid).all() and (mid <= hi).all()


def test_save_load_roundtrip(panel, tmp_path):
    model = QuantileForecaster().fit(panel)
    model.meta = {"trained_at": "now"}
    model.save(tmp_path)
    loaded = QuantileForecaster.load(tmp_path)
    rows = panel.tail(50)
    np.testing.assert_allclose(model.predict_z(rows), loaded.predict_z(rows))
    assert (tmp_path / "model_card.json").is_file()


def test_prob_up_is_monotone_and_bounded():
    assert prob_up(-0.02, 0.0, 0.02) == pytest.approx(0.5)
    assert prob_up(-0.01, 0.01, 0.03) > 0.5 > prob_up(-0.03, -0.01, 0.01)
    assert 0.02 <= prob_up(0.05, 0.06, 0.07) <= 0.98
    assert 0.02 <= prob_up(-0.07, -0.06, -0.05) <= 0.98
    assert prob_up(0.0, 0.0, 0.0) == 0.5  # degenerate quantiles
