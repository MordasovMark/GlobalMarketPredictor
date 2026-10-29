"""Test setup: fully offline.

Settings are read at import time, so the environment is configured here,
before any ``gmp`` module is imported. Prices are synthetic random walks
written as yfinance-style CSVs; no network is ever used.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_TMP = Path(tempfile.mkdtemp(prefix="gmp-tests-"))
PRICE_DIR = _TMP / "prices"
ARTIFACT_DIR = _TMP / "artifacts"
PRICE_DIR.mkdir()
ARTIFACT_DIR.mkdir()

os.environ.update(
    {
        "GMP_PRICE_CSV_DIR": str(PRICE_DIR),
        "GMP_ARTIFACT_DIR": str(ARTIFACT_DIR),
        "GMP_NETWORK": "0",
        "GMP_WARM_ON_BOOT": "0",
        "GMP_TRAIN_ON_BOOT": "0",
    }
)


def synthetic_ohlcv(
    seed: int, n: int = 700, start: str = "2023-01-02", drift: float = 0.0003, vol: float = 0.015
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def _write_universe() -> None:
    from gmp.universe import training_universe

    for i, sym in enumerate(training_universe() + ["SPY"]):
        df = synthetic_ohlcv(seed=i)
        df.index.name = "Date"
        df.to_csv(PRICE_DIR / f"{sym}.csv")


_write_universe()
