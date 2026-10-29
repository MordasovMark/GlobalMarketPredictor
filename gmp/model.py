"""Quantile gradient-boosting forecaster.

For each horizon h in {1, 5, 10} trading days we fit three models that predict
the 10th, 50th and 90th percentile of the *vol-scaled* forward return

    z = log(P[t+h] / P[t]) / (sigma_daily[t] * sqrt(h))

Scaling by trailing volatility makes targets comparable across tickers and
regimes, so one pooled model can learn from ~100 stocks at once. Predictions
are converted back to returns with the same sigma at inference time.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES, HORIZONS
from .settings import settings

log = logging.getLogger(__name__)

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
ARTIFACT_NAME = "forecaster.joblib"
MODEL_CARD_NAME = "model_card.json"
OOS_NAME = "oos_predictions.csv.gz"
ARTIFACT_VERSION = 2


def qcol(q: float, h: int) -> str:
    return f"q{int(round(q * 100))}_{h}d"


def make_regressor(quantile: float, seed: int = 7) -> HistGradientBoostingRegressor:
    # Shallow trees, big leaves, strong L2: daily returns are mostly noise and
    # a flexible model will happily memorise it.
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.03,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=1000,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=seed,
    )


@dataclass
class QuantileForecaster:
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    horizons: tuple[int, ...] = HORIZONS
    quantiles: tuple[float, ...] = QUANTILES
    models: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- training ---------------------------------------------------------
    def fit(self, panel: pd.DataFrame) -> QuantileForecaster:
        for h in self.horizons:
            target = f"target_z_{h}d"
            rows = panel[panel[target].notna()]
            # Winsorise extreme z (earnings gaps, splits gone wrong) so a few
            # rows can't dominate the loss.
            y = rows[target].clip(-6, 6).to_numpy()
            X = rows[self.features].to_numpy(dtype=np.float32)
            for q in self.quantiles:
                model = make_regressor(q)
                model.fit(X, y)
                self.models[qcol(q, h)] = model
        return self

    # -- inference --------------------------------------------------------
    def predict_z(self, panel: pd.DataFrame) -> pd.DataFrame:
        X = panel[self.features].to_numpy(dtype=np.float32)
        out = pd.DataFrame(index=panel.index)
        for h in self.horizons:
            cols = [qcol(q, h) for q in self.quantiles]
            preds = np.column_stack([self.models[c].predict(X) for c in cols])
            preds.sort(axis=1)  # no quantile crossing
            for i, c in enumerate(cols):
                out[c] = preds[:, i]
        return out

    def predict_returns(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Quantiles of forward *log* returns, plus the z-scores."""
        z = self.predict_z(panel)
        sigma = panel["daily_vol"].to_numpy()
        out = pd.DataFrame(index=panel.index)
        for h in self.horizons:
            scale = sigma * np.sqrt(h)
            for q in self.quantiles:
                c = qcol(q, h)
                out[c] = z[c].to_numpy() * scale
                out[f"z_{c}"] = z[c].to_numpy()
        return out

    # -- persistence ------------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ARTIFACT_NAME
        payload = {
            "version": ARTIFACT_VERSION,
            "features": self.features,
            "horizons": list(self.horizons),
            "quantiles": list(self.quantiles),
            "models": self.models,
            "meta": self.meta,
        }
        joblib.dump(payload, path, compress=3)
        (directory / MODEL_CARD_NAME).write_text(json.dumps(self.meta, indent=2, default=str) + "\n")
        return path

    @classmethod
    def load(cls, directory: Path) -> QuantileForecaster:
        payload = joblib.load(directory / ARTIFACT_NAME)
        if payload.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported artifact version {payload.get('version')}")
        missing = [
            qcol(q, h) for h in payload["horizons"] for q in payload["quantiles"] if qcol(q, h) not in payload["models"]
        ]
        if missing:
            raise ValueError(f"Artifact is missing models: {missing}")
        return cls(
            features=payload["features"],
            horizons=tuple(payload["horizons"]),
            quantiles=tuple(payload["quantiles"]),
            models=payload["models"],
            meta=payload.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Registry: one loaded model per process
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._model: QuantileForecaster | None = None
        self._oos: pd.DataFrame | None = None
        self._lock = threading.Lock()
        self.status = "not_loaded"  # not_loaded | ready | missing | training | error
        self.error: str | None = None

    def get(self) -> QuantileForecaster | None:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None and self.status not in ("training",):
                self._load_locked()
        return self._model

    def _load_locked(self) -> None:
        if not (self.directory / ARTIFACT_NAME).is_file():
            self.status = "missing"
            return
        try:
            self._model = QuantileForecaster.load(self.directory)
            self._oos = None
            self.status = "ready"
            self.error = None
        except Exception as exc:  # noqa: BLE001
            log.exception("Could not load model artifact")
            self.status = "error"
            self.error = str(exc)

    def reload(self) -> None:
        with self._lock:
            self._model = None
            self._load_locked()

    def oos_predictions(self) -> pd.DataFrame:
        """Walk-forward out-of-sample 5D predictions saved at training time."""
        if self._oos is None:
            path = self.directory / OOS_NAME
            if path.is_file():
                df = pd.read_csv(path, parse_dates=["date"])
                self._oos = df.set_index(["date", "ticker"]).sort_index()
            else:
                self._oos = pd.DataFrame()
        return self._oos


registry = ModelRegistry(settings.artifact_dir)
