"""Walk-forward training and evaluation.

Run it::

    python -m gmp.training                    # 10y of Yahoo data, full universe
    GMP_PRICE_CSV_DIR=data/prices python -m gmp.training --min-train-days 252 --step-days 42

Protocol
--------
* Expanding window. The model for test block *k* is trained only on dates
  before it.
* Purge gap of ``max(horizons)`` days between train and test, because a
  10-day label on the last train day overlaps the first test days.
* Metrics are reported only on these out-of-sample predictions, next to naive
  baselines. The final shipped model is then refit on all labelled data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import __version__
from .evaluation import horizon_metrics, pinball, quintile_backtest
from .features import FEATURES, HORIZONS, build_panel
from .market_data import get_benchmark, get_histories
from .model import OOS_NAME, QUANTILES, QuantileForecaster, qcol
from .settings import settings
from .universe import training_universe

log = logging.getLogger(__name__)

REQUIRED_FEATURES = ["ret_63d", "vol_21d", "dist_sma50", "beta_63"]


def load_panel(period: str = "10y", tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or training_universe()
    histories = get_histories(tickers, period)
    if len(histories) < 5:
        raise RuntimeError(f"Only {len(histories)} tickers downloaded; need at least 5")
    bench = get_benchmark(period)["Close"]
    panel = build_panel(histories, bench)
    # Drop warm-up rows where the slow features are not defined yet.
    return panel.dropna(subset=REQUIRED_FEATURES)


def walk_forward(
    panel: pd.DataFrame,
    min_train_days: int,
    step_days: int,
) -> tuple[pd.DataFrame, list[dict], QuantileForecaster | None, pd.DataFrame | None]:
    dates = panel.index.get_level_values("date").unique().sort_values()
    purge = max(HORIZONS)
    if len(dates) < min_train_days + step_days:
        raise RuntimeError(f"Not enough history: {len(dates)} dates, need {min_train_days + step_days}")
    all_dates = panel.index.get_level_values("date")
    preds, folds = [], []
    last_model, last_test = None, None
    for start in range(min_train_days, len(dates), step_days):
        test_dates = dates[start : start + step_days]
        train_end = dates[start - purge - 1]
        train = panel[all_dates <= train_end]
        test = panel[all_dates.isin(test_dates)]
        if test.empty:
            continue
        t0 = time.time()
        model = QuantileForecaster().fit(train)
        preds.append(model.predict_returns(test))
        folds.append(
            {
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_dates[0].strftime("%Y-%m-%d"),
                "test_end": test_dates[-1].strftime("%Y-%m-%d"),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "seconds": round(time.time() - t0, 1),
            }
        )
        log.info(
            "Fold %d: train<=%s test %s..%s",
            len(folds),
            folds[-1]["train_end"],
            folds[-1]["test_start"],
            folds[-1]["test_end"],
        )
        last_model, last_test = model, test
    return pd.concat(preds).sort_index(), folds, last_model, last_test


def feature_importance(model: QuantileForecaster, test: pd.DataFrame, h: int = 5) -> list[dict]:
    """Permutation importance of the median model on its own out-of-sample block."""
    rows = test[test[f"target_z_{h}d"].notna()]
    if len(rows) > 20_000:
        rows = rows.sample(20_000, random_state=0)
    X = rows[model.features].to_numpy(dtype=np.float32)
    y = rows[f"target_z_{h}d"].clip(-6, 6).to_numpy()
    est = model.models[qcol(0.5, h)]

    def scorer(estimator, X_, y_):
        return -pinball(y_, estimator.predict(X_), 0.5)

    result = permutation_importance(est, X, y, scoring=scorer, n_repeats=3, random_state=0)
    order = np.argsort(result.importances_mean)[::-1]
    return [{"feature": model.features[i], "importance": round(float(result.importances_mean[i]), 6)} for i in order]


def train(
    period: str = "10y",
    min_train_days: int = 756,
    step_days: int = 63,
    out_dir: Path | None = None,
    tickers: list[str] | None = None,
    quick: bool = False,
) -> dict:
    """Train, evaluate and save. ``quick`` skips walk-forward (used on first boot)."""
    out_dir = out_dir or settings.artifact_dir
    started = time.time()
    panel = load_panel(period, tickers)
    dates = panel.index.get_level_values("date")
    log.info(
        "Panel: %d rows, %d tickers, %s..%s",
        len(panel),
        panel.index.get_level_values("ticker").nunique(),
        dates.min().date(),
        dates.max().date(),
    )

    oos, folds, metrics, backtest, importance = None, [], {}, {}, []
    if not quick:
        oos, folds, last_model, last_test = walk_forward(panel, min_train_days, step_days)
        first_test = pd.Timestamp(folds[0]["test_start"])
        train_z_ref = {h: panel.loc[dates < first_test, f"target_z_{h}d"].dropna().clip(-6, 6) for h in HORIZONS}
        metrics = {f"{h}d": horizon_metrics(oos, panel, h, train_z_ref[h]) for h in HORIZONS}
        backtest = quintile_backtest(oos, panel, h=5)
        importance = feature_importance(last_model, last_test) if last_model is not None else []

    final = QuantileForecaster().fit(panel)
    labelled_end = panel[panel["target_z_10d"].notna()].index.get_level_values("date").max()
    final.meta = {
        "app_version": __version__,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "data_source": f"csv:{settings.price_csv_dir.name}" if settings.price_csv_dir else f"yahoo:{period}",
        "data_start": dates.min().strftime("%Y-%m-%d"),
        "data_end": dates.max().strftime("%Y-%m-%d"),
        "labelled_end": labelled_end.strftime("%Y-%m-%d"),
        "n_rows": int(len(panel)),
        "n_tickers": int(panel.index.get_level_values("ticker").nunique()),
        "features": FEATURES,
        "horizons": list(HORIZONS),
        "quantiles": list(QUANTILES),
        "walk_forward": None
        if quick
        else {
            "min_train_days": min_train_days,
            "step_days": step_days,
            "purge_days": max(HORIZONS),
            "folds": folds,
            "oos_start": folds[0]["test_start"],
            "oos_end": folds[-1]["test_end"],
        },
        "metrics": metrics,
        "backtest_5d_quintiles": backtest,
        "feature_importance_5d": importance,
        "training_seconds": round(time.time() - started, 1),
    }
    final.save(out_dir)
    if oos is None:
        return final.meta

    # Keep the 5-day out-of-sample predictions: the strategy backtests replay
    # them so in-sample fits never leak into the trading simulation.
    cols = [qcol(q, 5) for q in QUANTILES] + [f"z_{qcol(0.5, 5)}"]
    oos[cols].astype("float32").round(5).reset_index().to_csv(out_dir / OOS_NAME, index=False, compression="gzip")
    log.info("Saved model to %s in %.0fs", out_dir, time.time() - started)
    return final.meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GMP quantile forecaster")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="skip walk-forward evaluation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = train(args.period, args.min_train_days, args.step_days, args.out, quick=args.quick)
    if args.quick:
        print("Quick fit saved (no out-of-sample metrics).")
        return
    m5 = meta["metrics"]["5d"]
    print(
        f"5D out-of-sample: hit rate {m5.get('hit_rate')} (always-up {m5.get('always_up_hit_rate')}), "
        f"rank IC {m5.get('rank_ic_mean')} (t={m5.get('rank_ic_tstat')}), "
        f"80% band coverage {m5.get('interval_coverage_80')}"
    )


if __name__ == "__main__":
    main()
