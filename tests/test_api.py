"""End-to-end API tests on synthetic offline data with a freshly trained model."""

import pytest
from fastapi.testclient import TestClient

from gmp.model import registry
from gmp.settings import settings
from gmp.training import train


@pytest.fixture(scope="module")
def client():
    meta = train(period="2y", min_train_days=300, step_days=100, out_dir=settings.artifact_dir)
    assert meta["metrics"]["5d"]["n"] > 0
    registry.reload()
    from gmp import service
    from gmp.app import app

    service.snapshot.refresh()
    with TestClient(app) as c:
        yield c


def test_health_reports_ready_model(client):
    body = client.get("/health").json()
    assert body["model"]["status"] == "ready"
    assert set(body["snapshot"]["ready"]) >= {"ratings", "universe", "portfolio"}


def test_model_card_has_out_of_sample_metrics(client):
    card = client.get("/api/model").json()
    m5 = card["metrics"]["5d"]
    assert 0 <= m5["hit_rate"] <= 1
    assert card["walk_forward"]["purge_days"] == 10


def test_analyze_returns_history_and_forecast_cone(client):
    body = client.get("/api/analyze", params={"ticker": "AAPL", "time_range": "1mo"}).json()
    future = [p for p in body["chart_data"] if p["price"] is None]
    assert body["forecast_available"] and len(future) == 10
    for p in future:
        assert p["predictedLow"] <= p["predictedPrice"] <= p["predictedHigh"]
    assert body["signal"] in {"BUY", "SELL", "HOLD"}
    assert body["forecast_summary"]["bear"] <= body["forecast_summary"]["bull"]


def test_simulate_uses_out_of_sample_signals(client):
    body = client.get("/api/portfolio/simulate", params={"ticker": "MSFT", "time_range": "6mo"}).json()
    assert body["initial_balance"] == 10000
    assert "buy_hold_roi_pct" in body


def test_ratings_and_detail(client):
    ratings = client.get("/api/ratings").json()["ratings"]
    assert len(ratings) == 20
    assert all(0 <= r["score"] <= 100 for r in ratings)
    detail = client.get("/api/ratings/BRK.B").json()
    assert {m["name"] for m in detail["models"]} == {"DeepValue Engine", "Sentiment NLP", "Tech-Momentum Net"}
    assert "momentum" in detail["charts"]


def test_strategy_portfolio(client):
    body = client.get("/api/strategy/portfolio").json()
    assert body["capital"] == 100000
    assert body["rules"]["execution"] == "next session open"


def test_input_validation(client):
    assert client.get("/api/quotes", params={"tickers": "AAPL;rm"}).status_code == 400
    assert client.get("/api/news/AAPL", params={"limit": 99}).status_code == 422
    assert client.get("/api/ratings/ZZZZ").status_code == 404


def test_no_secrets_in_frontend_source():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "frontend" / "src"
    text = "\n".join(p.read_text() for p in src.rglob("*.js*"))
    assert "token=" not in text and "VITE_FINNHUB_KEY" not in text
