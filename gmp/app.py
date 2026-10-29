"""FastAPI application. Interactive docs at /docs."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__, service
from .fear_greed import fear_greed
from .market_data import DataUnavailable, get_history
from .news import ticker_news
from .ratings import rating_detail
from .settings import settings
from .universe import InvalidTicker, dashboard_stock, to_display, to_yahoo

log = logging.getLogger("gmp")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if settings.warm_on_boot:
        service.snapshot.start()
    yield
    service.snapshot.stop()


app = FastAPI(
    title="Global Market Predictor API",
    version=__version__,
    description="Quantile ML forecasts, multi-engine stock ratings, news sentiment and backtests.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(InvalidTicker)
async def _invalid_ticker(_: Request, exc: InvalidTicker):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DataUnavailable)
async def _unavailable(_: Request, exc: DataUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "30"})


def _tickers_param(raw: str, limit: int = 40) -> list[str]:
    items = [t.strip() for t in raw.split(",") if t.strip()]
    if not items:
        raise HTTPException(status_code=400, detail="Provide at least one ticker")
    if len(items) > limit:
        raise HTTPException(status_code=400, detail=f"At most {limit} tickers per request")
    return [to_yahoo(t) for t in items]


def _snapshot_or_503(key: str):
    value = service.snapshot.get(key)
    if value is None:
        err = service.snapshot.status()["errors"].get(key)
        raise HTTPException(
            status_code=503,
            detail=f"{key} is still being computed" + (f" (last error: {err})" if err else ""),
            headers={"Retry-After": "20"},
        )
    return value


# ---------------------------------------------------------------------------
# Schemas (for the docs page)
# ---------------------------------------------------------------------------
class MacroItem(BaseModel):
    symbol: str
    price: float | None = None
    change_pct: float | None = None


class NewsItem(BaseModel):
    headline: str
    source: str
    time: str
    sentiment: str
    sentiment_label: str
    sentiment_score: float
    link: str = ""
    published: str | None = None


class NewsResponse(BaseModel):
    ticker: str
    source_ticker: str
    generated_at: str
    is_live: bool
    sentiment_backend: str
    news_sentiment_score: float
    headline_count: int
    thesis: str
    daily_flow: list[dict]
    items: list[NewsItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"name": "Global Market Predictor API", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "model": service.model_summary(),
        "snapshot": service.snapshot.status(),
    }


@app.get("/api/model")
def api_model():
    """Model card: training window, walk-forward metrics, backtest, feature importance."""
    return service.model_card()


@app.get("/api/universe")
def api_universe():
    """The 20 dashboard stocks with live price, 52-week range and fundamentals."""
    return _snapshot_or_503("universe")


@app.get("/api/quotes")
def api_quotes(tickers: str = Query(..., description="Comma-separated, e.g. AAPL,MSFT")):
    return service.quotes(_tickers_param(tickers))


@app.get("/api/market", response_model=list[MacroItem])
def api_market():
    """SPY, QQQ, IWM: latest price and day change."""
    return service.macro()


@app.get("/api/macro", response_model=list[MacroItem])
def api_macro(symbols: str = "SPY,QQQ,IWM"):
    return service.macro([to_display(s) for s in _tickers_param(symbols, limit=10)])


@app.get("/api/price/{ticker}")
def api_price(ticker: str):
    sym = to_yahoo(ticker)
    q = service.quotes([sym]).get(to_display(sym))
    if q is None:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
    return {"ticker": to_display(sym), "price": q["price"], "change_pct": q["changePercent"]}


@app.get("/api/analyze")
def api_analyze(ticker: str, time_range: str = "3mo"):
    """Price history plus the model's forecast cone (10/50/90th percentile)."""
    return service.analyze(ticker, time_range)


@app.get("/api/forecast/{ticker}")
def api_forecast(ticker: str):
    """Forecast ranges for 1, 5 and 10 trading days."""
    data = service.analyze(ticker, "1mo")
    return {
        k: data.get(k)
        for k in ("ticker", "price", "change_pct", "signal", "prob_up_5d", "horizons", "forecast_as_of", "model")
    }


@app.get("/api/history/{ticker}")
def api_history(ticker: str, period: str = Query("1y", pattern="^(1mo|3mo|6mo|1y|2y)$")):
    df = get_history(ticker, "2y")
    days = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731}[period]
    df = df[df.index >= df.index[-1] - pd.Timedelta(days=days)]
    return [{"date": d.strftime("%Y-%m-%d"), "price": round(float(c), 2)} for d, c in df["Close"].items()]


@app.get("/api/news/{ticker}", response_model=NewsResponse)
def api_news(ticker: str, limit: int = Query(4, ge=1, le=12)):
    """Live RSS headlines with sentiment scores."""
    return ticker_news(ticker, limit)


@app.get("/api/fear-greed")
def api_fear_greed():
    """US market Fear & Greed index computed from six market components."""
    return fear_greed()


@app.get("/api/ratings")
def api_ratings():
    """Consensus rating for every dashboard stock."""
    ratings = _snapshot_or_503("ratings")
    return {"as_of": service.snapshot.status()["updated_at"], "ratings": list(ratings.values())}


@app.get("/api/ratings/{ticker}")
def api_rating(ticker: str):
    ratings = _snapshot_or_503("ratings")
    stock = dashboard_stock(ticker)
    if stock is None or stock.ticker not in ratings:
        raise HTTPException(status_code=404, detail=f"{ticker} is not in the rated universe")
    return rating_detail(ratings[stock.ticker])


@app.get("/api/portfolio/simulate")
def api_portfolio_simulate(ticker: str, time_range: str = "3mo"):
    """$10k paper trade of one stock using the model's out-of-sample signals."""
    return service.simulate(ticker, time_range)


@app.get("/api/strategy/portfolio")
def api_strategy_portfolio():
    """Model portfolio: weekly top-5 by forecast across the 20 stocks, last 12 months."""
    return _snapshot_or_503("portfolio")
