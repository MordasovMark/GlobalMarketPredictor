"""Runtime configuration, read once from environment variables.

Nothing secret lives in the code. Copy `.env.example` to `.env` for local work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't need python-dotenv. Real env vars win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Where trained models and their metrics live.
    artifact_dir: Path = field(default_factory=lambda: Path(os.getenv("GMP_ARTIFACT_DIR", ROOT_DIR / "artifacts")))
    # Optional offline price source: a folder of yfinance-style CSVs (one per ticker).
    # When set, no network calls are made for prices. Used for tests and offline demos.
    price_csv_dir: Path | None = field(
        default_factory=lambda: Path(p) if (p := os.getenv("GMP_PRICE_CSV_DIR")) else None
    )
    # Server-side Finnhub key for real-time quotes. Optional: yfinance is the fallback.
    finnhub_api_key: str = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY", "").strip())
    # "lexicon" (fast, no extra deps) or "finbert" (needs requirements-nlp.txt).
    sentiment_backend: str = field(
        default_factory=lambda: os.getenv("GMP_SENTIMENT_BACKEND", "lexicon").strip().lower()
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _list(
            "GMP_CORS_ORIGINS",
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:4173",
                "https://globalmarketpredictor.onrender.com",
            ],
        )
    )
    # Train a model in the background on boot if no artifact is found.
    train_on_boot: bool = field(default_factory=lambda: _bool("GMP_TRAIN_ON_BOOT", True))
    # Warm the dashboard snapshot (ratings, portfolio) in the background on boot.
    warm_on_boot: bool = field(default_factory=lambda: _bool("GMP_WARM_ON_BOOT", True))
    # Allow outbound network calls (RSS, yfinance). Tests switch this off.
    network_enabled: bool = field(default_factory=lambda: _bool("GMP_NETWORK", True))
    http_timeout_s: float = field(default_factory=lambda: float(os.getenv("GMP_HTTP_TIMEOUT", "8")))


settings = Settings()
