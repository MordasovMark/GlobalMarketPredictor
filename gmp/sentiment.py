"""Headline sentiment.

Two backends:

* ``lexicon`` (default): a finance word list in the spirit of Loughran &
  McDonald (2011), with phrase matches and simple negation. Zero extra
  dependencies, fast enough to score hundreds of headlines per request.
* ``finbert``: ProsusAI/finbert via ``transformers`` (install
  ``requirements-nlp.txt`` and set ``GMP_SENTIMENT_BACKEND=finbert``). Falls
  back to the lexicon if the model can't load.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from .settings import settings

log = logging.getLogger(__name__)

POSITIVE = {
    "beat",
    "beats",
    "beating",
    "surge",
    "surges",
    "surged",
    "soar",
    "soars",
    "soared",
    "rally",
    "rallies",
    "rallied",
    "jump",
    "jumps",
    "jumped",
    "gain",
    "gains",
    "gained",
    "rise",
    "rises",
    "rising",
    "rose",
    "record",
    "growth",
    "grows",
    "grew",
    "strong",
    "stronger",
    "strength",
    "upgrade",
    "upgrades",
    "upgraded",
    "outperform",
    "outperforms",
    "bullish",
    "boost",
    "boosts",
    "boosted",
    "raise",
    "raises",
    "raised",
    "exceed",
    "exceeds",
    "exceeded",
    "profit",
    "profitable",
    "profits",
    "win",
    "wins",
    "won",
    "approval",
    "approved",
    "approves",
    "breakthrough",
    "expands",
    "expansion",
    "accelerate",
    "accelerates",
    "momentum",
    "optimistic",
    "optimism",
    "rebound",
    "rebounds",
    "recovery",
    "recovers",
    "highs",
    "buyback",
    "dividend",
    "partnership",
    "launches",
    "innovative",
    "tops",
    "topped",
    "upbeat",
    "robust",
    "resilient",
    "climbs",
    "climbed",
    "advance",
    "advances",
    "positive",
    "favorable",
    "attractive",
    "opportunity",
    "lifts",
    "lifted",
}
NEGATIVE = {
    "miss",
    "misses",
    "missed",
    "fall",
    "falls",
    "fell",
    "falling",
    "drop",
    "drops",
    "dropped",
    "plunge",
    "plunges",
    "plunged",
    "slump",
    "slumps",
    "slumped",
    "tumble",
    "tumbles",
    "tumbled",
    "sink",
    "sinks",
    "sank",
    "loss",
    "losses",
    "lose",
    "loses",
    "weak",
    "weaker",
    "weakness",
    "downgrade",
    "downgrades",
    "downgraded",
    "underperform",
    "bearish",
    "cut",
    "cuts",
    "cutting",
    "lawsuit",
    "lawsuits",
    "sue",
    "sued",
    "probe",
    "investigation",
    "fraud",
    "fine",
    "fined",
    "penalty",
    "recall",
    "recalls",
    "warning",
    "warns",
    "warned",
    "risk",
    "risks",
    "concern",
    "concerns",
    "fear",
    "fears",
    "decline",
    "declines",
    "declined",
    "slowdown",
    "slows",
    "layoffs",
    "layoff",
    "bankruptcy",
    "default",
    "crash",
    "crashes",
    "selloff",
    "sell-off",
    "volatile",
    "volatility",
    "lowers",
    "lowered",
    "disappoint",
    "disappoints",
    "disappointing",
    "halt",
    "halts",
    "delay",
    "delays",
    "delayed",
    "shortage",
    "tariff",
    "tariffs",
    "antitrust",
    "ban",
    "bans",
    "slides",
    "slid",
    "lows",
    "pressure",
    "pressures",
    "headwind",
    "headwinds",
    "struggles",
    "sues",
}
POSITIVE_PHRASES = ("all-time high", "price target raised", "raises guidance", "beats estimates", "record high")
NEGATIVE_PHRASES = ("cuts guidance", "lowers guidance", "price target cut", "misses estimates", "52-week low")
NEGATORS = {"not", "no", "never", "without", "fails", "failed", "despite"}

_WORD = re.compile(r"[a-z][a-z\-']*")


@dataclass(frozen=True)
class Sentiment:
    label: str  # positive | negative | neutral
    confidence: float  # 0..1
    net: float  # -1..1


def lexicon_score(text: str) -> Sentiment:
    lowered = (text or "").lower()
    words = _WORD.findall(lowered)
    pos = neg = 0.0
    for i, w in enumerate(words):
        polarity = 1 if w in POSITIVE else -1 if w in NEGATIVE else 0
        if not polarity:
            continue
        if any(n in NEGATORS for n in words[max(0, i - 3) : i]):
            polarity = -polarity
        if polarity > 0:
            pos += 1
        else:
            neg += 1
    pos += 2 * sum(p in lowered for p in POSITIVE_PHRASES)
    neg += 2 * sum(p in lowered for p in NEGATIVE_PHRASES)
    total = pos + neg
    if total == 0:
        return Sentiment("neutral", 0.5, 0.0)
    net = (pos - neg) / total
    confidence = min(0.95, 0.5 + 0.12 * total) * abs(net) if net else 0.5
    if net > 0.2:
        return Sentiment("positive", round(max(confidence, 0.5), 3), net)
    if net < -0.2:
        return Sentiment("negative", round(max(confidence, 0.5), 3), net)
    return Sentiment("neutral", 0.5, net)


class _FinBert:
    def __init__(self) -> None:
        self._pipe = None
        self._failed = False
        self._lock = threading.Lock()

    def get(self):
        if self._pipe is not None or self._failed:
            return self._pipe
        with self._lock:
            if self._pipe is None and not self._failed:
                try:
                    from transformers import pipeline

                    self._pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                except Exception as exc:  # noqa: BLE001
                    log.warning("FinBERT unavailable, using lexicon: %s", exc)
                    self._failed = True
        return self._pipe


_finbert = _FinBert()


def backend_name() -> str:
    if settings.sentiment_backend == "finbert" and _finbert.get() is not None:
        return "finbert"
    return "lexicon"


def score_texts(texts: list[str]) -> list[Sentiment]:
    if not texts:
        return []
    if settings.sentiment_backend == "finbert":
        pipe = _finbert.get()
        if pipe is not None:
            try:
                results = pipe(texts, truncation=True)
                out = []
                for r in results:
                    label = str(r.get("label", "neutral")).lower()
                    conf = float(r.get("score", 0.5))
                    net = conf if label == "positive" else -conf if label == "negative" else 0.0
                    out.append(Sentiment(label if label in {"positive", "negative"} else "neutral", conf, net))
                return out
            except Exception as exc:  # noqa: BLE001
                log.warning("FinBERT scoring failed, using lexicon: %s", exc)
    return [lexicon_score(t) for t in texts]
