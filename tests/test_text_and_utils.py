import pandas as pd
import pytest

from gmp.forecast import cone, next_sessions
from gmp.news import aggregate_score, is_relevant, parse_feed
from gmp.ratings import consensus
from gmp.sentiment import lexicon_score
from gmp.universe import InvalidTicker, to_display, to_yahoo

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
<item><title>Apple beats estimates as iPhone sales surge - Reuters</title>
<source url="https://reuters.com">Reuters</source>
<pubDate>Mon, 27 Oct 2025 14:00:00 GMT</pubDate><link>https://x/1</link></item>
<item><title>Markets drift ahead of Fed decision</title><pubDate>Mon, 27 Oct 2025 13:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_parse_feed_strips_publisher_suffix():
    items = parse_feed(RSS, "Google News")
    assert items[0]["headline"] == "Apple beats estimates as iPhone sales surge"
    assert items[0]["source"] == "Reuters"
    assert items[1]["source"] == "Google News"


def test_short_tickers_need_cashtag_or_company_name():
    assert not is_relevant("Markets rally as volatility eases", "V")
    assert is_relevant("$V jumps after earnings", "V")
    assert is_relevant("Visa raises guidance", "V")
    assert is_relevant("Home Depot beats", "HD")
    assert is_relevant("NVDA hits record", "NVDA")


def test_lexicon_and_negation():
    assert lexicon_score("Apple beats estimates, shares surge").label == "positive"
    assert lexicon_score("Tesla shares plunge after recall").label == "negative"
    assert lexicon_score("Company did not beat expectations").label != "positive"
    assert lexicon_score("Board meeting scheduled for Tuesday").label == "neutral"


def test_aggregate_score_neutral_without_news_and_shrinks():
    assert aggregate_score([]) == 50.0
    one = [{"_net": 1.0, "_weight": 1.0}]
    many = one * 12
    assert 50 < aggregate_score(one) < aggregate_score(many) <= 100


def test_consensus_requires_two_engines():
    assert consensus({"value": 90, "sentiment": 50, "momentum": 90}) >= 66
    assert consensus({"value": 95, "sentiment": 30, "momentum": 59}) <= 65  # only one bullish
    assert consensus({"value": 10, "sentiment": 55, "momentum": 50}) >= 41  # only one bearish
    assert consensus({"value": None, "sentiment": None, "momentum": None}) == 50.0


def test_ticker_normalisation():
    assert to_yahoo("brk.b") == "BRK-B"
    assert to_display("BRK-B") == "BRK.B"
    assert to_yahoo("TEVA.TA") == "TEVA.TA"
    with pytest.raises(InvalidTicker):
        to_yahoo("AAPL; DROP TABLE")


def test_next_sessions_skip_weekends_and_holidays():
    days = next_sessions(pd.Timestamp("2025-12-24"), 3)  # Christmas is closed
    assert [d.strftime("%Y-%m-%d") for d in days] == ["2025-12-26", "2025-12-29", "2025-12-30"]


def test_cone_widens_and_matches_known_horizons():
    fc = {
        "horizons": {
            "1d": {"q10": -0.02, "q50": 0.001, "q90": 0.02},
            "5d": {"q10": -0.045, "q50": 0.004, "q90": 0.05},
            "10d": {"q10": -0.06, "q50": 0.008, "q90": 0.07},
        }
    }
    path = cone(fc, 10)
    widths = [p["q90"] - p["q10"] for p in path]
    assert all(b >= a for a, b in zip(widths, widths[1:]))
    assert path[4]["q50"] == pytest.approx(0.004)
    assert path[9]["q10"] == pytest.approx(-0.06)
