"""Forecast cone for the price chart.

The model predicts the 10/50/90th percentile of log returns at 1, 5 and 10
trading days. For the days in between we interpolate: the median linearly in
time, and the band width in sqrt(time) (how uncertainty grows for a random
walk). Dates skip weekends and NYSE holidays.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_DAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def next_sessions(last: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    start = pd.Timestamp(last).normalize() + NYSE_DAY
    return list(pd.date_range(start, periods=n, freq=NYSE_DAY))


def cone(forecast: dict, days: int = 10) -> list[dict]:
    """Daily (q10, q50, q90) log returns for days 1..``days``."""
    hs = np.array([0.0] + [float(k.rstrip("d")) for k in forecast["horizons"]])
    q50 = np.array([0.0] + [v["q50"] for v in forecast["horizons"].values()])
    lo_w = np.array([0.0] + [v["q50"] - v["q10"] for v in forecast["horizons"].values()])
    hi_w = np.array([0.0] + [v["q90"] - v["q50"] for v in forecast["horizons"].values()])
    order = np.argsort(hs)
    hs, q50, lo_w, hi_w = hs[order], q50[order], lo_w[order], hi_w[order]
    out = []
    for d in range(1, days + 1):
        med = float(np.interp(d, hs, q50))
        s, sh = math.sqrt(d), np.sqrt(hs)
        lo = med - float(np.interp(s, sh, lo_w))
        hi = med + float(np.interp(s, sh, hi_w))
        out.append({"day": d, "q10": lo, "q50": med, "q90": hi})
    return out
