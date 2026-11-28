"""Small statistics helpers: robust centre and spread, and a log-log slope for scaling sweeps."""

from __future__ import annotations

import math
from statistics import median


def mad(xs) -> float:
    """Median absolute deviation (unscaled)."""
    if not xs:
        return float("nan")
    m = median(xs)
    return median(abs(x - m) for x in xs)


def summarize(xs) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": median(xs), "mad": mad(xs), "min": min(xs), "max": max(xs)}


def loglog_slope(points):
    """Least-squares slope of log(y) against log(x): the empirical complexity exponent.

    `points` is an iterable of (x, y) with x, y > 0. Returns (slope, r2) or None for fewer than
    two usable points.
    """
    pts = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    syy = sum((p[1] - my) ** 2 for p in pts)
    r2 = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return slope, r2
