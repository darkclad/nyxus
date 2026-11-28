"""The feature list the per-feature suite walks, from the vetting registry.

`tests/vetting/oracle_coverage.csv` has one row per assertion (feature code x configuration x
oracle), each with the code's dimensionality (2D, 3D, IMQ) and family, so a code appears once per
oracle row; the suite takes each (dimensionality, code) once (presets.feature_cases). A code the
backend no longer accepts shows up as an `error` case, which is how the suite notices the registry
and the backend disagreeing.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "tests" / "vetting" / "oracle_coverage.csv"


def load(path: Path = REGISTRY):
    """Rows as dicts with keys dim, feature, family, in file order."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{"dim": r["dim"], "feature": r["feature"], "family": r["family"]} for r in csv.DictReader(f)]
