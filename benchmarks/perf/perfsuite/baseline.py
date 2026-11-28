"""Timing baselines kept on the host that recorded them, outside the repository.

Timings compare only on one machine, so a timing baseline belongs to its host, not to the
repository: `baseline save` keeps a results file in a per-host store, and `compare` with one file
judges it against that host's baseline for the same tier. The store is `$PERFSUITE_BASELINES`, by
default `~/.nyxus-perf/baselines`, laid out as `<host>/<tier>.json` (host and tier names with
characters outside [a-z0-9._-] %-escaped, so two names never share a file, on file systems that
ignore case too); saving over a baseline keeps the one before as `<host>/prev/<tier>.json`.

The machine-independent measure, instruction counts (see icount), is the one that is committed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .fileio import write_atomic
from .icount import is_counts


def store() -> Path:
    return Path(os.environ.get("PERFSUITE_BASELINES") or Path.home() / ".nyxus-perf" / "baselines")


_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}


def _safe(name: str) -> str:
    """`name` as one path component, reversibly and the same on every file system: every character
    outside [a-z0-9._-] (upper-case letters and the escape character % too) becomes %XX, so two
    names never share a component, even where file names ignore case. A trailing `.`, which Windows
    drops, is escaped; a component that would be empty, `.`, `..`, or a Windows device name (CON,
    NUL, COM1, ...) is escaped whole. (A space is outside the set, so a trailing one is escaped too.)"""
    out = re.sub(r"[^a-z0-9._-]", lambda m: "".join(f"%{b:02X}" for b in m.group(0).encode("utf-8")), name)
    # a trailing dot is dropped by Windows ("build." would open "build"), so it is escaped too
    if out.endswith("."):
        out = out[:-1] + "%2E"
    if out in ("", "%2E", ".%2E") or out.split(".", 1)[0] in _RESERVED:
        out = "".join(f"%{b:02X}" for b in name.encode("utf-8")) or "%"
    return out


def path_for(host: str, tier: str) -> Path:
    return store() / _safe(host) / f"{_safe(tier)}.json"


def prev_path(p: Path) -> Path:
    """Where save() keeps the baseline a newer one replaced: `<host>/prev/<tier>.json`."""
    return p.parent / "prev" / p.name


def problems(meta: dict, cases: list) -> list:
    """Why a results file should not become a baseline; empty when it can."""
    out = []
    if "finished" not in meta:
        out.append("the run did not finish")
    bad = [c["id"] for c in cases if c.get("status") != "ok"]
    if bad:
        out.append(f"{len(bad)} cases are not ok, first {bad[0]}")
    if meta.get("git_dirty"):
        out.append("the checkout had uncommitted changes")
    elif meta.get("git_dirty") is not False:
        out.append("git could not tell whether the checkout had uncommitted changes")
    if not meta.get("host") or not meta.get("tier"):
        out.append("its meta names no host or tier")
    return out


COUNTS_REFERENCE = "baselines/instructions.json"


def _timing_results(results) -> dict:
    """A results file (a path, or its parsed contents); refuses an instruction-count file, whose
    reference is committed."""
    d = results if isinstance(results, dict) else json.loads(Path(results).read_text())
    what = "these results" if isinstance(results, dict) else str(results)
    if not isinstance(d, dict):
        raise ValueError(f"{what}: not a perfsuite results file")
    if is_counts(d):
        raise ValueError(f"{what} hold instruction counts, which are not a host baseline: compare them "
                         f"with the committed {COUNTS_REFERENCE}")
    if not isinstance(d.get("cases"), list) or not isinstance(d.get("meta"), dict):
        raise ValueError(f"{what}: not a perfsuite results file")
    return d


def save(results_path, *, force=False) -> Path:
    """Copy `results_path` to its host's store; refuses a run with problems() unless `force`."""
    d = _timing_results(results_path)
    meta, cases = d["meta"], d["cases"]
    why = problems(meta, cases)
    if why and not force:
        raise ValueError("not saved as a baseline: " + "; ".join(why) + " (--force saves it anyway)")
    dest = path_for(meta.get("host") or "unknown", meta.get("tier") or "unknown")
    if dest.exists():
        write_atomic(prev_path(dest), dest.read_bytes())
    write_atomic(dest, Path(results_path).read_bytes())
    return dest


def find(results) -> tuple:
    """(path, parsed contents) of the host baseline for `results` (a results file's path or its
    parsed contents): the store's file for its host and tier, which must name that host and tier
    itself (a hand-copied file from another machine is refused)."""
    meta = _timing_results(results)["meta"]
    host, tier = meta.get("host") or "unknown", meta.get("tier") or "unknown"
    p = path_for(host, tier)
    if not p.exists():
        raise FileNotFoundError(f"no baseline for host {host!r}, tier {tier!r} at {p}; "
                                "save one with `python -m perfsuite baseline save <results.json>`")
    try:
        data = _timing_results(p)
    except ValueError:
        raise ValueError(f"{p} is not a perfsuite results file; save a baseline over it") from None
    stored = data["meta"]
    # named as save() names them: a run with no host or tier is stored as "unknown"
    if (stored.get("host") or "unknown", stored.get("tier") or "unknown") != (host, tier):
        raise ValueError(f"{p} holds host {stored.get('host')!r}, tier {stored.get('tier')!r}, not "
                         f"{host!r}, {tier!r}: timings compare only on one host")
    return p, data


def listing() -> list:
    """(host, tier, path) of every stored baseline, as the baselines name themselves."""
    root = store()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("*/*.json")):  # prev/ is one level further down
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        m = d.get("meta") if isinstance(d, dict) else None
        if isinstance(m, dict):  # anything else in the store is not a baseline
            out.append((m.get("host", "?"), m.get("tier", "?"), p))
    return out
