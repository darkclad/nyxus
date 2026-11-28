"""Compare two results files and flag slowdowns and memory growth.

A case regresses when all three hold:
    new median > old median * (1 + tol)              relative: --tol, default 0.15
    new median - old median > k * max(old MAD, new MAD)   beyond the noise: --k, default 3
    new median - old median > floor                  absolute: --floor seconds, default 0.02;
                                                     for the kernels suite, which times a single
                                                     ROI with no I/O and little noise,
                                                     --kernel-floor, default 0.0005
Memory regresses when the peak RSS grows by more than --mem-tol (default 0.20) and 16 MB. It is
judged only for a case that was the first in its worker process in both files (`fresh_worker`),
which every case is unless it ran batched (the per-feature suite). A batched case starts from
whatever the cases before it left resident, so neither its peak nor its growth over its starting
RSS (`delta_rss_mb`) is its own: both are reported, not judged.

A case that was `ok` and is now `error`, `crashed` or `timeout` is always a regression. A kernel
whose reported value changed is listed separately: a timing change that comes with a value change
is a behaviour change, not only a speed change (NaN against NaN is no change). Cases in only one
file are listed, not judged, except a case only in the new file that is not `ok`: that is newly
failing too. It is how a run that died part-way shows, for example the `kernels/RUN` row of a
crashed or timed-out kernels run, whose unfinished rows are then only in the old file.

A new file whose run did not finish (no `finished` in its meta) is incomplete, and so is one that
lacks cases the old file has as `ok`; both make `compare` exit 1. Comparing a narrower run (a
`--filter` or `--suite` subset) against a full baseline is what `allow_missing` is for.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

KERNEL_FLOOR = 0.0005  # the kernels suite times one ROI with no I/O: its floor is its own


def load(results):
    """(meta, {id: case}) of a results file, given as a path or as its parsed contents."""
    d = results if isinstance(results, dict) else json.loads(Path(results).read_text())
    return d["meta"], {c["id"]: c for c in d["cases"]}


def same_value(a, b) -> bool:
    """Whether a kernel reported the same value: equal, or NaN both times."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _peak_rss(a, b):
    """Both cases' peak RSS in MB when it can be judged: each was the first case in its worker."""
    if not (a.get("fresh_worker") and b.get("fresh_worker")):
        return None, None
    return a.get("peak_rss_mb"), b.get("peak_rss_mb")


def compare(old_path, new_path, *, tol=0.15, k=3.0, floor=0.02, kernel_floor=KERNEL_FLOOR, mem_tol=0.20,
            allow_missing=False):
    """Judge `new_path` against `old_path`, each a results file's path or its parsed contents."""
    om, old = load(old_path)
    nm, new = load(new_path)
    rows, regressions, improvements, broken, value_changes = [], [], [], [], []
    for cid in sorted(set(old) & set(new)):
        a, b = old[cid], new[cid]
        if a["status"] == "ok" and b["status"] != "ok":
            broken.append((cid, b["status"], b.get("error", "")))
            continue
        if a["status"] != "ok" or b["status"] != "ok":
            continue
        if "value" in a and "value" in b and not same_value(a["value"], b["value"]):
            value_changes.append((cid, a["value"], b["value"]))
        ta, tb = a["time"].get("median"), b["time"].get("median")
        if ta is None or tb is None:
            continue
        fl = kernel_floor if a.get("suite") == "kernels" else floor
        noise = k * max(a["time"].get("mad", 0.0), b["time"].get("mad", 0.0))
        ratio = tb / ta if ta > 0 else float("inf")
        d = tb - ta
        row = {"id": cid, "old": ta, "new": tb, "ratio": ratio}
        if ratio > 1 + tol and d > noise and d > fl:
            regressions.append(row)
        elif ratio < 1 / (1 + tol) and -d > noise and -d > fl:
            improvements.append(row)
        ma, mb = _peak_rss(a, b)
        if ma and mb and mb - ma > 16 and mb > ma * (1 + mem_tol):  # a peak RSS is never 0 for a live process
            regressions.append({"id": cid + " [peak RSS MB]", "old": ma, "new": mb, "ratio": mb / ma})
        rows.append(row)
    for cid in sorted(set(new) - set(old)):
        if new[cid]["status"] != "ok":
            broken.append((cid, new[cid]["status"], new[cid].get("error", "")))
    missing = [] if allow_missing else sorted(c for c in set(old) - set(new) if old[c]["status"] == "ok")
    return {
        "old_meta": om, "new_meta": nm, "compared": len(rows),
        "incomplete": "finished" not in nm, "missing": missing,
        "regressions": sorted(regressions, key=lambda r: -r["ratio"]),
        "improvements": sorted(improvements, key=lambda r: r["ratio"]),
        "broken": broken, "value_changes": value_changes,
        "only_old": sorted(set(old) - set(new)), "only_new": sorted(set(new) - set(old)),
    }


def failed(r) -> bool:
    """Whether compare()'s result fails the new run; kept next to the keys compare() produces."""
    return bool(r["regressions"] or r["broken"] or r["value_changes"] or r["incomplete"] or r["missing"])


def format_report(r) -> str:
    lines = []
    om, nm = r["old_meta"], r["new_meta"]
    lines.append(f"old: {om.get('git_sha', '?')} on {om.get('host', '?')}   new: {nm.get('git_sha', '?')} on {nm.get('host', '?')}")
    if om.get("host") != nm.get("host"):
        lines.append("WARNING: different hosts; timings are not comparable")
    lines.append(f"compared {r['compared']} cases")
    for title, key in (("REGRESSIONS", "regressions"), ("improvements", "improvements")):
        lines.append(f"\n{title}: {len(r[key])}")
        for x in r[key][:60]:
            lines.append(f"  {x['ratio']:6.2f}x  {x['old']:10.4f} -> {x['new']:10.4f}  {x['id']}")
    lines.append(f"\nnewly failing: {len(r['broken'])}")
    for cid, st, err in r["broken"]:
        lines.append(f"  {st:8s} {cid}  {err}")
    if r.get("value_changes"):
        lines.append(f"\nkernel values changed: {len(r['value_changes'])}")
        for cid, va, vb in r["value_changes"]:
            lines.append(f"  {cid}: {va!r} -> {vb!r}")
    if r.get("incomplete"):
        lines.append("\nINCOMPLETE: the new run did not finish (no 'finished' in its meta)")
    if r.get("missing"):
        lines.append(f"\nMISSING: {len(r['missing'])} cases ok in old are not in new "
                     "(pass --allow-missing when comparing a subset)")
        for cid in r["missing"][:40]:
            lines.append(f"  {cid}")
    if r["only_old"] or r["only_new"]:
        lines.append(f"\nonly in old: {len(r['only_old'])}, only in new: {len(r['only_new'])}")
    return "\n".join(lines)
