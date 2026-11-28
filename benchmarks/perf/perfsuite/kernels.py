"""Kernel-only timings from the `bench_feature_kernels` executable (benchmarks/bench_feature_kernels.cpp).

Built with -DRUN_BENCHMARKS=ON. Each CSV row becomes a result row keyed
`kernels/<kernel>/side<N>`, with a sweep point (x = ROI pixels) so `report` fits each kernel's
complexity exponent.

The executable flushes each row as it finishes, so when it exits non-zero or runs past the
timeout the rows it already printed are kept, and one `kernels/RUN` row records the failure. An
executable that cannot be started, output without the CSV header, output lines that are not kernel
rows (the rows around them are kept), and a run that printed no rows are failures too.

A `select(case_id)` narrows the run to the kernels and sides whose ids it accepts; kernel names come
from the executable's `--help`, and each selected kernel is a call of its own. The timeout is one
budget for all the calls: a call that is still running when it runs out, or a call it ran out
before, is named in the failure as having run out of budget, not as having hung.
"""

from __future__ import annotations

import csv
import math
import subprocess
import time
from pathlib import Path

from .checktiming import find_executable

SIDES = {"smoke": [64, 256], "full": [64, 128, 256, 512, 1024, 2048], "large": [256, 1024, 2048, 4096]}

HEADER = "kernel,side,pixels,reps,median_ms,mad_ms,min_ms,max_ms,value"


def _text(out) -> str:
    if out is None:
        return ""
    return out.decode("utf-8", "replace") if isinstance(out, bytes) else out


def number(text: str) -> float:
    """A float from the CSV, NaN and infinities in any C runtime's spelling (MSVC: `-nan(ind)`, `inf`)."""
    try:
        return float(text)
    except ValueError:
        t = text.strip().lower()
        if "nan" in t:
            return math.nan
        if "inf" in t:
            return -math.inf if t.startswith("-") else math.inf
        raise


def _row(r: dict) -> dict:
    med = float(r["median_ms"]) / 1000
    return {
        "id": f"kernels/{r['kernel']}/side{r['side']}", "suite": "kernels", "api": "cpp",
        "dataset": f"blob-{r['side']}", "container": "-", "features": [r["kernel"]], "params": {},
        "mode": "-", "warmup": 1, "repeats": int(r["reps"]), "status": "ok",
        # the executable reports median/MAD/min/max rather than every repetition
        "time": {"n": int(r["reps"]), "median": med, "mad": float(r["mad_ms"]) / 1000,
                 "min": float(r["min_ms"]) / 1000, "max": float(r["max_ms"]) / 1000},
        "times": [], "value": number(r["value"]),
        "sweep": {"name": f"kernel/{r['kernel']}", "x": int(r["pixels"])},
    }


def parse(stdout: str, bad=None):
    """Result rows from the CSV the executable printed, an incomplete last line dropped; None when
    the output does not start with the CSV header.

    A line that is not a kernel row raises ValueError, or, when `bad` is a list, is appended to it
    and skipped, so the rows around it are kept.
    """
    lines = stdout.splitlines()
    if stdout and not stdout.endswith("\n"):
        lines = lines[:-1]  # cut off mid-row by a kill
    if not lines or lines[0].strip() != HEADER:
        return None
    rows = []
    fields = HEADER.split(",")
    for ln in lines[1:]:
        if not ln.strip():
            continue
        cells = next(csv.reader([ln]))
        try:
            if len(cells) != len(fields):
                raise ValueError(f"{len(cells)} fields")
            rows.append(_row(dict(zip(fields, cells))))
        except ValueError as e:
            if bad is None:
                raise ValueError(f"not a kernel row: {ln!r} ({e})") from None
            bad.append(ln)
    return rows


def names(exe) -> dict:
    """{kernel: largest side} as the executable lists them under `Kernels:` in its --help."""
    p = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=60)
    lines = p.stdout.splitlines()
    if "Kernels:" not in lines:
        raise RuntimeError(f"{exe}: --help lists no kernels; is this bench_feature_kernels?")
    out = {}
    for ln in lines[lines.index("Kernels:") + 1:]:
        if ln.strip():
            try:
                name, max_side = ln.split()
                out[name] = int(max_side)
            except ValueError:
                raise RuntimeError(f"{exe}: unexpected --help line {ln!r}") from None
    return out


def plan(exe, tier: str = None, select=None, sides=None) -> list:
    """The (kernel, side) pairs a run covers: `sides` (default the tier's) up to each kernel's
    largest, narrowed by `select(case_id)`."""
    sides = SIDES[tier] if sides is None else sides
    return [(k, s) for k, top in names(exe).items() for s in sides
            if s <= top and (select is None or select(f"kernels/{k}/side{s}"))]


def _failure(status, error, n):
    return {"id": "kernels/RUN", "suite": "kernels", "status": status,
            "error": f"{error}; {n} rows finished before it", "time": {"n": 0}, "times": []}


def _invoke(args, timeout, timed_out=None):
    """One run of the executable: (rows, failure text or None, status). `timed_out` is the failure
    text of a timeout."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, failure = p.stdout, p.stderr, (f"exit {p.returncode}" if p.returncode != 0 else None)
        status = "error"
    except subprocess.TimeoutExpired as e:
        stdout, stderr = _text(e.stdout), _text(e.stderr)
        failure = timed_out or f"timeout after {timeout:.0f} s"
        status = "timeout"
    except OSError as e:  # exists but cannot be started: not an executable, no permission, ...
        return [], f"cannot start {args[0]}: {e}", "error"
    bad = []
    rows = parse(stdout, bad)
    if rows is None and not failure:
        failure = "output has no kernel CSV header; is this bench_feature_kernels?"
    if bad and not failure:
        failure = f"{len(bad)} output lines are not kernel rows, first {bad[0][:200]!r}"
    if failure and stderr.strip():
        failure += f": {stderr.strip()[:300]}"
    return rows or [], failure, status


def run(exe: Path, tier: str, *, timeout=3600, select=None, log=print):
    """Rows of one kernels run; `timeout` holds for the whole run.

    Unfiltered, the executable runs once over the tier's sides. With `select`, it runs once per
    selected kernel, at that kernel's selected sides only.
    """
    exe = find_executable(exe, "bench_feature_kernels")
    if select is None:
        calls = [[str(exe), *[a for s in SIDES[tier] for a in ("--side", str(s))]]]
    else:
        try:
            pairs = plan(exe, tier, select)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
            log(f"kernels: {exe.name} --help: {e}")
            return [_failure("error", f"{exe.name} --help: {e}", 0)]
        if not pairs:
            log("kernels: --filter selects none")
            return []
        by = {}
        for k, s in pairs:
            by.setdefault(k, []).append(s)
        calls = [[str(exe), "--kernel", k, *[a for s in ss for a in ("--side", str(s))]] for k, ss in by.items()]
    deadline = time.monotonic() + timeout
    rows = []
    for i, args in enumerate(calls):
        what = f"--kernel {args[2]} " if select is not None else ""
        left = deadline - time.monotonic()
        if left <= 0:
            failure = f"the {timeout:.0f} s kernels budget ran out before {what}started"
            log(f"kernels: {failure}")
            rows.append(_failure("timeout", failure, len(rows)))
            break
        # the first call has the whole budget; a later one is told it ran out, not that it hung
        timed_out = (f"timeout after {timeout:.0f} s" if i == 0 else
                     f"the {timeout:.0f} s kernels budget ran out during it ({left:.1f} s were left)")
        got, failure, status = _invoke(args, left, timed_out)
        if select is not None:
            got = [r for r in got if select(r["id"])]
        rows += got
        if not got and not failure:
            failure = "printed no kernel rows"
        if failure:
            log(f"kernels: {exe.name} {what}{failure}")
            rows.append(_failure(status, what + failure, len(rows)))
            break
    log(f"kernels: {len(rows)} rows")
    return rows
