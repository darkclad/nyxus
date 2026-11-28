"""Run cases in worker processes and write one results file.

Every case runs in a fresh worker process by default, so a crash, a leak or a process-global
setting (`ram_limit` is one) cannot reach the next case. `--batch N` puts up to N cases of the same
dataset in one worker, which the per-feature suite needs: it has over a thousand short cases and a
process start costs more than most of them. A worker that dies mid-batch is restarted after the
case it was running, which is recorded as `crashed`. Each result records `fresh_worker`, whether
the case was the first in its worker: only then is its peak RSS its own.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import datasets, nyxenv
from .fileio import write_atomic
from .stats import summarize
from .worker import HARNESS_FAILURE

HERE = Path(__file__).resolve().parent


class HarnessFailure(RuntimeError):
    """The suite itself cannot run here (as opposed to a case that failed). `results` holds the
    cases measured before it happened."""

    def __init__(self, message, results=()):
        super().__init__(message)
        self.results = list(results)


def resolve_paths(case: dict, data_root: Path) -> dict:
    ds = datasets.Dataset.parse(case["dataset"])
    ip, sp = datasets.file_paths(data_root, ds, case["container"])
    c = dict(case)
    c["params"] = dict(case["params"])  # the caller's case keeps its own params
    if case["container"] == "numpy":
        c["int_file"], c["seg_file"] = str(ip), (str(sp) if sp else None)
    else:
        c["int_dir"], c["seg_dir"] = str(ip.parent), (str(sp.parent) if sp else None)
    extra = datasets.intensity_params(ds)
    if case["api"] == "imq":
        extra.pop("dynamic_range", None)
    for k, v in extra.items():
        c["params"].setdefault(k, v)
    return c


def git_sha(path: Path):
    """The full SHA of the commit the checkout at `path` is on; None when git cannot tell."""
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
    except OSError:
        return None
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None


def git_dirty(path: Path):
    """Whether the checkout at `path` differs from its commit: changed files, or new files git does
    not ignore (an uncommitted source file changes the build as much as an edited one). None when
    git cannot tell (not a checkout, git missing, or a repository git refuses to read): unknown is
    not clean."""
    try:
        p = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=path,
                           capture_output=True, text=True)
    except OSError:
        return None
    if p.returncode != 0:
        return None
    return bool(p.stdout.strip())


def machine_info() -> dict:
    info = {
        "host": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil

        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        if sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        nyxenv.prepare()
        import nyxus

        info["nyxus_version"] = getattr(nyxus, "__version__", "?")
        info["nyxus_module"] = str(Path(nyxus.__file__).parent)
        info.update(nyxenv.gpu_info())
    except Exception as e:  # noqa: BLE001
        info["nyxus_error"] = str(e)
    # recorded as instruction counts record them (the same helpers): None for what git cannot tell
    info["git_sha"] = git_sha(HERE)
    info["git_dirty"] = git_dirty(HERE)
    return info


def _batches(cases, batch):
    cur = []
    for c in cases:
        if cur and (len(cur) >= batch or cur[-1]["dataset"] != c["dataset"]):
            yield cur
            cur = []
        cur.append(c)
    if cur:
        yield cur


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


WORKER = [sys.executable, "-m", "perfsuite.worker"]


def _run_worker(batch, timeout, log_dir: Path, poll=0.5, worker=None):
    """Run `batch` in one worker; returns {id: result} for the cases that finished.

    `timeout` holds for each case: the worker is killed when no case has finished for that
    long, so one hung case costs `timeout` seconds whatever the batch size. `worker` replaces
    the worker command line (the cases and results paths are appended), for tests.
    """
    with tempfile.TemporaryDirectory(prefix="perfsuite-") as td:
        cf = Path(td) / "cases.json"
        rf = Path(td) / "results.jsonl"
        cf.write_text(json.dumps(batch))
        # a name of its own: two workers started within one millisecond must not share a log
        log = log_dir / f"worker-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.log"
        env = nyxenv.child_env()
        t0 = time.time()
        with open(log, "xb") as lf:
            p = subprocess.Popen([*(worker or WORKER), str(cf), str(rf)],
                                 stdout=lf, stderr=subprocess.STDOUT, env=env)
            seen, since = _size(rf), time.monotonic()
            while True:
                try:
                    rc = p.wait(timeout=poll)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now_size = _size(rf)
                if now_size != seen:  # a case finished: its budget starts over
                    seen, since = now_size, time.monotonic()
                elif time.monotonic() - since > timeout:
                    p.kill()
                    p.wait()
                    rc = "timeout"
                    break
        done = {}
        if rf.exists():
            for line in rf.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # cut off by a kill mid-write: that case did not finish
                done[r["id"]] = r
        return done, rc, log, time.time() - t0


def log_dir_of(out: Path) -> Path:
    """Where a run's logs go: `<results stem>-logs` next to the results file."""
    out = Path(out)
    return out.parent / (out.stem + "-logs")


def _log(msg):
    print(msg, flush=True)  # progress must show up even when stdout is redirected to a file


def run(cases, data_root: Path, out_path: Path, *, batch=1, timeout=3600, tier="?", log=_log, info=None,
        log_dir=None):
    """Run `cases`, saving `out_path` after each; `info` is a machine_info() the caller already has.
    Worker logs go to `log_dir`, by default log_dir_of(out_path). Raises HarnessFailure when a
    worker reports that it cannot measure."""
    data_root = Path(data_root)
    out_path = Path(out_path)
    log_dir = Path(log_dir) if log_dir is not None else log_dir_of(out_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(info) if info is not None else machine_info()
    meta.update({"tier": tier, "started": _dt.datetime.now().isoformat(timespec="seconds"),
                 "data_root": str(data_root)})
    results = []
    resolved = [resolve_paths(c, data_root) for c in cases]
    total, n = len(resolved), 0  # cases are taken in order: the first n are done
    saved_at = time.monotonic()
    while n < total:
        b = next(_batches(resolved[n:n + batch], batch))
        done, rc, wlog, el = _run_worker(b, timeout, log_dir)
        if rc == HARNESS_FAILURE:
            # the cases this worker finished before it failed are measured: keep them
            results += [_finish(c, done[c["id"]], fresh_worker=(i == 0)) for i, c in enumerate(b) if c["id"] in done]
            _save(out_path, meta, results)
            raise HarnessFailure(f"the worker cannot measure (exit {rc}); see {wlog}", results)
        for i, c in enumerate(b):
            n += 1
            if c["id"] in done:
                results.append(_finish(c, done[c["id"]], fresh_worker=(i == 0)))
                continue
            # first unfinished case of the batch is the one the worker died on
            status = "timeout" if rc == "timeout" else "crashed"
            results.append(_finish(c, {"status": status, "error": f"worker exit {rc}; log {wlog.name}",
                                       "times": [], "peak_rss": []}))
            break
        last = results[-1]
        med = last.get("time", {}).get("median")
        log(f"[{n}/{total}] {last['status']:7s} {('%.3fs' % med) if med is not None else '':>9s}  {last['id']}")
        # the file is rewritten whole, so not after every case of a thousand-case suite: at most every
        # SAVE_EVERY_S seconds, and at the end (a crash loses at most that much)
        if time.monotonic() - saved_at >= SAVE_EVERY_S:
            _save(out_path, meta, results)
            saved_at = time.monotonic()
    meta["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    _save(out_path, meta, results)
    return results


SAVE_EVERY_S = 10.0


def _finish(case: dict, r: dict, fresh_worker=True) -> dict:
    """The saved result row of `case`.

    `fresh_worker` says the case was the first in its worker process, so its peak RSS is its own;
    a later case in a batch starts from what the earlier ones left resident, and its peak and its
    growth both depend on which cases ran before it.
    """
    keep = {k: case[k] for k in ("id", "suite", "api", "dataset", "container", "features", "params", "mode",
                                 "sweep", "warmup", "repeats")}
    keep["status"] = r.get("status", "ok")
    keep["time"] = summarize(r.get("times", []))
    keep["times"] = r.get("times", [])
    keep["fresh_worker"] = fresh_worker
    # a repeat whose memory monitor died has no peak; the others' maximum could miss the highest,
    # so the case then has no peak either
    peaks = [] if r.get("rss_lost") else r.get("peak_rss", [])
    if r.get("rss_lost"):
        keep["rss_lost"] = r["rss_lost"]
    keep["peak_rss_mb"] = round(max(peaks) / 2**20, 1) if peaks else None
    if r.get("base_rss") and peaks:
        # a repeat can peak below the starting resident set once memory is returned; growth is then 0
        keep["delta_rss_mb"] = round(max(0, max(peaks) - r["base_rss"]) / 2**20, 1)
    for k in ("rows", "cols", "construct_s", "digest", "error", "traceback"):
        if r.get(k) is not None:
            keep[k] = r[k]
    return keep


def _save(path: Path, meta, results):
    write_atomic(path, json.dumps({"meta": meta, "cases": results}, indent=1))


# Features whose value legitimately depends on the input path, with where the docs say so.
# Keyed by input path: the case's container, or "gpu" for a case computed on the GPU; each value
# matches the feature names that path legitimately computes differently.
DOCUMENTED_DIFFERENCES = {
    # docs/source/Math/f_pixel_intensity.rst: not computed for arrays passed to Nyxus.featurize
    "numpy": re.compile(r"COVERED_IMAGE_INTENSITY_RANGE"),
    # docs/source/gpu.rst: on the GPU, about half of the central, normalized, Hu and weighted
    # moment columns of the shape and intensity moments differ from the CPU by more than rounding.
    # Shape: *CENTRAL_MOMENT_*, WT_NORM_CTR_MOM_*, *HU_M*; intensity: IMOM_[W][N]CM_*, IMOM_[W]HU*.
    # The raw and normalized spatial moments (SPAT_MOMENT, IMOM_RM, IMOM_NRM) are still compared.
    "gpu": re.compile(r".*(CENTRAL|_CTR_|HU_M|IMOM_W?N?CM_|IMOM_W?HU\d).*"),
}


def _paths(r):
    gpu = r.get("params", {}).get("use_gpu_device", -1)
    return [r["container"]] + (["gpu"] if gpu is not None and gpu >= 0 else [])


def _documented_difference(a, b, key) -> bool:
    feature = key.split("#", 1)[0]  # "<feature>#nonfinite" belongs to its feature
    return any(DOCUMENTED_DIFFERENCES[p].fullmatch(feature) for r in (a, b) for p in _paths(r)
               if p in DOCUMENTED_DIFFERENCES)


def check_digests(results, rel=1e-6) -> list:
    """Cases that differ only in how the pixels reach the features must give the same values.

    That is: container (each has its own reader), in-RAM vs out-of-core, CPU vs GPU, and thread
    count. Values are compared as per-feature sums over the ROIs, plain and weighted by ROI label
    (so values given to the wrong labels disagree), together with each feature's count of NaN and
    infinite values (worker._digest). A feature one case reports and the other
    does not is a disagreement too, with None for the missing side. Returns a list of
    (case_a, case_b, feature, a, b) disagreements.
    """
    def key(r):
        p = {k: v for k, v in r["params"].items() if k not in ("ram_limit", "use_gpu_device", "n_feature_calc_threads")}
        return (r["suite"], r["dataset"], tuple(r["features"]), json.dumps(p, sort_keys=True))

    groups = {}
    for r in results:
        if r.get("digest"):
            groups.setdefault(key(r), []).append(r)
    bad = []
    for rs in groups.values():
        ref = rs[0]
        for other in rs[1:]:
            da, db = ref["digest"], other["digest"]
            for f in sorted(set(da) | set(db)):
                if _documented_difference(ref, other, f):
                    continue
                a, b = da.get(f), db.get(f)
                if a is None or b is None or abs(a - b) > rel * max(1.0, abs(a), abs(b)):
                    bad.append((ref["id"], other["id"], f, a, b))
    return bad
