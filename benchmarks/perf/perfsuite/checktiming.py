"""Per-feature-group time breakdown from a CLI built with -DCHECKTIMING=ON.

The CHECKTIMING build wraps each feature group's calculation, and each image-scan phase, in a
stopwatch. With `--verbose=1` the 2D segmented workflow writes the totals to
`<outDir>/inclusive_nyxustiming.csv`, one row per stopwatch: category, name, share of the total, and
`rawtime` in microseconds. The stopwatches cover the 2D segmented workflow only; 3D and whole-slide
runs produce no file, so profile datasets must be 2D and have a mask.

This module runs that CLI on a dataset's tiled TIFF a few times and turns the CSV into result rows
keyed `profile/<dataset>/<cpu|gpu>/<category>/<name>`, so `compare` treats them like any other case.
"""

from __future__ import annotations

import csv
import functools
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import datasets
from .stats import summarize


def parse_csv(path: Path) -> dict:
    """{"category/name": seconds} from an inclusive_nyxustiming.csv."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("rawtime"):
                continue
            key = f"{row['h2']}/{row['h3']}"
            out[key] = out.get(key, 0.0) + float(row["rawtime"]) / 1e6
    return out


def run_cli(cli: Path, ds: datasets.Dataset, data_root: Path, *, features="*ALL*", threads=4, gpu=False,
            extra=(), timeout=7200, log_path: Path | None = None):
    """One CLI run; returns (wall seconds, {group: seconds}, return code or "timeout")."""
    if ds.dim != 2 or not ds.has_mask:
        raise ValueError(f"profile needs a 2D dataset with a mask, got {ds.name}: the CHECKTIMING "
                         "stopwatches cover the 2D segmented workflow only")
    ip, sp = datasets.file_paths(data_root, ds, "tiff-tiled")
    with tempfile.TemporaryDirectory(prefix="perfsuite-cli-") as out:
        args = [str(cli), f"--intDir={ip.parent}", f"--segDir={sp.parent}", f"--outDir={out}",
                f"--features={features}", "--filePattern=.*", "--verbose=1", f"--reduceThreads={threads}",
                "--outputType=singlecsv"]
        if gpu:
            args += ["--useGpu=true", "--gpuDeviceID=0"]
        args += list(extra)
        t0 = time.perf_counter()
        with open(log_path or os.devnull, "ab") as lf:
            try:
                rc = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                return time.perf_counter() - t0, {}, "timeout"
        wall = time.perf_counter() - t0
        csvp = Path(out) / "inclusive_nyxustiming.csv"
        groups = parse_csv(csvp) if csvp.exists() else {}
        return wall, groups, rc


def find_executable(path, what: str) -> Path:
    """`path` as an executable: a file as given, else looked up the way a shell would (PATH for a
    bare name, PATHEXT on Windows, so `build/nyxus` finds `build/nyxus.exe`); raises when there is none."""
    p = Path(path)
    if p.is_file():
        return p
    found = shutil.which(str(p))
    if not found:
        raise FileNotFoundError(f"{what} not found: {path}")
    return Path(found)


def find_cli(cli) -> Path:
    return find_executable(cli, "CHECKTIMING CLI")


SOURCE_DIR = Path(__file__).resolve().parents[3] / "src" / "nyx"
_STOPWATCH = re.compile(r'^(?P<pre>[^"]*?)\bSTOPWATCH\s*\(\s*"(?P<cat>[^"/]+)/(?P<name>[^"/]+)/')


@functools.lru_cache(maxsize=None)
def stopwatch_groups(source_dir: Path = SOURCE_DIR) -> tuple:
    """The `<category>/<name>` of every STOPWATCH in the Nyxus C++ sources, commented-out ones left out.

    `--filter` decides whether a profile run is needed by matching the row ids it can produce, which
    are known only after the run; the sources are where they are named. This harness lives in the
    Nyxus repository, next to them.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Nyxus sources not found at {source_dir}: the profile group names are read "
                                "from their STOPWATCH calls")
    out = set()
    for p in sorted(source_dir.rglob("*")):
        if p.suffix not in (".cpp", ".h", ".cu", ".cuh", ".hpp"):
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _STOPWATCH.search(line)
            if m and "//" not in m.group("pre") and "#define" not in m.group("pre"):
                out.add(f"{m.group('cat')}/{m.group('name')}")
    return tuple(sorted(out))


def run_groups(tag: str) -> list:
    """The stopwatches a profile run (2D, `*ALL*`) on `tag` (cpu|gpu) can report: not the 3D or
    image-quality ones, which `*ALL*` on a 2D image never computes, and the GPU ones only on gpu."""
    out = []
    for g in stopwatch_groups():
        cat = g.split("/", 1)[0]
        if cat.startswith("3D") or cat == "ImageQuality":
            continue
        if tag != "gpu" and (cat.startswith("GPU-") or cat == "Upload2Gpu"):
            continue
        out.append(g)
    return out


def row_ids(name: str, tag: str):
    """The row ids a profile run of `name` on `tag` (cpu|gpu) can produce."""
    return [f"profile/{name}/{tag}/{g}" for g in ("WALL", *run_groups(tag))]


def gpu_device_present(info: dict) -> bool:
    """Whether this machine has a CUDA device: the Python module saw one, or nvidia-smi lists one
    (the module may be a CPU build while the CLI is a CUDA one)."""
    if info.get("gpu_available"):
        return True
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0 and "GPU" in p.stdout


def _cli_help(cli) -> str:
    try:
        p = subprocess.run([str(find_cli(cli)), "--help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (p.stdout or "") + (p.stderr or "")


def cli_has_gpu(cli) -> bool:
    """Whether the CLI was built with USEGPU=ON: only such a build lists --useGpu in its help."""
    return "--useGpu" in _cli_help(cli)


def require_checktiming_cli(cli) -> Path:
    """`cli` as a path, refusing one that is not a CHECKTIMING build: only such a build lists
    --exclusivetiming in its help, and only it writes the timing CSV the profile suite reads."""
    path = find_cli(cli)
    if "--exclusivetiming" not in _cli_help(path):
        raise RuntimeError(f"{cli}: not a CHECKTIMING build of the nyxus CLI (its --help lists no "
                           "--exclusivetiming); build it with -DCHECKTIMING=ON")
    return path


NO_TIMING_CSV = ("exit 0 but no inclusive_nyxustiming.csv was written: "
                 "is the CLI built with -DCHECKTIMING=ON?")


def _error_text(rc, timeout) -> str:
    if rc == "timeout":
        return f"timeout after {timeout} s"
    return rc if rc == NO_TIMING_CSV else f"CLI exit {rc}"


def profile(cli: Path, dataset_names, data_root: Path, *, repeats=3, threads=16, gpu=False, log_dir: Path,
            timeout=7200, select=None, keep_row=None, log=print):
    """Rows for each dataset, CPU and (with `gpu`) GPU. `select(name, tag)` false skips that run;
    `keep_row(id)` false drops that row of a run."""
    cli = find_cli(cli)
    rows = []
    modes = [("cpu", False)] + ([("gpu", True)] if gpu else [])
    for name in dataset_names:
        ds = datasets.Dataset.parse(name)
        for tag, use_gpu in modes:
            if select is not None and not select(name, tag):
                continue
            walls, per = [], {}
            rc, i = 0, 0
            log_path = Path(log_dir) / f"cli-{name}-{tag}.log"
            log_path.unlink(missing_ok=True)  # the repeats of this run append; an earlier run's log goes
            for i in range(repeats):
                w, g, rc = run_cli(cli, ds, data_root, threads=threads, gpu=use_gpu, timeout=timeout,
                                   log_path=log_path)
                if rc != 0:
                    break
                if not g:
                    rc = NO_TIMING_CSV
                    break
                walls.append(w)
                for k, v in g.items():
                    per.setdefault(k, []).append(v)
            base = {"suite": "profile", "api": "cli", "dataset": name, "container": "tiff-tiled",
                    "features": ["*ALL*"], "params": {"n_feature_calc_threads": threads, "gpu": use_gpu},
                    "mode": "inram", "sweep": None, "warmup": 0, "repeats": repeats}
            status = "ok" if rc == 0 else ("timeout" if rc == "timeout" else "error")
            which = f"repeat {i + 1} of {repeats}: " if repeats > 1 else ""
            err = {} if rc == 0 else {"error": which + _error_text(rc, timeout)}
            run_rows = [{**base, "id": f"profile/{name}/{tag}/WALL", "status": status, "time": summarize(walls),
                         "times": walls, **err}]
            # a failed run is one failure, carried by its WALL row; its group times are not a result
            for k, v in (sorted(per.items()) if rc == 0 else ()):
                run_rows.append({**base, "id": f"profile/{name}/{tag}/{k}", "status": status, "time": summarize(v),
                                 "times": v})
            if keep_row is not None:
                # this run's rows, once each; a failed run keeps its WALL row, which carries the failure
                run_rows = [r for r in run_rows
                            if keep_row(r["id"]) or (r["status"] != "ok" and r["id"].endswith("/WALL"))]
            rows += run_rows
            log(f"profile {name} {tag}: {status}, {len(per)} groups, wall {summarize(walls).get('median', float('nan')):.2f}s")
    return rows
