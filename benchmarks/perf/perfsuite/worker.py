"""Run featurization cases in this process and report their timings as JSON lines.

    python -m perfsuite.worker <cases.json> <results.jsonl>

`cases.json` holds a list of cases (see runner.make_case). Each finished case appends one line to
`results.jsonl` before the next starts, so the runner can tell which case a crash killed and resume
after it.

What is timed is the featurize call alone: building the Nyxus object, and for the numpy container
loading the .npy files, happen before the clock starts. A monitor process (rss_monitor) samples
this process's resident set every few milliseconds; the peak it sees during a repeat is that
repeat's memory high water mark, so cases batched into one process still get their own peaks. It
has to be another process: Nyxus holds the interpreter lock for the whole featurize call, so no
thread of this one could sample during it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


class MonitorGone(RuntimeError):
    """The rss_monitor process exited or stopped answering."""


class RssSampler:
    """Highest resident set seen between start() and stop(), in bytes, sampled by a monitor process.

    A monitor that dies is replaced: the repeat it was sampling has no peak (stop() returns None),
    and the next start() samples with a new one, so one lost monitor costs one repeat's memory
    figure, not the featurize results of every case after it.
    """

    def __init__(self, period=0.005):
        import psutil

        self._p = psutil.Process()
        self._period = period
        self._mon = None
        self.restarts = 0
        self._launch()
        self.active = False
        self.samples = 0  # samples the monitor took during the last start()..stop()

    def _launch(self):
        import subprocess

        from .nyxenv import child_env

        self._mon = subprocess.Popen(
            [sys.executable, "-m", "perfsuite.rss_monitor", str(os.getpid()), str(self._period)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=child_env())

    def _relaunch(self):
        try:
            self._mon.kill()
            self._mon.wait(timeout=10)
        except Exception:  # noqa: BLE001 - already gone
            pass
        self.restarts += 1
        self._launch()

    def _ask(self, cmd: str) -> str:
        if self._mon.poll() is not None:
            raise MonitorGone(f"rss_monitor exited (code {self._mon.returncode})")
        try:
            self._mon.stdin.write(cmd + "\n")
            self._mon.stdin.flush()
            reply = self._mon.stdout.readline().strip()
        except (OSError, ValueError) as e:  # a broken or closed pipe
            raise MonitorGone(f"rss_monitor unreachable: {e}") from None
        if not reply:
            raise MonitorGone(f"rss_monitor exited (code {self._mon.poll()})")
        return reply

    def rss(self) -> int:
        return self._p.memory_info().rss

    def start(self):
        try:
            self._ask("start")  # returns once the monitor has its baseline, so the call is covered
        except MonitorGone:
            self._relaunch()
            self._ask("start")  # a monitor that cannot start at all is a harness failure
        self.active = True

    def stop(self):
        """The peak in bytes, or None when the monitor died during the repeat."""
        self.active = False
        try:
            peak, samples = self._ask("stop").split()
        except MonitorGone:
            self.samples = 0
            self._relaunch()
            return None
        self.samples = int(samples)
        return int(peak)

    def close(self):
        try:
            self._mon.stdin.write("quit\n")
            self._mon.stdin.close()
        except (OSError, ValueError):
            pass
        self._mon.wait(timeout=10)


def label_weights(labels) -> np.ndarray:
    """A weight in [1, 2) per ROI label, fixed by the label alone, so a weighted sum over ROIs
    changes when values move from one label to another."""
    lab = np.asarray(labels, dtype=np.int64)
    return 1.0 + ((lab * 2654435761) % 4093) / 4093.0


def _digest(df) -> dict:
    """Per-feature summaries over all ROIs, for cross-mode checks.

    `<feature>` is the sum of the finite values, and `<feature>#by_label` the same sum with each
    ROI's value weighted by its label (label_weights): two readers that give the same values to
    different labels agree on the first and not the second. A column with NaN or infinite values
    also gets `<feature>#nonfinite`, their count, so NaN in one case against a number in another
    shows up.
    """
    out = {}
    skip = {"ROI_label", "t_index", "c_index", "phys_x", "phys_y", "phys_z"}
    w = label_weights(df["ROI_label"]) if "ROI_label" in df.columns else None
    for col in df.columns:
        if col in skip or df[col].dtype.kind not in "fiu":
            continue
        v = np.asarray(df[col], dtype=np.float64)
        finite = np.isfinite(v)
        out[col] = float(v[finite].sum())
        if w is not None:
            out[f"{col}#by_label"] = float((v[finite] * w[finite]).sum())
        if not finite.all():
            out[f"{col}#nonfinite"] = float((~finite).sum())
    return out


def _build(case: dict):
    from . import nyxenv

    nyxenv.prepare()
    import nyxus

    kw = dict(case.get("params", {}))
    api = case["api"]
    if api == "2d":
        return nyxus.Nyxus(case["features"], **kw)
    if api == "3d":
        return nyxus.Nyxus3D(case["features"], **kw)
    if api == "imq":
        return nyxus.ImageQuality(case["features"], **kw)
    raise ValueError(f"unknown api {api!r}")


def _call(obj, case: dict, arrays):
    if case["container"] == "numpy":
        inten, mask = arrays
        if mask is None:
            mask = np.ones(inten.shape, dtype=np.uint8)
        return obj.featurize(inten, mask)
    # Nyxus3D.featurize_directory takes file_pattern without a default; pass it to every class
    seg = None if case.get("whole") else case["seg_dir"]
    return obj.featurize_directory(case["int_dir"], seg, ".*")


def run_case(case: dict, sampler: RssSampler) -> dict:
    res = {"id": case["id"], "status": "ok", "times": [], "peak_rss": [], "error": None}
    arrays = None
    try:
        if case["container"] == "numpy":
            inten = np.load(case["int_file"])
            mask = np.load(case["seg_file"]) if case.get("seg_file") else None
            arrays = (inten, mask)
        t0 = time.perf_counter()
        obj = _build(case)
        res["construct_s"] = time.perf_counter() - t0
        base = sampler.rss()
        res["base_rss"] = base
        n = case.get("warmup", 0) + case.get("repeats", 1)
        for i in range(n):
            sampler.start()
            try:
                t0 = time.perf_counter()
                df = _call(obj, case, arrays)
                dt = time.perf_counter() - t0
            finally:  # a raising call stops sampling too, so the monitor is idle when the next case starts
                peak = sampler.stop()
            if i < case.get("warmup", 0):
                continue
            res["times"].append(dt)
            if peak is None:
                res["rss_lost"] = res.get("rss_lost", 0) + 1
            else:
                res["peak_rss"].append(peak)
            if "rows" not in res:
                res["rows"], res["cols"] = int(df.shape[0]), int(df.shape[1])
                if case.get("digest"):
                    res["digest"] = _digest(df)
            del df
    except MonitorGone:
        raise  # the harness cannot measure: not this case's failure (main exits HARNESS_FAILURE)
    except Exception as e:  # noqa: BLE001 - a feature error is a result, not a harness failure
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc(limit=6)
    return res


# Exit status of a worker that cannot measure at all (its memory monitor will not start): the
# runner stops the whole run instead of charging the failure to case after case.
HARNESS_FAILURE = 70


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cases = json.loads(Path(argv[0]).read_text())
    out = Path(argv[1])
    sampler = None
    try:
        sampler = RssSampler()
        with out.open("a", encoding="utf-8") as f:
            for case in cases:
                # stdout belongs to Nyxus; announce the case on stderr so a crash log names it.
                print(f"[perfsuite] {case['id']}", file=sys.stderr, flush=True)
                r = run_case(case, sampler)
                r["pid"] = os.getpid()
                f.write(json.dumps(r) + "\n")
                f.flush()
    except MonitorGone as e:
        print(f"[perfsuite] harness failure: the memory monitor cannot run: {e}", file=sys.stderr, flush=True)
        return HARNESS_FAILURE
    finally:
        if sampler is not None:
            sampler.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
