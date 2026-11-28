"""Command line: python -m perfsuite <command> ...  (run from benchmarks/perf)

    list     print the cases a tier and suite selection would run
    synth    generate the datasets they need
    run      generate, run, check value agreement, write a results JSON; exit 1 when a case
             is not ok or two cases disagree on a value, 2 when the harness cannot measure
    compare  judge a new results file against a baseline (given, or this host's); exit 1 on
             regressions, newly failing cases, changed kernel values, or a new run that is
             incomplete; instruction-count files are judged as counts
    baseline save a results file as this host's timing baseline (outside the repository), or list
    icount   count instructions per kernel call under valgrind: the committed, machine-independent
             measure (benchmarks/perf/baselines/instructions.json)
    report   summarize one results file as Markdown
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from . import baseline, checktiming, compare, datasets, icount, kernels, presets, runner
from .stats import loglog_slope


def _selector(args):
    """case id -> whether --filter keeps it."""
    rx = re.compile(args.filter) if args.filter else None
    return lambda cid: rx is None or bool(rx.search(cid))


def _select(args):
    suites = args.suite or [s for s in presets.SUITES]
    keep = _selector(args)
    cases = [c for c in presets.build(args.tier, suites) if keep(c["id"])]
    return suites, cases


def _profile_plan(args, suites):
    """(dataset, cpu|gpu) runs of the profile suite that --filter keeps; their rows are
    `profile/<dataset>/<cpu|gpu>/<group>`. The gpu runs happen only with a CUDA CLI."""
    if "profile" not in suites:
        return []
    keep = _selector(args)
    return [(n, t) for n in presets.profile_datasets(args.tier) for t in ("cpu", "gpu")
            if keep(f"profile/{n}/{t}/") or any(keep(cid) for cid in checktiming.row_ids(n, t))]


def _needed(cases, prof):
    """{dataset: containers} the `cases` and the `prof` (dataset, cpu|gpu) profile runs read."""
    need = defaultdict(set)
    for c in cases:
        need[c["dataset"]].add(c["container"])
    for n, _ in prof:
        need[n].add("tiff-tiled")
    return need


def cmd_list(args):
    suites, cases = _select(args)
    for c in cases:
        print(c["id"])
    prof = _profile_plan(args, suites)
    for n, t in prof:
        print(f"profile/{n}/{t}/*" + ("   (CUDA CLI only)" if t == "gpu" else ""))
    kern = []
    if "kernels" in suites:
        if args.bench_exe:
            exe = checktiming.find_executable(args.bench_exe, "bench_feature_kernels")
            kern = [f"kernels/{k}/side{s}" for k, s in kernels.plan(exe, args.tier, _selector(args))]
            for cid in kern:
                print(cid)
        else:
            kern = [f"kernels/*/side{s}" for s in kernels.SIDES[args.tier]]
            for cid in kern:
                print(f"{cid}   (pass --bench-exe to list and --filter the kernels)")
    need = _needed(cases, prof)
    gb = sum(datasets.estimated_bytes(datasets.Dataset.parse(n)) * len(cs) for n, cs in need.items()) / 2**30
    print(f"# {len(cases)} cases + {len(prof)} profile runs + {len(kern)} kernel lines, "
          f"{len(need)} datasets, ~{gb:.1f} GB on disk")


def _ensure(data: Path, need):
    for n, cs in sorted(need.items()):
        datasets.ensure(data, datasets.Dataset.parse(n), sorted(cs))


def cmd_synth(args):
    suites, cases = _select(args)
    _ensure(Path(args.data), _needed(cases, _profile_plan(args, suites)))


def cmd_run(args):
    """Exit status 1 when a case is not ok or two cases disagree on a value, else 0."""
    suites, cases = _select(args)
    data = Path(args.data)
    # a wrong executable is found now, not after the Python suites have run for hours
    if "profile" in suites and args.cli and _profile_plan(args, suites):
        checktiming.require_checktiming_cli(args.cli)
    if "kernels" in suites and args.bench_exe:
        kernels.names(checktiming.find_executable(args.bench_exe, "bench_feature_kernels"))  # raises unless it is one
    info = runner.machine_info()
    meta = {**info, "tier": args.tier, "suites": suites, "data_root": str(data),
            "started": _dt.datetime.now().isoformat(timespec="seconds")}
    # what will run is settled before any dataset is generated, so a skipped suite costs nothing
    if "gpu" in suites and not info.get("gpu_available"):
        print("gpu suite skipped: this nyxus build reports no usable GPU", file=sys.stderr)
        cases = [c for c in cases if c["suite"] != "gpu"]
    prof = _profile_plan(args, suites)
    prof_gpu = False
    if prof and not args.cli:
        print("profile suite skipped: pass --cli <nyxus CLI built with -DCHECKTIMING=ON>", file=sys.stderr)
        prof = []
    elif prof:
        # a CUDA CLI on a machine with a CUDA device
        prof_gpu = (not args.no_gpu_profile and checktiming.cli_has_gpu(args.cli)
                    and checktiming.gpu_device_present(info))
        prof = [(n, t) for n, t in prof if t == "cpu" or prof_gpu]
    if not args.no_synth:
        _ensure(data, _needed(cases, prof))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = runner.log_dir_of(out)  # every stage's logs, the features stage's too
    results = []

    def save():  # after every stage, so a stage that fails leaves the earlier ones on disk
        runner._save(out, meta, results)

    batch = args.batch
    feats = [c for c in cases if c["suite"] == "features"]
    rest = [c for c in cases if c["suite"] != "features"]
    try:
        if rest:
            results += runner.run(rest, data, out, batch=batch or 1, timeout=args.timeout, tier=args.tier,
                                  info=info, log_dir=logs)
            save()
        if feats:
            part = out.with_name(out.stem + "-features.json")
            results += runner.run(feats, data, part, batch=batch or 64, timeout=args.timeout, tier=args.tier,
                                  info=info, log_dir=logs)
            save()
            part.unlink(missing_ok=True)  # its cases are in `out` now
    except runner.HarnessFailure as e:
        results += e.results  # what that stage measured before it stopped
        save()  # no `finished`: compare calls the run incomplete
        print(f"\nHARNESS FAILURE: {e}", file=sys.stderr)
        return 2
    if prof:
        logs.mkdir(parents=True, exist_ok=True)
        results += checktiming.profile(args.cli, presets.profile_datasets(args.tier), data,
                                       repeats=presets.REPEATS[args.tier][1], threads=args.cli_threads,
                                       gpu=prof_gpu, log_dir=logs, timeout=args.timeout,
                                       select=lambda n, t: (n, t) in prof,
                                       keep_row=_selector(args) if args.filter else None)
        save()
    if "kernels" in suites:
        if not args.bench_exe:
            print("kernels suite skipped: pass --bench-exe <bench_feature_kernels built with -DRUN_BENCHMARKS=ON>",
                  file=sys.stderr)
        else:
            results += kernels.run(args.bench_exe, args.tier, timeout=args.timeout,
                                   select=_selector(args) if args.filter else None)
    meta["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    save()
    bad = runner.check_digests(results)
    failed = [r for r in results if r["status"] != "ok"]
    print(f"\n{len(results) - len(failed)}/{len(results)} cases ok -> {out}")
    for r in failed:
        print(f"  {r['status']:8s} {r['id']}  {r.get('error', '')}")
    if bad:
        print(f"\nVALUE DISAGREEMENTS: {len(bad)}")
        for a, b, f, va, vb in bad[:40]:
            print(f"  {f}: {va!r} vs {vb!r}\n    {a}\n    {b}")
    return 1 if (bad or failed) else 0


def cmd_compare(args):
    """`compare NEW` judges NEW against its host's baseline (see baseline); `compare OLD NEW`
    against OLD. Instruction-count files (icount) are compared as counts."""
    if len(args.files) > 2:
        raise SystemExit("compare: give NEW, or OLD NEW")
    nd = json.loads(Path(args.files[-1]).read_text())  # each file is read and parsed once
    if len(args.files) == 1:
        try:
            old, od = baseline.find(nd)
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(f"compare: {e}") from None
        print(f"baseline: {old}")
    else:
        od = json.loads(Path(args.files[0]).read_text())
    if icount.is_counts(od) != icount.is_counts(nd):
        raise SystemExit("compare: one file holds instruction counts and the other timings")
    if icount.is_counts(nd):
        r = icount.compare(od, nd, tol=args.icount_tol, lib_tol=args.icount_lib_tol,
                           allow_toolchain_change=args.allow_toolchain_change, allow_missing=args.allow_missing)
        print(icount.format_report(r))
        # A count that is unusable in itself (unfinished, failed, or holding counts that are not
        # counts) fails as such, whatever it was made with. Counts from another toolchain say
        # nothing about the code: their own status. Then the counts themselves.
        if r["incomplete"] or r["invalid"] or r["old_incomplete"]:
            return 1
        if r["toolchain_change"] or r["method_change"]:
            return TOOLCHAIN_CHANGED  # either way the counts were not made as these were
        return 1 if icount.failed(r) else 0
    r = compare.compare(od, nd, tol=args.tol, k=args.k, floor=args.floor, kernel_floor=args.kernel_floor,
                        mem_tol=args.mem_tol, allow_missing=args.allow_missing)
    print(compare.format_report(r))
    return 1 if compare.failed(r) else 0


def cmd_baseline(args):
    if args.action == "save":
        if not args.results:
            raise SystemExit("baseline save: name the results file to keep")
        try:
            print(f"saved {baseline.save(args.results, force=args.force)}")
        except ValueError as e:
            raise SystemExit(f"baseline save: {e}") from None
    else:
        rows = baseline.listing()
        print(f"baselines in {baseline.store()}:" if rows else f"no baselines in {baseline.store()}")
        for host, tier, p in rows:
            print(f"  {host:24s} {tier:6s} {p}")
    return 0


# compare's exit status when instruction counts that are sound in themselves were made with another
# toolchain or counting method: they say nothing about the code (the CI job then asks for its own)
TOOLCHAIN_CHANGED = 3


def cmd_icount(args):
    try:
        data = icount.count(args.bench_exe, args.build_dir, sides=args.side or icount.SIDES, timeout=args.timeout,
                            jobs=args.jobs, select=_selector(args) if args.filter else None,
                            source_sha=args.source_sha)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        # nothing selected, a mismatched build, no valgrind, no compile_commands.json, not the
        # benchmark, one that cannot be run or does not answer: a setup problem, said as one, and
        # no file written (OSError covers a missing or non-executable file)
        raise SystemExit(f"icount: {e}") from None
    icount.save(args.out, data)  # also when a run failed: the counts that finished are kept
    print(f"{len(data['counts'])} kernel counts -> {args.out}")
    if data["meta"].get("error"):
        print(f"icount: {data['meta']['error']}", file=sys.stderr)
        return 1
    return 0


def _fmt(s):
    if s is None:
        return "-"
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s"


def _mb(v):
    return "-" if v is None else f"{v:g}"


def _rss(c):
    """Peak RSS, marked * when the case ran after others in its worker (compare does not judge it)."""
    v = c.get("peak_rss_mb")
    if v is None:
        return "-"
    return f"{v:g}" + ("" if c.get("fresh_worker", True) else "*")


def cmd_report(args):
    # the report is Markdown with non-ASCII text (·, r²); redirected to a file it must be UTF-8,
    # not the console code page Windows would otherwise use
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    d = json.loads(Path(args.results).read_text())
    m, cases = d["meta"], d["cases"]
    dirty = {True: " (dirty)", False: ""}.get(m.get("git_dirty"), " (dirty: unknown)")
    print(f"# Nyxus perf: {m.get('tier')} tier, {m.get('git_sha', '?')}{dirty}\n")
    print(f"{m.get('cpu')} · {m.get('physical_cpus')} cores · {m.get('ram_gb')} GB · "
          f"{m.get('gpu', 'no GPU') if m.get('gpu_available') else 'no GPU'} · {m.get('os')} · nyxus {m.get('nyxus_version')}\n")
    by = defaultdict(list)
    for c in cases:
        by[c["suite"]].append(c)
    for suite in ("load", "groups", "gpu", "profile", "kernels"):
        rs = by.get(suite)
        if not rs:
            continue
        print(f"## {suite}\n\n| case | median | MAD | peak RSS MB | growth MB | status |\n"
              "|---|---:|---:|---:|---:|---|")
        for c in rs:
            t = c.get("time", {})
            print(f"| `{c['id'].split('/', 1)[1]}` | {_fmt(t.get('median'))} | {_fmt(t.get('mad'))} | "
                  f"{_rss(c)} | {_mb(c.get('delta_rss_mb'))} | {c['status']} |")
        print()
    fs = by.get("features")
    if fs:
        ok = sorted((c for c in fs if c["status"] == "ok"), key=lambda c: -c["time"]["median"])
        print(f"## features ({len(ok)}/{len(fs)} ok), slowest {min(args.top, len(ok))}\n")
        print("Peak RSS marked * is after other cases in the same worker, so not the case's own; "
              "compare judges only unmarked peaks.\n")
        print("| feature | family | dataset | median | peak RSS MB | growth MB |\n|---|---|---|---:|---:|---:|")
        for c in ok[: args.top]:
            print(f"| {c['features'][0]} | {c['id'].rsplit('/', 1)[1]} | {c['dataset']} | "
                  f"{_fmt(c['time']['median'])} | {_rss(c)} | {_mb(c.get('delta_rss_mb'))} |")
        print()
    failed = [c for c in cases if c["status"] != "ok"]
    if failed:  # every suite's, the sweeps' too, whose table below shows only the points that ran
        print(f"## not ok ({len(failed)})\n\n| case | status | error |\n|---|---|---|")
        for c in failed:
            err = (c.get("error") or "").replace("|", "\\|").replace("\n", " ")[:200]
            print(f"| `{c['id']}` | {c['status']} | {err} |")
        print()
    sw = defaultdict(list)
    for c in cases:
        if c.get("sweep") and c["status"] == "ok":
            sw[c["sweep"]["name"]].append((c["sweep"]["x"], c["time"]["median"]))
    if sw:
        print("## scaling exponents (slope of log time vs log x)\n\n| sweep | points | slope | r² |\n|---|---:|---:|---:|")
        for name, pts in sorted(sw.items()):
            fit = loglog_slope(pts)
            if fit:
                print(f"| {name} | {len(pts)} | {fit[0]:.2f} | {fit[1]:.3f} |")
        print()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="perfsuite", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def sel(sp):
        sp.add_argument("--tier", choices=presets.TIERS, default="smoke")
        sp.add_argument("--suite", action="append", choices=presets.SUITES,
                        help="repeatable; default all suites")
        sp.add_argument("--filter", help="regex on case ids")

    sp = sub.add_parser("list"); sel(sp)
    sp.add_argument("--bench-exe", help="bench_feature_kernels, to list the kernels cases by name")
    sp.set_defaults(fn=cmd_list)
    sp = sub.add_parser("synth"); sel(sp); sp.add_argument("--data", required=True); sp.set_defaults(fn=cmd_synth)
    sp = sub.add_parser("run"); sel(sp)
    sp.add_argument("--data", required=True, help="dataset cache directory")
    sp.add_argument("--out", required=True, help="results JSON")
    sp.add_argument("--batch", type=int, default=0, help="cases per worker process (default 1; 64 for features)")
    sp.add_argument("--timeout", type=int, default=3600,
                    help="seconds per case; also per CLI run (profile) and for the whole kernels run")
    sp.add_argument("--cli", help="nyxus CLI built with -DCHECKTIMING=ON, for the profile suite")
    sp.add_argument("--cli-threads", type=int, default=16)
    sp.add_argument("--bench-exe", help="bench_feature_kernels executable, for the kernels suite")
    sp.add_argument("--no-gpu-profile", action="store_true")
    sp.add_argument("--no-synth", action="store_true")
    sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("compare", help="[OLD] NEW: without OLD, NEW's host baseline")
    sp.add_argument("files", nargs="+", metavar="FILE", help="[OLD] NEW")
    sp.add_argument("--tol", type=float, default=0.15); sp.add_argument("--k", type=float, default=3.0)
    sp.add_argument("--floor", type=float, default=0.02, help="seconds; the kernels suite has --kernel-floor")
    sp.add_argument("--kernel-floor", type=float, default=compare.KERNEL_FLOOR)
    sp.add_argument("--mem-tol", type=float, default=0.20)
    sp.add_argument("--allow-missing", action="store_true",
                    help="do not fail on baseline cases or kernel counts the new file lacks (comparing a subset)")
    sp.add_argument("--icount-tol", type=float, default=0.0,
                    help="relative change an instruction count may make (default 0: counts repeat exactly)")
    sp.add_argument("--icount-lib-tol", type=float, default=0.005,
                    help="change of a kernel's library instructions, as a share of its whole count")
    sp.add_argument("--allow-toolchain-change", action="store_true",
                    help="compare instruction counts made with different toolchains")
    sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("baseline", help="keep or list this host's timing baselines")
    sp.add_argument("action", choices=("save", "list"))
    sp.add_argument("results", nargs="?")
    sp.add_argument("--force", action="store_true", help="save a run that did not finish cleanly")
    sp.set_defaults(fn=cmd_baseline)
    sp = sub.add_parser("icount", help="count instructions per kernel call under valgrind")
    sp.add_argument("--bench-exe", required=True)
    sp.add_argument("--build-dir", required=True,
                    help="the benchmark's CMake build directory, whose compile_commands.json is hashed")
    sp.add_argument("--out", required=True)
    sp.add_argument("--side", type=int, action="append", help=f"repeatable; default {list(icount.SIDES)}")
    sp.add_argument("--filter", help="regex on kernels/<kernel>/side<N>")
    sp.add_argument("--timeout", type=int, default=3600, help="seconds per valgrind run")
    sp.add_argument("--jobs", type=int, default=0, help="valgrind runs at once (default: one per CPU)")
    sp.add_argument("--source-sha", help="the commit the counted one was made from (CI: a pull request's head)")
    sp.set_defaults(fn=cmd_icount)
    sp = sub.add_parser("report"); sp.add_argument("results"); sp.add_argument("--top", type=int, default=40)
    sp.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
