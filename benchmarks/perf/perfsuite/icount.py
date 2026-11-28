"""Instructions per kernel call, counted by valgrind: a measure that does not depend on the machine.

A timing depends on the CPU, its clock, its caches and what else the machine is doing, so a timing
baseline only means something on the host that recorded it (see `baseline`). The number of
instructions a kernel executes depends on the code and on how it was built, not on the speed of the
hardware, so it can be committed and checked anywhere the same build is made.

Each `bench_feature_kernels` kernel and side runs once under `valgrind --tool=callgrind`, with one
repetition, no warm-up and no timing, collecting only inside `bench_kernel()`: the kernel's call
on a prepared ROI. Process start, dataset generation, dependency preparation and building the ROI
(whose cost grows with every feature code Nyxus registers) are outside it and not counted, so a
count changes only with its kernel. Timing is off because it does not
repeat: a clock read can retry, and printing times that differ costs a different number of
instructions.

Two counts are kept per kernel. `instructions` are those executed in the benchmark executable
itself (the Nyxus code and the library templates compiled into it); they are the same on every
x86-64 CPU and repeat exactly, so the default tolerance is 0. `library_instructions` are those
executed in the shared libraries the repetition calls (libm, libc, libstdc++): work moved there,
an exp() called twice as often, shows in them. glibc picks its routines by the CPU's vendor,
features and cache sizes (hiding AVX2, FMA, BMI2 and ERMS from it moved the library instructions
by up to 2 % of a kernel's whole count), so the count runs with GLIBC_TUNABLES making that choice
the same on every x86-64 CPU (GLIBC_PIN: the baseline instruction set, fixed preferences, fixed
cache thresholds). Library instructions are judged with a tolerance taken as a share of the
kernel's whole count (`lib_tol`, default 0.5 %), for what a glibc tunable cannot reach.

What a count depends on besides the code is recorded with it (`toolchain`), in two kinds. The
toolchain proper (COMPARED: the compiler, the exact compiler, C and C++ library and valgrind package
versions where dpkg knows them, the C library, the architecture, the valgrind version) is the
environment: `compare` refuses counts made with another one rather than reporting the difference
as a change in the code. The build configuration (BUILD: the build type, a hash of every compile
command of the Nyxus code and the benchmark from compile_commands.json, covering flags, standard,
definitions, include paths, IPO and PIC, and whether it is a GPU or CHECKTIMING build) is part of
the change being judged: counts made with another one are compared, and the difference is reported
with them. The CPU features are recorded for reference and not compared: the executable's own
instructions do not depend on them, and the library instructions run pinned.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from . import kernels
from .checktiming import find_executable
from .compare import same_value
from .fileio import write_atomic
from .runner import git_dirty, git_sha

KIND = "instructions"
SIDES = (64, 256)  # valgrind runs a kernel about 50 times slower; these sizes still set the scaling
REPETITION = "bench_kernel"  # the benchmark's kernel-call function, the only code collected
TOOL = ["--tool=callgrind", "--compress-strings=no", "--compress-pos=no", "--collect-atstart=no",
        f"--toggle-collect={REPETITION}*"]
# glibc's routine choice made the same on every x86-64 CPU: every optional instruction set off
# (SSE2 is the x86-64 baseline), the vendor-preference bits set one way, and the size thresholds
# that are otherwise derived from the CPU's caches given explicitly. glibc ignores names it lacks.
GLIBC_PIN = ":".join([
    "glibc.cpu.hwcaps=" + ",".join(
        ["-" + f for f in ("SSE3", "SSSE3", "SSE4_1", "SSE4_2", "POPCNT", "AVX", "AVX2", "AVX_VNNI", "FMA", "FMA4",
                           "F16C", "BMI1", "BMI2", "LZCNT", "MOVBE", "ERMS", "FSRM", "RTM", "AVX512F", "AVX512CD",
                           "AVX512BW", "AVX512DQ", "AVX512VL", "AVX512_VNNI", "AVX512_BF16", "AVX512_FP16")]
        + ["-" + f for f in ("Fast_Rep_String", "Fast_Copy_Backward", "Fast_Unaligned_Load", "Fast_Unaligned_Copy",
                             "Prefer_PMINUB_for_stringop", "Prefer_No_VZEROUPPER", "Prefer_ERMS", "Prefer_FSRM",
                             "Prefer_No_AVX512", "Avoid_Short_Distance_REP_MOVSB", "Slow_SSE4_2", "Slow_BSF")]),
    "glibc.cpu.x86_data_cache_size=32768", "glibc.cpu.x86_shared_cache_size=1048576",
    "glibc.cpu.x86_non_temporal_threshold=786432", "glibc.cpu.x86_rep_movsb_threshold=2048",
    "glibc.cpu.x86_rep_stosb_threshold=2048",
])
GLIBC_FEATURES = ("avx", "avx2", "fma", "bmi2", "erms")  # recorded for reference
# What a count was made with, in two kinds. The toolchain is the environment: counts made with
# another one are not compared at all (compare exits 3, and the CI job asks for counts made with its
# pinned toolchain). The build configuration is part of the change being judged: counts made with another one
# are compared, and the difference is reported next to them.
COMPARED = ("compiler", "packages", "libc", "machine", "valgrind")
BUILD = ("build_type", "compile_hash", "gpu", "checktiming")
# dpkg-query patterns: every installed compiler version, its libraries, the assembler and linker
# (binutils), the C library, cmake (which writes the compile and link lines) and valgrind
PACKAGES = ("gcc-[0-9]*", "g++-[0-9]*", "cpp-[0-9]*", "libgcc-*-dev", "libstdc++-*-dev", "libstdc++6", "binutils",
            "binutils-x86-64-linux-gnu", "libc6", "libc6-dev", "cmake", "valgrind")
_SUMMARY = re.compile(r"^summary:\s+(\d+)", re.M)


def _valgrind() -> str:
    exe = shutil.which("valgrind")
    if not exe:
        raise FileNotFoundError("valgrind not found: instruction counts need valgrind (Linux)")
    return exe


def valgrind_version() -> str:
    return subprocess.run([_valgrind(), "--version"], capture_output=True, text=True).stdout.strip()


def build_info(exe) -> dict:
    """`--build-info` of the benchmark: compiler, pointer bits, build type, gpu, checktiming."""
    p = subprocess.run([str(exe), "--build-info"], capture_output=True, text=True, timeout=60)
    out = {}
    for ln in p.stdout.splitlines():
        k, _, v = ln.partition(" ")
        if k:
            out[k] = v.strip()
    if p.returncode != 0 or "compiler" not in out:
        raise RuntimeError(f"{exe}: --build-info printed no compiler; is it a current bench_feature_kernels?")
    return out


def source_dir(build_dir) -> str:
    """The source directory `build_dir` was configured from (CMakeCache.txt's CMAKE_HOME_DIRECTORY)."""
    cache = Path(build_dir) / "CMakeCache.txt"
    try:
        for ln in cache.read_text(errors="replace").splitlines():
            if ln.startswith("CMAKE_HOME_DIRECTORY:"):
                return ln.split("=", 1)[1].strip()  # as configured: the spelling the commands use
    except OSError:
        pass
    raise FileNotFoundError(f"{cache} names no CMAKE_HOME_DIRECTORY: is {build_dir} a CMake build directory?")


HASHED_TARGETS = ("/nyxus_core.dir/", "/bench_feature_kernels.dir/")  # the code a count runs


def _split(command: str) -> list:
    """A compile_commands.json command line as its arguments: POSIX shell rules, or on Windows the
    command-line rules there, where a backslash is a path separator, not an escape."""
    if os.name != "nt":
        return shlex.split(command)
    return [a[1:-1] if len(a) > 1 and a[0] == a[-1] == '"' else a for a in shlex.split(command, posix=False)]


def _path_swaps(build: str, src: str) -> list:
    """(path, placeholder) pairs that take a build's own directories out of a command line: the
    build and source directories, each as CMake wrote it and as its symlinks resolve, longest first
    (a build directory inside the source directory is replaced as itself, not as part of the source)."""
    pairs = {(build, "<build>"), (os.path.realpath(build), "<build>"), (src, "<src>"),
             (os.path.realpath(src), "<src>")}
    return sorted((p for p in pairs if p[0]), key=lambda s: -len(s[0]))


def _normalise(text: str, swaps: list) -> str:
    for old, new in swaps:
        text = text.replace(old, new)
    return text


def compile_hash(build_dir) -> str:
    """sha256 of the compile commands of the Nyxus sources and the benchmark in `build_dir`'s
    compile_commands.json, with the build and source directories and each command's own source and
    output file taken out, so two checkouts built alike hash alike."""
    path = Path(build_dir) / "compile_commands.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: configure the benchmark build with CMake's Ninja or Makefile "
                                "generator (RUN_BENCHMARKS exports it)")
    src = source_dir(build_dir)
    commands = set()
    swaps_for = {}  # per build directory: every entry of a build has the same one
    for e in json.loads(path.read_text()):
        d = e.get("directory", "")
        if d not in swaps_for:
            swaps_for[d] = _path_swaps(d, src)
        swaps = swaps_for[d]
        args = e.get("arguments") or _split(e["command"])
        # by target, not by source: the Python backend and the unit tests compile the same sources
        # with other definitions, and appear here when compile commands are exported for them too
        obj = e.get("output") or next((args[i + 1] for i, a in enumerate(args[:-1]) if a == "-o"), "")
        if not any(t in obj.replace("\\", "/") for t in HASHED_TARGETS):
            continue
        out, skip = [], False
        for a in args:
            if skip:
                skip = False
                continue
            if a in ("-o", "-c"):  # the object written, the source compiled: this entry's own files
                skip = True
                continue
            if a == e["file"]:  # the source given apart from -c
                continue
            out.append(_normalise(a, swaps))
        commands.add(" ".join(out))
    if not commands:
        raise RuntimeError(f"{path} holds no compile command of the Nyxus sources or the benchmark")
    # code is generated at link time too (LTO), and linker flags change the executable: the link
    # configuration goes into the hash with the compile commands
    # the build directory absolute, as the compile entries name it: a relative --build-dir such as
    # "build" would otherwise be replaced wherever that word appears (-Wl,--build-id)
    link_swaps = _path_swaps(os.path.abspath(str(build_dir)), src)
    link = [_normalise(ln, link_swaps) for ln in link_config(build_dir)]
    return hashlib.sha256("\n".join(sorted(commands) + link).encode()).hexdigest()


_LINK_CACHE_KEYS = ("CMAKE_EXE_LINKER_FLAGS", "CMAKE_INTERPROCEDURAL_OPTIMIZATION", "CMAKE_LINKER:",
                    "CMAKE_AR:", "CMAKE_RANLIB:")


def link_config(build_dir) -> list:
    """The link settings of the benchmark: the CMake cache's linker entries, and its own link line:
    with the Ninja generator the LINK_FLAGS in build.ninja, with the Makefile generator the command
    in CMakeFiles/bench_feature_kernels.dir/link.txt."""
    out = []
    try:
        for ln in (Path(build_dir) / "CMakeCache.txt").read_text(errors="replace").splitlines():
            if ln.startswith(_LINK_CACHE_KEYS):
                out.append(ln.split(":", 1)[0] + "=" + ln.split("=", 1)[1].strip() if "=" in ln else ln)
    except OSError:
        pass
    ninja = Path(build_dir) / "build.ninja"
    if ninja.is_file():
        inside = False
        for ln in ninja.read_text(errors="replace").splitlines():
            if ln.startswith("build bench_feature_kernels"):
                inside = True
            elif inside and not ln.startswith(" "):
                break
            elif inside and ln.strip().startswith("LINK_FLAGS ="):
                out.append("LINK_FLAGS=" + ln.split("=", 1)[1].strip())
    link_txt = Path(build_dir) / "CMakeFiles" / "bench_feature_kernels.dir" / "link.txt"
    if link_txt.is_file():
        # the whole link command but its inputs and output, which name this build's own files
        args = [a for a in _split(link_txt.read_text(errors="replace").strip())
                if not a.endswith((".o", ".obj", ".a")) and "bench_feature_kernels" not in a]
        out.append("LINK=" + " ".join(args))
    return sorted(out)


def package_versions() -> str:
    """'name=version ...' of PACKAGES as dpkg knows them; 'unknown' without dpkg."""
    if not shutil.which("dpkg-query"):
        return "unknown"
    p = subprocess.run(["dpkg-query", "-W", "-f", "${db:Status-Abbrev}${Package}=${Version}\n", *PACKAGES],
                       capture_output=True, text=True)
    # installed ones only ("ii "): a pattern also lists packages dpkg merely knows of
    got = [ln[3:].strip() for ln in p.stdout.splitlines() if ln.startswith("ii ")]
    return " ".join(sorted(g for g in got if "=" in g and not g.endswith("="))) or "unknown"


def cpu_features(cpuinfo=Path("/proc/cpuinfo")) -> str:
    """The GLIBC_FEATURES this CPU has, space-separated ('unknown' without /proc/cpuinfo)."""
    try:
        text = Path(cpuinfo).read_text()
    except OSError:
        return "unknown"
    flags = set()
    for ln in text.splitlines():
        if ln.startswith("flags"):
            flags = set(ln.split(":", 1)[1].split())
            break
    return " ".join(f for f in GLIBC_FEATURES if f in flags)


def provenance(build_dir) -> tuple:
    """(git sha, git_dirty) of the checkout `build_dir` was configured from; None for what git
    cannot tell (no checkout, or no CMake cache naming one)."""
    try:
        src = Path(source_dir(build_dir))
    except FileNotFoundError:
        return None, None
    return git_sha(src), git_dirty(src)


def toolchain(exe, build_dir) -> dict:
    """What an instruction count depends on besides the code."""
    libc = " ".join(platform.libc_ver()).strip() or "unknown"
    return {**build_info(exe), "compile_hash": compile_hash(build_dir), "packages": package_versions(),
            "libc": libc, "machine": platform.machine(), "valgrind": valgrind_version(),
            "cpu_features": cpu_features()}


def split_counts(callgrind_out: str, exe) -> tuple:
    """(instructions in `exe`, instructions in every other object) of an uncompressed callgrind
    output file. Each function's own cost lines count toward the object named by the `ob=` line in
    force; the one cost line after a `calls=` line is the inclusive cost of that call, which the
    callee's own lines already count, so it is skipped."""
    want = os.path.realpath(str(exe))
    resolved = {}  # callgrind repeats the same few object names before every function
    ob, skip, own, other, seen = None, False, 0, 0, False
    for ln in callgrind_out.splitlines():
        if skip:
            skip = False
            continue
        if ln.startswith("ob="):
            name = ln[3:].strip()
            if name not in resolved:
                resolved[name] = os.path.realpath(name)
            ob = resolved[name]
            seen = seen or ob == want
        elif ln.startswith("calls="):
            skip = True
        elif ln[:1].isdigit():
            parts = ln.split()
            if len(parts) >= 2:
                if ob == want:
                    own += int(parts[1])
                else:
                    other += int(parts[1])
    if not seen:
        raise RuntimeError(f"callgrind recorded no code in {exe}")
    return own, other


def _instructions(args, timeout) -> tuple:
    """(own, library, kernel row) of one run of the benchmark under valgrind, collecting inside
    REPETITION only, with glibc's routine choice pinned."""
    env = {**os.environ, "GLIBC_TUNABLES": GLIBC_PIN}
    with tempfile.TemporaryDirectory(prefix="perfsuite-icount-") as td:
        out = Path(td) / "cg.out"
        p = subprocess.run([_valgrind(), *TOOL, f"--callgrind-out-file={out}", *args],
                           capture_output=True, text=True, timeout=timeout, env=env)
        if p.returncode != 0:
            raise RuntimeError(f"{' '.join(args)}: exit {p.returncode}: {p.stderr.strip()[-300:]}")
        text = out.read_text()
        if not _SUMMARY.search(text):
            raise RuntimeError(f"{' '.join(args)}: callgrind wrote no instruction summary")
        rows = kernels.parse(p.stdout)
        if not rows:  # no CSV header, or no row: the kernel value would go unchecked
            raise RuntimeError(f"{' '.join(args)}: printed no kernel row: {p.stdout.strip()[:200]!r}")
        own, lib = split_counts(text, args[0])
        return own, lib, rows[0]


def _one(exe, k, s, timeout) -> dict:
    own, lib, row = _instructions([str(exe), "--kernel", k, "--side", str(s), "--reps", "1", "--no-warmup",
                                   "--no-timing"], timeout)
    if row.get("id") != f"kernels/{k}/side{s}":
        raise RuntimeError(f"kernels/{k}/side{s}: the benchmark reported {row.get('id')} instead")
    if own <= 0:
        raise RuntimeError(f"kernels/{k}/side{s}: no instructions collected in {REPETITION}(); is the "
                           "benchmark current, and its repetition function not inlined?")
    return {"instructions": own, "library_instructions": lib, "value": row["value"]}


def count(exe, build_dir, *, sides=SIDES, select=None, timeout=3600, jobs=None, source_sha=None,
          log=print) -> dict:
    """An instructions results file: {"meta": ..., "counts": {case id: {...}}}. The valgrind runs are
    independent (a count does not depend on what else runs), so `jobs` of them run at once. When one
    fails, no run starts after it (those already running finish), and the file holds the counts that
    finished and the error in meta["error"], the first failure in the plan's order, with no
    `finished`."""
    exe = find_executable(exe, "bench_feature_kernels")
    if Path(build_dir).resolve() not in Path(exe).resolve().parents:
        raise ValueError(f"{exe} is not in {build_dir}: the counts would be recorded with another build's "
                         "compile commands")
    tc = toolchain(exe, build_dir)
    sha, dirty = provenance(build_dir)
    meta = {"kind": KIND, "toolchain": tc, "sides": list(sides),
            # the commit the counted code is from, and whether the checkout differed from it; and, when
            # the counted commit is one made for the run (a pull request's merge commit), the commit it
            # was made from, which stays in the history
            "git_sha": sha, "git_dirty": dirty, **({"source_sha": source_sha} if source_sha else {}),
            "started": _dt.datetime.now().isoformat(timespec="seconds"),
            "method": f"valgrind callgrind Ir inside {REPETITION}(), one repetition, no warm-up, no timing; "
                      f"GLIBC_TUNABLES={GLIBC_PIN}"}
    todo = kernels.plan(exe, sides=sides, select=select)
    if not todo:
        raise ValueError("no kernel and side to count: --filter and --side select none")
    counts, errors = {}, {}
    stop = threading.Event()  # set by the first failure: no run starts after it

    def task(k, s):
        if stop.is_set():
            return None
        try:
            return _one(exe, k, s, timeout)
        except Exception:
            stop.set()
            raise

    pool = ThreadPoolExecutor(max_workers=jobs or os.cpu_count() or 1)
    try:
        futures = {pool.submit(task, k, s): f"kernels/{k}/side{s}" for k, s in todo}
        wait(futures)
        for f, cid in futures.items():
            try:
                c = f.result()
            except Exception as e:  # noqa: BLE001 - recorded, and the run is marked unfinished
                errors[cid] = str(e)
                continue
            if c is not None:
                counts[cid] = c
    finally:
        pool.shutdown(wait=True)
    order = [f"kernels/{k}/side{s}" for k, s in todo]  # what is saved and reported, in the plan's order
    counts = {cid: counts[cid] for cid in order if cid in counts}
    failed_ids = [cid for cid in order if cid in errors]
    error = (f"{failed_ids[0]}: {errors[failed_ids[0]]}" + (f" (and {len(failed_ids) - 1} more)"
                                                             if len(failed_ids) > 1 else "")) if failed_ids else None
    for cid, c in counts.items():
        log(f"icount: {cid} {c['instructions']:,} + {c['library_instructions']:,} in libraries")
    if error:
        meta["error"] = error
    else:
        meta["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    return {"meta": meta, "counts": counts}


def save(path, data):
    write_atomic(path, json.dumps(data, indent=1, sort_keys=True) + "\n")


def is_counts(data) -> bool:
    return data.get("meta", {}).get("kind") == KIND


def compare(old: dict, new: dict, *, tol=0.0, lib_tol=0.005, allow_toolchain_change=False,
            allow_missing=False) -> dict:
    """Judge new instruction counts against old ones.

    `instructions` repeat exactly, so any change beyond `tol` (default 0) either way is a change in
    the code: a rise regresses, and a fall (ratio below 1 / (1 + tol), the mirror of the rise) fails
    too until the committed counts are updated, or a later rise would be judged against the stale
    count. `library_instructions` fail when they change by more than `lib_tol` of the kernel's whole
    old count. A changed kernel value, a count that is not positive, a kernel only the new counts
    have (it has no committed count to be judged by), a kernel the old counts have and the new ones
    lack (unless `allow_missing`, for a narrowed count), and counts made with a different toolchain
    (unless `allow_toolchain_change`) fail too.
    """
    ot, nt = old["meta"].get("toolchain", {}), new["meta"].get("toolchain", {})
    tc = {} if allow_toolchain_change else {k: (ot.get(k), nt.get(k)) for k in COMPARED if ot.get(k) != nt.get(k)}
    # how the counts were made (collection boundary, valgrind options, the glibc pin) is not a
    # toolchain: counts made another way measure something else, and no flag compares them
    method = old["meta"].get("method") != new["meta"].get("method")
    build = {k: (ot.get(k), nt.get(k)) for k in BUILD if ot.get(k) != nt.get(k)}
    regressions, improvements, value_changes, invalid, library = [], [], [], [], []
    oc, nc = old["counts"], new["counts"]
    for cid in sorted(set(oc) & set(nc)):
        a, b = oc[cid].get("instructions"), nc[cid].get("instructions")
        la, lb = oc[cid].get("library_instructions"), nc[cid].get("library_instructions")  # None: not counted
        if not all(isinstance(v, int) for v in (a, b, la, lb)) or a <= 0 or b <= 0 or la < 0 or lb < 0:
            invalid.append((cid, a, b))
            continue
        row = {"id": cid, "old": a, "new": b, "ratio": b / a}
        if row["ratio"] > 1 + tol:
            regressions.append(row)
        elif row["ratio"] < 1 / (1 + tol):
            improvements.append(row)
        if abs(lb - la) > lib_tol * (a + la):
            library.append({"id": cid, "old": la, "new": lb, "share": (lb - la) / (a + la)})
        if not same_value(oc[cid].get("value"), nc[cid].get("value")):
            value_changes.append((cid, oc[cid].get("value"), nc[cid].get("value")))
    return {"kind": KIND, "toolchain_change": tc, "build_change": build,
            "method_change": (old["meta"].get("method"), new["meta"].get("method")) if method else None,
            # a reference that did not finish is no reference: what it lacks would read as new kernels
            "old_incomplete": "finished" not in old["meta"] or bool(old["meta"].get("error")),
            "regressions": regressions, "improvements": improvements, "library_changes": library,
            "value_changes": value_changes, "invalid": invalid,
            "missing": [] if allow_missing else sorted(set(oc) - set(nc)),
            "only_new": sorted(set(nc) - set(oc)), "incomplete": "finished" not in new["meta"],
            "error": new["meta"].get("error"), "compared": len(set(oc) & set(nc))}


def failed(r: dict) -> bool:
    return bool(r["toolchain_change"] or r.get("method_change") or r["regressions"] or r["improvements"]
                or r["library_changes"]
                or r["value_changes"] or r["invalid"] or r["missing"] or r["only_new"] or r["incomplete"]
                or r.get("old_incomplete"))


def format_report(r: dict) -> str:
    lines = [f"instruction counts: compared {r['compared']} kernels"]
    if r.get("old_incomplete"):
        lines.append("REFERENCE INCOMPLETE: the counts compared against did not finish; commit counts from a "
                     "run that did")
    if r.get("method_change"):
        a, b = r["method_change"]
        lines.append("COUNTING METHOD CHANGED; counts made another way measure something else, and what "
                     "follows says nothing about the code:")
        lines += [f"  was: {a}", f"  now: {b}"]
    if r["toolchain_change"]:
        lines.append("TOOLCHAIN CHANGED; the counts are not comparable, and what follows says nothing about "
                     "the code (count both with one toolchain, or pass --allow-toolchain-change):")
        lines += [f"  {k}: {a} -> {b}" for k, (a, b) in r["toolchain_change"].items()]
    if r.get("build_change"):
        lines.append("build configuration differs (part of the change; the counts below are compared):")
        lines += [f"  {k}: {a} -> {b}" for k, (a, b) in r["build_change"].items()]
    for title, key in (("REGRESSIONS", "regressions"),
                       ("IMPROVEMENTS: commit the new counts to benchmarks/perf/baselines/instructions.json",
                        "improvements")):
        lines.append(f"\n{title}: {len(r[key])}")
        for x in r[key]:
            lines.append(f"  {x['ratio']:6.3f}x  {x['old']:>14,} -> {x['new']:>14,}  {x['id']}")
    if r["library_changes"]:
        lines.append(f"\nLIBRARY INSTRUCTIONS CHANGED beyond the tolerance: {len(r['library_changes'])}")
        lines += [f"  {x['share']:+7.2%} of the whole  {x['old']:>12,} -> {x['new']:>12,}  {x['id']}"
                  for x in r["library_changes"]]
    if r["value_changes"]:
        lines.append(f"\nkernel values changed: {len(r['value_changes'])}")
        lines += [f"  {cid}: {a!r} -> {b!r}" for cid, a, b in r["value_changes"]]
    if r["invalid"]:
        lines.append(f"\nINVALID: {len(r['invalid'])} counts are not positive integers")
        lines += [f"  {cid}: {a!r} -> {b!r}" for cid, a, b in r["invalid"]]
    if r["missing"]:
        lines.append(f"\nMISSING: {len(r['missing'])} kernels counted before are not counted now "
                     "(pass --allow-missing when counting a subset)")
        lines += [f"  {cid}" for cid in r["missing"]]
    if r["only_new"]:
        lines.append(f"\nNEW: {len(r['only_new'])} kernels have no committed count; commit the new counts")
        lines += [f"  {cid}" for cid in r["only_new"]]
    if r["incomplete"]:
        lines.append("\nINCOMPLETE: the new count did not finish" + (f": {r['error']}" if r.get("error") else ""))
    return "\n".join(lines)
