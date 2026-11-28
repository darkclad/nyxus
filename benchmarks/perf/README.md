# Nyxus performance suite

Measures how long Nyxus takes, and how much memory it uses, to load images and compute features,
on seeded synthetic data that is identical on every machine. It answers three questions:

- **Load:** what does reading an image and its mask and building the ROIs cost, per container
  (tiled TIFF, striped TIFF, OME-TIFF, OME-Zarr, NIfTI, numpy array), per pixel type, as images
  grow to 32k x 32k and 1024 x 1024 x 256, and out of core (to 4096 x 4096 in 2D, see Datasets)?
- **Features:** what does every feature cost: each group token, each of the ~760 feature codes in
  the vetting registry, and the hot spots as ROI size, ROI count, grey depth, Gabor kernel size and
  thread count grow? CPU against GPU for the CUDA-offloaded groups?
- **Regressions:** is a build slower, or hungrier, than a baseline?

The cases are also a correctness check. Cases that differ only in how the pixels reach the
features (container, in-RAM or out-of-core, CPU or GPU, thread count) must produce the same feature
values, and `run` reports any feature whose per-ROI sum differs between them. Differences the
Nyxus docs state are left out: `COVERED_IMAGE_INTENSITY_RANGE` for numpy arrays, and the central,
normalized, Hu and weighted moment columns on the GPU (`docs/source/gpu.rst`). The `gpu` suite
leaves out GPU Gabor, which the Python package cannot run on the GPU (the `profile` suite's CLI GPU
pass times it), and times GPU erosion together with the shape moments, without which it returns 0.

## Layers

| suite | what is timed | how |
|---|---|---|
| `load` | read + ROI build + the cheapest feature group | `featurize_directory` / `featurize` per container, in-RAM and out-of-core |
| `groups` | each 2D, 3D and image-quality group token | one mid-size dataset per dimensionality |
| `features` | every feature code in `tests/vetting/oracle_coverage.csv`, alone | numpy input (2D), OME-TIFF (3D) |
| `sweep` | scaling with ROI size, ROI count, grey depth, Gabor kernel, threads, 3D ROI size | `report` fits the exponent of each |
| `gpu` | erosion (with the shape moments) and the two moment groups, CPU (1 and 16 threads) against GPU | values compared too |
| `profile` | per-feature-group time inside one `*ALL*` run | CLI built with `-DCHECKTIMING=ON` |
| `kernels` | feature classes alone on one synthetic ROI, no I/O, no threads | `bench_feature_kernels` |

Every Python case runs in its own process, so a crash, a leak or a process-wide setting such as
`ram_limit` stays with its case. The per-feature suite batches 64 cases per process (process start
would otherwise dominate) and restarts after a case that kills its worker, recording it as
`crashed`. A monitor process samples the worker's resident set about every 5 ms, giving each repeat
its own peak. It is a separate process because Nyxus holds Python's interpreter lock for the whole
featurize call, so a thread inside the worker could not sample during it.

## Tiers

| tier | images | cases | time (16-core workstation) | disk |
|---|---|---|---|---|
| `smoke` | 1k-2k 2D, 64 x 128 x 128 3D | ~820 | a few minutes | < 1 GB |
| `full` | up to 16k 2D (sweeps), 256^3 3D | ~990 | about an hour | ~8 GB |
| `large` | 32k x 32k 2D, 256 x 1024 x 1024 3D | ~860 | several hours | ~65 GB, 64+ GB RAM |

Timed repeats are preceded by a warm-up in `full`, so file reads are warm-cache reads.

## Build

The Python suites time whichever `nyxus` the interpreter imports. For the GPU suite it must be a
CUDA build. For `profile` and `kernels`, build the CLI and the benchmarks too:

```bash
cmake -S . -B build-perf -DBUILD_CLI=ON -DCHECKTIMING=ON -DRUN_BENCHMARKS=ON -DUSEGPU=ON -DALLEXTRAS=ON
cmake --build build-perf -j
```

`CHECKTIMING` adds per-group stopwatches and writes `inclusive_nyxustiming.csv`; it slows a run
slightly, so time the plain build with the Python suites and use the CLI only for the breakdown.
`profile` adds a GPU pass when the CLI itself is a CUDA build (its `--help` lists `--useGpu`) and
the machine has a CUDA device (the Python module sees one, or `nvidia-smi -L` lists one).

Instruction counts (see Baselines) compare only with counts from the same build, so they take the
build the committed counts were made with, the one the CI job makes: Linux, Ubuntu 24.04's gcc,
CPU only, no extras.

```bash
sudo apt-get install build-essential cmake ninja-build libtiff-dev libdeflate-dev valgrind
cmake -G Ninja -B build-bench -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DBUILD_LIB=OFF \
      -DNOEXTRAS=ON -DUSEGPU=OFF -DCHECKTIMING=OFF -DRUN_BENCHMARKS=ON
ninja -C build-bench bench_feature_kernels
```

The counted build is the code as users build it, without CHECKTIMING (its stopwatches would be
counted too); the CHECKTIMING CLI the `profile` suite needs is a separate build.

On Windows, Python does not search `PATH` for an extension's DLLs. The suite registers
`%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin` before importing nyxus, plus any directories listed in
`PERFSUITE_DLL_DIRS`.

Python dependencies: `numpy`, `pandas`, `psutil`, `tifffile`, `zarr` (3.x), plus `scipy` and
`pytest` for the harness tests.

## Run

From `benchmarks/perf`:

```bash
python -m perfsuite list --tier full                       # what would run
python -m perfsuite run  --tier smoke --data <data-dir> --out results/smoke.json \
       --cli ../../build-perf/nyxus --bench-exe ../../build-perf/bench_feature_kernels
python -m perfsuite run  --tier full --suite sweep --data <data-dir> --out results/sweep.json
python -m perfsuite report  results/smoke.json > results/smoke.md
python -m perfsuite compare results/base.json results/smoke.json   # exit 1 on regressions
python -m perfsuite baseline save results/smoke.json              # this host's timing baseline
python -m perfsuite compare results/smoke.json                     # ... and a run against it
python -m perfsuite icount --bench-exe ../../build-bench/bench_feature_kernels --build-dir ../../build-bench \
       --out results/icount.json                                    # Linux
python -m perfsuite compare baselines/instructions.json results/icount.json
```

`--suite` repeats; `--filter` takes a regular expression on case ids and applies to every suite:
profile rows are matched as `profile/<dataset>/<cpu|gpu>/<group>` (a run happens when the filter
matches any row it can produce: the 2D stopwatches named in the Nyxus sources, the GPU ones on the
`gpu` pass only), kernels as `kernels/<kernel>/side<N>`.
`list --bench-exe <bench_feature_kernels>` names the kernels cases; without it `list` shows one line
per side. Datasets are generated into `--data`, only for the cases that will run (a suite skipped
for want of a GPU or a `--cli` generates nothing), and reused while their recipe is unchanged;
`list` estimates their size uncompressed. Logs go to `<results stem>-logs/` next to the results
file, so keep results under `results/`, which git ignores: an untracked file elsewhere in the
checkout marks the next run's `git_dirty`.

## Results

One JSON file per run. `meta` records the host, CPU, cores, RAM, GPU, OS, nyxus version and module
path, git SHA and whether the tree was dirty. `cases` has, per case: its id and parameters,
`status` (`ok`, `error`, `crashed`, `timeout`), every timed repeat in seconds with
median/MAD/min/max, peak RSS in MB, its growth over the worker's resident set when the case
started (`delta_rss_mb`, 0 when no repeat rose above it), whether the case was the first in its
worker (`fresh_worker`), output rows and columns, and for value-checked cases the per-feature sums
over the ROIs, plain and weighted by ROI label (so values given to the wrong labels disagree).
A batched per-feature case starts from whatever the cases before it left resident, so its peak and
its growth both depend on what ran before it; `report` marks such peaks with `*`.

`compare` calls a case a regression when its median is more than 15 % slower, the difference
exceeds three times the larger MAD, and it exceeds 20 ms (`--floor`; 0.5 ms for `kernels`,
`--kernel-floor`). A peak RSS rising
by more than 20 % and 16 MB is a regression too, judged only where the case was the first in its
worker in both runs (every case outside the batched per-feature suite, unless `--batch` was
given). A case that stopped being `ok` is a regression, and so is one that only the new run has
and that is not `ok` (a kernels run that crashed, timed out or ran out of its time budget leaves a
`kernels/RUN` row of that kind). `compare` also exits 1 when a kernel's reported value changed, when
the new run did not finish, and when it lacks cases the baseline has as `ok`; pass `--allow-missing`
to compare a `--filter` or `--suite` subset against a full baseline. Comparing runs from different
hosts prints a warning: timings only compare on the same machine. Even on one machine, two smoke
runs of the same build can differ by more than 15 % on some cases, so judge timings on `full`,
which repeats more, or run baseline and candidate back to back.

## Baselines

There are two, because a timing depends on the machine and an instruction count does not.

**Timing baselines stay on their host.** `baseline save <results.json>` keeps a run as its host's
baseline for its tier, in `$PERFSUITE_BASELINES` (default `~/.nyxus-perf/baselines`, outside the
checkout), as `<host>/<tier>.json`; the baseline it replaces is kept as `<host>/prev/<tier>.json`.
A run that did not finish, has a case that is not `ok`, or came from a checkout with uncommitted
changes (or one git could not read) is refused unless `--force`. `compare <results.json>` with one
file judges it against its host's baseline, which must name the same host and tier itself;
`baseline list` shows what is stored. Results and timing baselines are never committed.

**Instruction counts are committed.** `icount` runs each `bench_feature_kernels` kernel at sides 64
and 256 once under `valgrind --tool=callgrind`, with one repetition, no warm-up and no timing,
collecting only inside the benchmark's `bench_kernel()`: the kernel's call on a prepared ROI.
Process start, data generation, dependency preparation and building the ROI (whose cost grows with
every feature code Nyxus registers) are not counted, so a count changes only with its kernel.
Timing is off because it does not repeat: a clock read can retry, and printing times
that differ costs a different number of instructions. The valgrind runs are independent and run
one per CPU at once (`--jobs`); when one fails, no further run starts (those already running
finish), and the counts that finished are still written, marked unfinished, with the first
failure in plan order. The counts record the commit they were made from and whether the checkout
differed from it.

Two counts are kept per kernel. `instructions` are those executed in the benchmark executable
itself (the Nyxus code and the library templates compiled into it). They are the same on every
x86-64 CPU and repeat exactly: told the CPU lacks AVX2, FMA, BMI2 and ERMS
(`GLIBC_TUNABLES=glibc.cpu.hwcaps=...`), glibc picks other routines and the whole process's count
moves by 0.8 to 2 %, while the executable's count of the Gabor, GLCM and erosion kernels stays the
same to the instruction. `library_instructions` are those executed in the shared libraries the
repetition calls (libm, libc, libstdc++), so work moved there (an `exp()` called twice as often)
shows. glibc picks those routines by the CPU, so they are counted with `GLIBC_TUNABLES` making the
choice the same on every x86-64 CPU (the baseline instruction set, fixed preferences, fixed cache
thresholds), and judged with a tolerance taken as a share of the kernel's whole count
(`--icount-lib-tol`, default 0.5 %) for what a tunable cannot reach. Twice the `exp()` calls in
Gabor is 2 % of its whole count.

`baselines/instructions.json` holds the counts, the kernel values, and what they depend on besides
the code (`toolchain`), in two kinds, and how they were made (`method`: the collection boundary,
the valgrind options, the glibc pin). The toolchain proper is the environment: the compiler, the
exact versions (from dpkg) of every installed gcc, g++ and cpp, their libraries, binutils (the
assembler and linker), the C library, cmake and valgrind, the architecture and the valgrind
version. The build configuration is part of the change being judged: the build type, a hash of
every compile command of the Nyxus code and the benchmark (flags, standard, definitions, include
paths, PIC, from the build's `compile_commands.json`, which `--build-dir` names and the benchmark
must come from) and of the link settings (linker flags and IPO, from the CMake cache and the
benchmark's link line), and GPU and CHECKTIMING on or off. The CPU features, and the commit the
counts were made from and whether its checkout was modified, are recorded for reference. The
counts are made with the build above, the one the `Build_and_Run_Benchmarks` CI job makes and
counts again on every pull request; the job's `instruction-counts` artifact names the commit it
counted.

`compare` fails when an `instructions` count changes (`--icount-tol`, default 0; a fall past
`1 / (1 + tol)` mirrors a rise past `1 + tol`): a rise is a regression, and a fall means the
committed counts are out of date, which left stale would let a later rise pass. It also fails when
library instructions change beyond their tolerance, when a kernel value changes, when a count is
not positive, when a kernel has no committed count (a new one), and when a kernel is no longer
counted (`--allow-missing` for a `--filter` or `--side` subset). Counts made with a different
toolchain are not compared: `compare` exits 3 (`--allow-toolchain-change` compares anyway).
Counts made with another build configuration are compared, and the difference is reported with
them: a change to the compile flags is judged by what it does to the counts.

The CI job pins its toolchain: it runs in an Ubuntu 24.04 image fixed by digest and installs every
package from Ubuntu's archive as it stood at the job's `SNAPSHOT` date, which serves exactly the
packages the committed counts were made with (`apt-get update --snapshot 20261127T000000Z` lists
gcc-13 13.3.0-6ubuntu2~24.04.1, libc6 2.39-0ubuntu8.12, libstdc++6 14.2.0-4ubuntu2~24.04.1 and
valgrind 3.22.0-0ubuntu3, the versions in their `toolchain`; an apt pin brings every installed
package to the snapshot's version), and the harness's Python packages from `requirements-ci.txt`,
the complete set, installed with `--no-deps --require-hashes`. Updating the toolchain is a change
to `SNAPSHOT` (and the image, or `requirements-ci.txt`) together with new counts. When the toolchain
differs from the committed counts' all the same (they were made elsewhere, or the pins moved without
them), the job fails: such counts say nothing about the change. To update the counts, replace
`baselines/instructions.json` with the job's `instruction-counts` artifact; for a pull request it
names both the merge commit counted (`git_sha`) and the pull request's head (`source_sha`). To make
counts locally with the same toolchain, build in that image with the snapshot configured as the job
does (`APT::Snapshot` and the pin), so that every package comes from it; an Ubuntu 24.04 machine
with other package versions records another toolchain.

## Datasets

Names encode the recipe, for example `2d-4096x4096-grid1024-uint16-s0`: dimensionality, size,
layout, pixel type, seed. Layouts:

- `gridN`: N ROIs on a grid. Each is a blob whose radius is modulated by two harmonics, so ROIs
  are non-convex and hull, Feret and contour features have real work to do.
- `giant`: one ROI covering most of the image.
- `whole`: no mask; the whole image is one ROI.

Intensities are a smooth field plus seeded noise. Float images hold [0, 1] and are featurized with
`min_intensity=0`, `max_intensity=1`. A mask with more than 65,535 labels is written as uint32.

Out-of-core cases set `ram_limit=0`, which sends every ROI out of core whatever its size. They time one
repeat, on file containers and datasets of at most 64 ROIs, and in 2D on images of at most
4096 x 4096: the 2D out-of-core path costs about 20 microseconds per ROI pixel (a 2048 x 2048
image takes ~45 s against 0.3 s in RAM), where the 3D path streams at about 0.1 microseconds a
voxel. They are left out where nyxus refuses rather than goes out of core:
numpy arrays, 2D images without a mask, and NIfTI volumes (one read delivers every plane). A
harness test checks that these cases do go out of core: at `verbose=2` nyxus logs each oversized
ROI, and it must log every ROI out of core and none in RAM.

## Harness tests

```bash
python -m pytest benchmarks/perf/tests
```

The end-to-end tests featurize tiny datasets through real worker processes, and are skipped when
`nyxus` cannot be imported. The tests of `bench_feature_kernels` itself run when
`PERFSUITE_BENCH_EXE` names a built one. The Ubuntu CI workflow runs these tests, and
`tests/vetting/check_test_names.py` holds them to the repository's test-naming rules.
