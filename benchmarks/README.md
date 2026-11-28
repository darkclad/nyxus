# Nyxus Benchmarks

This directory contains small, dependency-light microbenchmarks for performance-sensitive feature code,
and [`perf/`](perf/README.md), the end-to-end performance suite: loading in every container, in RAM and
out of core, every feature group and feature code, scaling sweeps, CPU against GPU, and comparison
against a baseline.

## Feature kernels

`bench_feature_kernels` times the costliest 2D feature classes on one synthetic ROI per size, with no
image loading, ROI gathering or thread pool: contour, convex hull, erosion, Gabor, intensity and shape
moments, Zernike, GLCM, GLRLM, GLSZM, GLDZM and NGLDM. The ROI is a non-convex blob over a seeded
texture. A kernel's dependencies (contour, convex hull) are computed before timing and not timed.
Each row reports the median, MAD, minimum and maximum time, and one value the kernel produced, so a
timing change that comes with a value change is visible. Each kernel has a largest side, past which
its cost grows too fast; `--help` lists the kernels with that side, and larger sides are skipped. An
unknown `--kernel` name is refused. `--greydepth` takes any coarse grey depth Nyxus takes: N > 0
bins to N levels, 0 does not bin, N < 0 bins to |N| PyRadiomics-style bins.

```bash
cmake -S . -B build-bench -DRUN_BENCHMARKS=ON -DNOEXTRAS=ON
cmake --build build-bench --target bench_feature_kernels -j
./build-bench/bench_feature_kernels                          # sides 64..2048, every kernel
./build-bench/bench_feature_kernels --kernel gabor --side 512 --greydepth 64
```

## Convex Hull

`bench_convex_hull` measures the 2D convex hull path used by `ConvexHullFeature::calculate()`.
It intentionally reports both timing and output shape (`hull_size`, `hull_area`, `solidity`) so performance comparisons can also catch ordering-sensitive correctness regressions.

Build:

```bash
cmake -S . -B build-bench -DRUN_BENCHMARKS=ON -DNOEXTRAS=ON
cmake --build build-bench --target bench_convex_hull -j
```

Run the default workload:

```bash
./build-bench/bench_convex_hull
```

Run a focused workload:

```bash
./build-bench/bench_convex_hull --cycles 5 --case y_major_raster --side 1024
```

Input-order cases:

- `x_major_sorted`: already sorted by the convex hull comparator (`x`, then `y`).
- `y_major_raster`: row-major scan order, matching the usual trivial ROI pixel ingestion path.
- `reversed`: reverse of the sorted comparator order.
- `shuffled`: deterministic pseudorandom order.

For before/after comparisons, build the same target in two clean worktrees and compare median times for the same case/side/cycle settings.
