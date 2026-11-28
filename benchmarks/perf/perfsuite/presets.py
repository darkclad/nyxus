"""Which cases each tier runs.

Tiers:
    smoke  a few minutes; small images; every suite touched once (the CI tier)
    full   about an hour on a 16-core workstation; scaling sweeps included
    large  32k x 32k 2D and 1024 x 1024 x 256 3D images; needs ~64 GB RAM and ~65 GB of disk

Suites:
    load      read + ROI build under the cheapest feature group, per container, dtype, in-RAM vs
              out-of-core (where nyxus supports it, see ooc_supported), thread count. In-RAM and
              out-of-core results are compared.
    groups    every 2D, 3D and image-quality group token on one mid-size dataset
    features  every feature code in the vetting registry, one at a time
    sweep     scaling sweeps: ROI size, ROI count, grey depth, Gabor kernel, threads, 3D ROI size
    gpu       the CUDA-offloaded groups, CPU against GPU
    profile   per-feature-group time breakdown from a CHECKTIMING CLI build (see checktiming.py)
    kernels   feature-class kernels alone on one synthetic ROI (see kernels.py)
"""

from __future__ import annotations

from . import registry
from .datasets import CONTAINERS_2D, CONTAINERS_3D, Dataset

TIERS = ("smoke", "full", "large")
SUITES = ("load", "groups", "features", "sweep", "gpu", "profile", "kernels")

GROUPS_2D = [
    "*ALL_INTENSITY*", "*ALL_MORPHOLOGY*", "*BASIC_MORPHOLOGY*", "*ALL_GLCM*", "*ALL_GLRLM*",
    "*ALL_GLDZM*", "*ALL_GLSZM*", "*ALL_GLDM*", "*ALL_NGLDM*", "*ALL_NGTDM*", "*ALL_NEIGHBOR*",
    "*ALL_IH*", "*GEOMOMS*", "*IGEOMOMS*", "*SGEOMOMS*", "GABOR", "*ALL*",
]
GROUPS_3D = [
    "*3D_ALL_INTENSITY*", "*3D_ALL_MORPHOLOGY*", "*3D_GLCM*", "*3D_GLDM*", "*3D_GLDZM*",
    "*3D_GLSZM*", "*3D_GLRLM*", "*3D_NGLDM*", "*3D_NGTDM*", "*3D_ALL_TEXTURE*", "*3D_ALL*",
]
GROUPS_IMQ = ["*ALL_IMQ*"]

# The groups with a CUDA implementation that the Python package can run on the GPU
# (docs/source/gpu.rst). GPU Gabor is not one: from Python it runs with an empty filter bank and
# raises or returns 0, so its GPU time comes from the profile suite's CLI GPU pass. GPU erosion
# only works with a moment feature in the same run (#520), so it is timed together with the shape
# moments, which have a case of their own to set it against.
GPU_SETS = {
    "erosion": ["EROSIONS_2_VANISH", "*SGEOMOMS*"],
    "imoments": ["*IGEOMOMS*"],
    "smoments": ["*SGEOMOMS*"],
}

TEXTURE_2D = ["*ALL_GLCM*", "*ALL_GLRLM*", "*ALL_GLDZM*", "*ALL_GLSZM*", "*ALL_GLDM*", "*ALL_NGLDM*", "*ALL_NGTDM*"]
# The cheapest group whose values still check in-RAM == out-of-core == every container.
LOAD_FEATURES = {"2d": ["*ALL_INTENSITY*"], "3d": ["*3D_ALL_INTENSITY*"]}

# A ROI goes out of core when its estimated footprint reaches ram_limit, which is set in whole
# megabytes. 0 sends every ROI out of core whatever its size; 1 MB would leave small grid cells
# (a 128 x 128 cell is about 0.3 MB) in RAM.
OOC_RAM_LIMIT_MB = 0
# The 2D out-of-core path costs about 20 microseconds per ROI pixel (a 2048 x 2048 image takes
# ~45 s against 0.3 s in RAM) and reads every tile of the image for each ROI, so 2D out-of-core
# cases stay on images of at most OOC_MAX_PIXELS_2D and datasets of at most OOC_MAX_ROIS, and
# every out-of-core case times one repeat. The 3D path streams at about 0.1 microseconds a voxel.
OOC_MAX_ROIS = 64
OOC_MAX_PIXELS_2D = 4096 * 4096
OOC_REPEATS = (0, 1)


def ooc_supported(ds: Dataset, container: str) -> bool:
    """Whether nyxus runs this dataset and container out of core rather than refusing it.

    Arrays passed to Nyxus.featurize are never processed out of core (the call raises), and a 2D
    whole-slide image whose footprint reaches ram_limit is refused; 3D whole volumes stream. A
    NIfTI volume is refused too: one read delivers every plane, so streaming cannot bound it.
    """
    if container in ("numpy", "nifti"):
        return False
    if ds.dim == 2 and (ds.layout == "whole" or ds.n_pixels > OOC_MAX_PIXELS_2D):
        return False
    return ds.expected_rois() <= OOC_MAX_ROIS

REPEATS = {"smoke": (0, 3), "full": (1, 5), "large": (0, 3)}  # (warm-up, timed)


def D(name: str) -> Dataset:
    return Dataset.parse(name)


def _label(features) -> str:
    return "+".join(f.strip("*").lower() or "x" for f in features)


def case(suite, api, ds, container, features, tier, *, params=None, mode="inram", tag="", sweep=None,
         digest=False, whole=None, repeats=None):
    params = dict(params or {})
    if mode == "ooc":
        params["ram_limit"] = OOC_RAM_LIMIT_MB
    threads = params.get("n_feature_calc_threads", 4)
    params.setdefault("n_feature_calc_threads", threads)
    warm, reps = REPEATS[tier]
    if repeats is not None:
        warm, reps = repeats
    parts = [suite, ds.name, container, mode, f"t{threads}", _label(features)]
    if tag:
        parts.append(tag)
    return {
        "id": "/".join(parts),
        "suite": suite,
        "api": api,
        "dataset": ds.name,
        "container": container,
        "features": list(features),
        "params": params,
        "mode": mode,
        "whole": ds.layout == "whole" if whole is None else whole,
        "warmup": warm,
        "repeats": reps,
        "digest": digest,
        "sweep": sweep,
    }


# ---------------------------------------------------------------------------------------------


def load_cases(tier):
    # The 2D datasets after the first carry the 2D out-of-core cases, which the big images cannot
    # (see OOC_MAX_PIXELS_2D); the 3D grid64 carries the out-of-core cases of the 3D grid512.
    if tier == "smoke":
        plan = [(D("2d-2048x2048-grid256-uint16-s0"), CONTAINERS_2D),
                (D("2d-1024x1024-grid64-uint16-s0"), CONTAINERS_2D),
                (D("3d-64x128x128-grid8-uint16-s0"), CONTAINERS_3D)]
        threads, dtype_sets = [1, 4], []
    elif tier == "full":
        plan = [(D(f"2d-8192x8192-{lay}-uint16-s0"), CONTAINERS_2D) for lay in ("grid4096", "giant", "whole")]
        plan += [(D(f"2d-4096x4096-{lay}-uint16-s0"), CONTAINERS_2D) for lay in ("grid64", "giant")]
        plan += [(D(f"3d-256x256x256-{lay}-uint16-s0"), CONTAINERS_3D) for lay in ("grid64", "giant", "whole")]
        threads = [1, 4, 16]
        dtype_sets = [D(f"2d-8192x8192-grid4096-{t}-s0") for t in ("uint8", "uint32", "float32")]
    else:
        plan = [(D(f"2d-32768x32768-{lay}-uint16-s0"), CONTAINERS_2D) for lay in ("grid50000", "giant", "whole")]
        plan += [(D(f"2d-4096x4096-{lay}-uint16-s0"), CONTAINERS_2D) for lay in ("grid64", "giant")]
        plan += [(D(f"3d-256x1024x1024-{lay}-uint16-s0"), CONTAINERS_3D) for lay in ("grid512", "grid64", "giant", "whole")]
        threads = [4, 16]
        dtype_sets = []
    out = []
    for ds, containers in plan:
        api = "2d" if ds.dim == 2 else "3d"
        for c in containers:
            for mode in ("inram", "ooc"):
                if mode == "ooc" and not ooc_supported(ds, c):
                    continue
                out.append(case("load", api, ds, c, LOAD_FEATURES[api], tier, mode=mode, digest=True,
                                repeats=OOC_REPEATS if mode == "ooc" else None))
    # thread scaling of the read path, on the first 2D dataset's tiled TIFF; the values must match
    # the 4-thread case above, so thread count is checked without a GPU too
    ds0 = plan[0][0]
    for t in threads:
        if t != 4:
            out.append(case("load", "2d", ds0, "tiff-tiled", LOAD_FEATURES["2d"], tier, digest=True,
                            params={"n_feature_calc_threads": t}, sweep={"name": "load-threads", "x": t}))
    for ds in dtype_sets:
        out.append(case("load", "2d", ds, "tiff-tiled", LOAD_FEATURES["2d"], tier, digest=True))
    return out


def group_cases(tier):
    d2 = {"smoke": "2d-2048x2048-grid256-uint16-s0", "full": "2d-4096x4096-grid1024-uint16-s0",
          "large": "2d-16384x16384-grid4096-uint16-s0"}[tier]
    d3 = {"smoke": "3d-64x128x128-grid8-uint16-s0", "full": "3d-128x256x256-grid27-uint16-s0",
          "large": "3d-256x512x512-grid64-uint16-s0"}[tier]
    ds2, ds3 = D(d2), D(d3)
    out = [case("groups", "2d", ds2, "tiff-tiled", [g], tier) for g in GROUPS_2D]
    # *ALL_BUT_GABOR* is a listed group name that the parser rejects (docs/source/troubleshooting.rst);
    # set subtraction gives the same selection
    out.append(case("groups", "2d", ds2, "tiff-tiled", ["*ALL*", "-GABOR"], tier))
    out += [case("groups", "3d", ds3, "ome-tiff", [g], tier) for g in GROUPS_3D]
    out += [case("groups", "imq", ds2, "tiff-tiled", [g], tier) for g in GROUPS_IMQ]
    return out


def feature_cases(tier):
    d2 = D({"smoke": "2d-1024x1024-grid64-uint16-s0"}.get(tier, "2d-2048x2048-grid256-uint16-s0"))
    d3 = D({"smoke": "3d-32x64x64-grid8-uint16-s0"}.get(tier, "3d-64x128x128-grid8-uint16-s0"))
    reps = (0, 1) if tier == "smoke" else (1, 3)
    seen, out = set(), []
    for row in registry.load():
        key = (row["dim"], row["feature"])
        if key in seen:
            continue
        seen.add(key)
        f = [row["feature"]]
        if row["dim"] == "2D":
            out.append(case("features", "2d", d2, "numpy", f, tier, repeats=reps, tag=row["family"]))
        elif row["dim"] == "3D":
            out.append(case("features", "3d", d3, "ome-tiff", f, tier, repeats=reps, tag=row["family"]))
        elif row["dim"] == "IMQ":
            out.append(case("features", "imq", d2, "tiff-tiled", f, tier, repeats=reps, tag=row["family"]))
    return out


def sweep_cases(tier):
    out = []
    if tier == "smoke":
        ds = D("2d-2048x2048-grid256-uint16-s0")
        for t in (1, 4):
            out.append(case("sweep", "2d", ds, "tiff-tiled", ["*ALL*", "-GABOR"], tier,
                            params={"n_feature_calc_threads": t}, sweep={"name": "threads", "x": t}))
        return out
    if tier == "large":
        return out  # the full tier's sweeps already reach the sizes that set the exponents

    # ROI size at fixed count (16 ROIs): x = pixels per grid cell
    for side in (512, 1024, 2048, 4096, 8192):
        ds = D(f"2d-{side}x{side}-grid16-uint16-s0")
        for fs in (["GABOR"], ["EROSIONS_2_VANISH"], ["*GEOMOMS*"], ["*ALL_GLCM*"], ["*ALL_NGLDM*"],
                   ["*ALL_MORPHOLOGY*"], ["*ALL_INTENSITY*"], ["ZERNIKE2D"]):
            out.append(case("sweep", "2d", ds, "tiff-tiled", fs, tier,
                            sweep={"name": f"roi-size/{_label(fs)}", "x": ds.n_pixels // 16}))
    # ROI count at fixed cell size (64 x 64): x = ROI count
    for n, side in ((16, 256), (256, 1024), (4096, 4096), (16384, 8192), (65536, 16384)):
        ds = D(f"2d-{side}x{side}-grid{n}-uint16-s0")
        for fs in (["*ALL_INTENSITY*"], ["*ALL_GLCM*"], ["*ALL_NEIGHBOR*"], ["*BASIC_MORPHOLOGY*"]):
            out.append(case("sweep", "2d", ds, "tiff-tiled", fs, tier,
                            sweep={"name": f"roi-count/{_label(fs)}", "x": n}))
    # grey depth
    ds = D("2d-4096x4096-grid1024-uint16-s0")
    for g in (8, 16, 32, 64, 128, 256):
        for fs in TEXTURE_2D:
            out.append(case("sweep", "2d", ds, "tiff-tiled", [fs], tier, params={"coarse_gray_depth": g},
                            tag=f"g{g}", sweep={"name": f"gray-depth/{_label([fs])}", "x": g}))
    # Gabor kernel size
    ds = D("2d-2048x2048-grid256-uint16-s0")
    for k in (8, 16, 32, 64):
        out.append(case("sweep", "2d", ds, "tiff-tiled", ["GABOR"], tier, params={"gabor_kersize": k},
                        tag=f"k{k}", sweep={"name": "gabor-kersize", "x": k}))
    # feature-calculation threads
    ds = D("2d-4096x4096-grid1024-uint16-s0")
    for t in (1, 2, 4, 8, 16, 32):
        out.append(case("sweep", "2d", ds, "tiff-tiled", ["*ALL*"], tier, params={"n_feature_calc_threads": t},
                        sweep={"name": "threads/all", "x": t}))
    # 3D ROI size at fixed count (8 ROIs): x = voxels per cell
    for side in (32, 64, 128, 256):
        ds = D(f"3d-{side}x{side}x{side}-grid8-uint16-s0")
        for fs in (["*3D_ALL_INTENSITY*"], ["*3D_ALL_MORPHOLOGY*"], ["*3D_ALL_TEXTURE*"]):
            out.append(case("sweep", "3d", ds, "ome-tiff", fs, tier,
                            sweep={"name": f"3d-roi-size/{_label(fs)}", "x": ds.n_pixels // 8}))
    return out


def gpu_cases(tier):
    names = {"smoke": ["2d-2048x2048-grid256-uint16-s0"],
             "full": ["2d-4096x4096-grid1024-uint16-s0", "2d-8192x8192-grid64-uint16-s0"],
             "large": ["2d-16384x16384-grid4096-uint16-s0", "2d-16384x16384-grid256-uint16-s0"]}[tier]
    out = []
    for n in names:
        ds = D(n)
        for key, fs in GPU_SETS.items():
            for dev, t in ((-1, 1), (-1, 16), (0, 16)):
                out.append(case("gpu", "2d", ds, "tiff-tiled", fs, tier,
                                params={"use_gpu_device": dev, "n_feature_calc_threads": t},
                                tag="gpu" if dev >= 0 else "cpu", digest=True))
    return out


BUILDERS = {"load": load_cases, "groups": group_cases, "features": feature_cases, "sweep": sweep_cases,
            "gpu": gpu_cases}


def build(tier: str, suites) -> list:
    out = []
    for s in suites:
        if s in BUILDERS:
            out.extend(BUILDERS[s](tier))
    ids = [c["id"] for c in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise AssertionError(f"duplicate case ids: {sorted(dup)[:5]}")
    return out


def profile_datasets(tier):
    return {"smoke": ["2d-2048x2048-grid256-uint16-s0"],
            "full": ["2d-4096x4096-grid1024-uint16-s0", "2d-8192x8192-grid64-uint16-s0"],
            "large": ["2d-16384x16384-grid4096-uint16-s0"]}[tier]
