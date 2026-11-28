"""Seeded synthetic image/mask datasets written to every container Nyxus reads.

A dataset is named by its spec, so a name identifies the same pixels on every machine:

    2d-4096x4096-grid1024-uint16-s0     2D, 4096x4096, 1024 ROIs on a grid, uint16 intensities
    2d-8192x8192-giant-uint16-s0        one ROI covering most of the image
    2d-8192x8192-whole-uint16-s0        no mask: the whole image is one ROI
    3d-128x256x256-grid27-uint16-s0     3D, Z x Y x X

Intensities are a smooth field plus seeded noise, so texture features see a non-degenerate grey
level distribution. Grid ROIs are blobs whose radius is modulated by two harmonics, which makes them
non-convex: hull, Feret and contour features do real work instead of meeting a disk.

Each block of BLOCK rows (2D) or slices (3D) draws its noise from its own seeded stream, so the pixels
do not depend on the machine. A block is computed in pieces of at most CHUNK_VALUES pixels, drawing
that stream in order, so the temporaries stay near CHUNK_VALUES x 8 bytes whatever the image size and
peak memory is the output arrays plus that.

Layout on disk, per dataset and container:

    <root>/<dataset>/<container>/int/img.<ext>
    <root>/<dataset>/<container>/seg/img.<ext>       (absent for layout 'whole')
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

BLOCK = 1024  # rows (2D) or slices (3D) per noise stream; fixed, the pixels depend on it
CHUNK_VALUES = 1 << 24  # pixels computed at a time; the pixels do not depend on it

CONTAINERS_2D = ("tiff-tiled", "tiff-striped", "ome-tiff", "ome-zarr", "numpy")
CONTAINERS_3D = ("ome-tiff", "ome-zarr", "nifti")

EXT = {
    "tiff-tiled": "tif",
    "tiff-striped": "tif",
    "ome-tiff": "ome.tif",
    "ome-zarr": "ome.zarr",
    "numpy": "npy",
    "nifti": "nii",
}

# Intensity range each dtype is filled with. Float images carry [0, 1] and are featurized with
# min_intensity=0, max_intensity=1 (see intensity_params).
RANGES = {
    "uint8": (1, 250),
    "uint16": (100, 60000),
    "uint32": (100, 4_000_000),
    "float32": (0.0, 1.0),
}

_NAME = re.compile(
    r"^(?P<dim>[23])d-(?P<shape>\d+(?:x\d+)+)-(?P<layout>grid\d+|giant|whole)-(?P<dtype>\w+)-s(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class Dataset:
    dim: int
    shape: tuple  # (Y, X) or (Z, Y, X)
    layout: str  # 'grid' | 'giant' | 'whole'
    n_rois: int = 0  # grid only
    dtype: str = "uint16"
    seed: int = 0

    @property
    def name(self) -> str:
        lay = f"grid{self.n_rois}" if self.layout == "grid" else self.layout
        return f"{self.dim}d-{'x'.join(map(str, self.shape))}-{lay}-{self.dtype}-s{self.seed}"

    @property
    def has_mask(self) -> bool:
        return self.layout != "whole"

    @property
    def mask_dtype(self) -> str:
        return "uint16" if self.expected_rois() <= 65535 else "uint32"

    @property
    def n_pixels(self) -> int:
        return int(np.prod(self.shape))

    def expected_rois(self) -> int:
        return self.n_rois if self.layout == "grid" else 1

    @staticmethod
    def parse(name: str) -> "Dataset":
        m = _NAME.match(name)
        if not m:
            raise ValueError(f"not a dataset name: {name!r}")
        shape = tuple(int(v) for v in m["shape"].split("x"))
        dim = int(m["dim"])
        if len(shape) != dim:
            raise ValueError(f"{name!r}: {dim}D dataset needs {dim} sizes")
        if min(shape) < 1:
            raise ValueError(f"{name!r}: every size must be at least 1")
        lay = m["layout"]
        n = int(lay[4:]) if lay.startswith("grid") else 0
        if lay.startswith("grid") and n < 1:
            raise ValueError(f"{name!r}: a grid needs at least one ROI")
        if m["dtype"] not in RANGES:
            raise ValueError(f"{name!r}: dtype must be one of {sorted(RANGES)}")
        return Dataset(dim, shape, "grid" if n else lay, n, m["dtype"], int(m["seed"]))


def intensity_params(ds: Dataset) -> dict:
    """Constructor kwargs a float dataset needs so its [0, 1] range is quantized as intended."""
    if ds.dtype == "float32":
        return {"min_intensity": 0.0, "max_intensity": 1.0, "dynamic_range": 10000}
    return {}


# ---------------------------------------------------------------------------------------------
# pixel generators


def _intensity_rows(ds: Dataset, start: int, stop: int, rng) -> np.ndarray:
    """Rows [start, stop) of a 2D image, or slices [start, stop) of a 3D one, drawing noise from `rng`."""
    lo, hi = RANGES[ds.dtype]
    if ds.dim == 2:
        H, W = ds.shape
        y = np.arange(start, stop, dtype=np.float32)[:, None] / H
        x = np.arange(W, dtype=np.float32)[None, :] / W
        field = (
            0.35 * np.sin(2 * np.pi * (3 * x + 2 * y))
            + 0.25 * np.cos(2 * np.pi * (7 * y - 5 * x))
            + 0.15 * np.sin(2 * np.pi * 41 * x) * np.cos(2 * np.pi * 37 * y)
        )
        shape = (stop - start, W)
    else:
        D, H, W = ds.shape
        z = np.arange(start, stop, dtype=np.float32)[:, None, None] / D
        y = np.arange(H, dtype=np.float32)[None, :, None] / H
        x = np.arange(W, dtype=np.float32)[None, None, :] / W
        field = (
            0.35 * np.sin(2 * np.pi * (3 * x + 2 * y + z))
            + 0.25 * np.cos(2 * np.pi * (5 * z - 4 * x + 3 * y))
            + 0.15 * np.sin(2 * np.pi * 23 * x) * np.cos(2 * np.pi * 19 * y)
        )
        shape = (stop - start, H, W)
    v = 0.5 + 0.5 * (field / 0.75)  # roughly [0, 1]
    v = v + rng.normal(0.0, 0.08, size=shape).astype(np.float32)
    np.clip(v, 0.0, 1.0, out=v)
    if ds.dtype == "float32":
        return v.astype(np.float32)
    return (lo + v * (hi - lo)).astype(ds.dtype)


def make_intensity(ds: Dataset) -> np.ndarray:
    out = np.empty(ds.shape, dtype=ds.dtype)
    n = ds.shape[0]
    step = max(1, CHUNK_VALUES // (ds.n_pixels // n))  # rows or slices per piece
    for s in range(0, n, BLOCK):
        e = min(n, s + BLOCK)
        rng = np.random.default_rng([ds.seed, s // BLOCK, 7])
        for a in range(s, e, step):
            b = min(e, a + step)
            out[a:b] = _intensity_rows(ds, a, b, rng)
    return out


def _blob_radius(theta, r0, a, b):
    return r0 * (1.0 + 0.22 * np.sin(3 * theta + a) + 0.08 * np.sin(7 * theta + b))


def _grid_counts(ds: Dataset):
    """Cells per axis: the smallest near-cubic grid holding n_rois cells."""
    k = ds.dim
    g = max(1, math.ceil(ds.n_rois ** (1.0 / k) - 1e-9))
    while g ** k < ds.n_rois:
        g += 1
    return g


def make_mask(ds: Dataset) -> np.ndarray | None:
    if not ds.has_mask:
        return None
    mask = np.zeros(ds.shape, dtype=ds.mask_dtype)
    rng = np.random.default_rng([ds.seed, 11])
    if ds.layout == "giant":
        _giant(ds, mask, rng)
        return mask
    g = _grid_counts(ds)
    cells = [s // g for s in ds.shape]
    if min(cells) < 4:
        raise ValueError(f"{ds.name}: {ds.n_rois} ROIs leave cells under 4 px; use a larger image")
    phases = rng.uniform(0, 2 * np.pi, size=(ds.n_rois, 3))
    if ds.dim == 2:
        cy, cx = cells
        yy, xx = np.mgrid[0:cy, 0:cx].astype(np.float32)
        dy, dx = yy - (cy - 1) / 2, xx - (cx - 1) / 2
        rho = np.hypot(dy, dx)
        th = np.arctan2(dy, dx)
        r0 = 0.38 * min(cy, cx)
        for lab in range(1, ds.n_rois + 1):
            i, j = divmod(lab - 1, g)
            a, b, _ = phases[lab - 1]
            inside = rho <= _blob_radius(th, r0, a, b)
            mask[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx][inside] = lab
    else:
        cz, cy, cx = cells
        zz, yy, xx = np.mgrid[0:cz, 0:cy, 0:cx].astype(np.float32)
        dz, dy, dx = zz - (cz - 1) / 2, yy - (cy - 1) / 2, xx - (cx - 1) / 2
        rho = np.sqrt(dz * dz + dy * dy + dx * dx)
        th = np.arctan2(dy, dx)
        ph = np.arccos(np.clip(dz / np.maximum(rho, 1e-6), -1, 1))
        r0 = 0.38 * min(cz, cy, cx)
        for lab in range(1, ds.n_rois + 1):
            k, rem = divmod(lab - 1, g * g)
            i, j = divmod(rem, g)
            a, b, c = phases[lab - 1]
            r = _blob_radius(th, r0, a, b) * (1.0 + 0.1 * np.sin(2 * ph + c))
            inside = rho <= r
            mask[k * cz:(k + 1) * cz, i * cy:(i + 1) * cy, j * cx:(j + 1) * cx][inside] = lab
    return mask


def _giant(ds: Dataset, mask: np.ndarray, rng) -> None:
    a, b, c = rng.uniform(0, 2 * np.pi, size=3)
    n = ds.shape[0]
    if ds.dim == 2:
        H, W = ds.shape
        r0 = 0.42 * min(H, W)
        x = np.arange(W, dtype=np.float32)[None, :] - (W - 1) / 2
        for s in range(0, n, BLOCK):
            e = min(n, s + BLOCK)
            y = np.arange(s, e, dtype=np.float32)[:, None] - (H - 1) / 2
            inside = np.hypot(y, x) <= _blob_radius(np.arctan2(y, x), r0, a, b)
            mask[s:e][inside] = 1
    else:
        D, H, W = ds.shape
        r0 = 0.42 * min(D, H, W)
        y = np.arange(H, dtype=np.float32)[:, None] - (H - 1) / 2
        x = np.arange(W, dtype=np.float32)[None, :] - (W - 1) / 2
        rxy = np.hypot(y, x)
        th = np.arctan2(y, x)
        for zi in range(D):
            z = zi - (D - 1) / 2
            rho = np.sqrt(rxy * rxy + z * z)
            ph = np.arccos(np.clip(z / np.maximum(rho, 1e-6), -1, 1))
            inside = rho <= _blob_radius(th, r0, a, b) * (1.0 + 0.1 * np.sin(2 * ph + c))
            mask[zi][inside] = 1


# ---------------------------------------------------------------------------------------------
# container writers


def _tile(n: int) -> int:
    return max(16, min(1024, (n // 16) * 16))


def _write_tiff(path: Path, arr: np.ndarray, kind: str) -> None:
    import tifffile

    big = arr.nbytes > 2**31 - 2**24
    if kind == "tiff-striped":
        tifffile.imwrite(path, arr, rowsperstrip=64, bigtiff=big, photometric="minisblack")
        return
    tile = (_tile(arr.shape[-2]), _tile(arr.shape[-1]))
    if kind == "ome-tiff":
        axes = "YX" if arr.ndim == 2 else "ZYX"
        tifffile.imwrite(path, arr, tile=tile, ome=True, bigtiff=big, metadata={"axes": axes},
                         photometric="minisblack")
    else:
        tifffile.imwrite(path, arr, tile=tile, bigtiff=big, photometric="minisblack")


def _write_zarr(path: Path, arr: np.ndarray) -> None:
    import zarr

    shutil.rmtree(path, ignore_errors=True)
    if arr.ndim == 2:
        chunks = (min(1024, arr.shape[0]), min(1024, arr.shape[1]))
        axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    else:
        chunks = (1, min(1024, arr.shape[1]), min(1024, arr.shape[2]))
        axes = [{"name": a, "type": "space"} for a in "zyx"]
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    a = g.create_array("0", shape=arr.shape, dtype=arr.dtype, chunks=chunks, compressors=None)
    step = chunks[0] if arr.ndim == 2 else 16
    for s in range(0, arr.shape[0], step):
        a[s:s + step] = arr[s:s + step]
    ms = {
        "version": "0.4",
        "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0] * arr.ndim}]}],
    }
    g.attrs.put({"multiscales": [ms]})


_NIFTI_DT = {"uint8": (2, 8), "int16": (4, 16), "float32": (16, 32), "uint16": (512, 16), "uint32": (768, 32)}


def write_nifti(path: Path, arr: np.ndarray) -> None:
    """Single-file NIfTI-1 (.nii), voxel size 1, identity orientation. `arr` is (Z, Y, X)."""
    code, bits = _NIFTI_DT[str(arr.dtype)]
    Z, Y, X = arr.shape
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, X, Y, Z, 1, 1, 1, 1)  # dim[0..7]
    struct.pack_into("<hh", hdr, 70, code, bits)  # datatype, bitpix
    struct.pack_into("<8f", hdr, 76, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # pixdim (qfac=1)
    struct.pack_into("<f", hdr, 108, 352.0)  # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)  # scl_slope
    struct.pack_into("<B", hdr, 123, 2)  # xyzt_units: mm
    struct.pack_into("<h", hdr, 252, 0)  # qform_code
    struct.pack_into("<h", hdr, 254, 1)  # sform_code: scanner
    struct.pack_into("<4f", hdr, 280, 1.0, 0.0, 0.0, 0.0)  # srow_x
    struct.pack_into("<4f", hdr, 296, 0.0, 1.0, 0.0, 0.0)  # srow_y
    struct.pack_into("<4f", hdr, 312, 0.0, 0.0, 1.0, 0.0)  # srow_z
    hdr[344:348] = b"n+1\x00"
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(b"\x00\x00\x00\x00")  # extension flag
        # NIfTI stores x fastest: a C-order (Z, Y, X) array already is.
        np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tofile(f)


def _write(path: Path, arr: np.ndarray, container: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if container in ("tiff-tiled", "tiff-striped", "ome-tiff"):
        _write_tiff(path, arr, container)
    elif container == "ome-zarr":
        _write_zarr(path, arr)
    elif container == "numpy":
        np.save(path, arr)
    elif container == "nifti":
        write_nifti(path, arr)
    else:
        raise ValueError(container)


# ---------------------------------------------------------------------------------------------
# on-disk cache


def container_dir(root: Path, ds: Dataset, container: str) -> Path:
    return Path(root) / ds.name / container


def file_paths(root: Path, ds: Dataset, container: str):
    d = container_dir(root, ds, container)
    name = "img." + EXT[container]
    return d / "int" / name, (d / "seg" / name if ds.has_mask else None)


def is_ready(root: Path, ds: Dataset, container: str) -> bool:
    marker = container_dir(root, ds, container) / "ready.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text()).get("spec") == _spec(ds)
    except (OSError, ValueError):
        return False


def _spec(ds: Dataset) -> dict:
    d = asdict(ds)
    d["shape"] = list(ds.shape)
    d["generator"] = 1  # bump when the pixel recipe changes
    return d


def ensure(root: Path, ds: Dataset, containers, log=print) -> None:
    """Write `ds` to each container that is missing or stale. Arrays are built once per call."""
    allowed = CONTAINERS_2D if ds.dim == 2 else CONTAINERS_3D
    todo = []
    for c in containers:
        if c not in allowed:
            raise ValueError(f"{ds.name}: container {c!r} not available in {ds.dim}D (have {allowed})")
        if not is_ready(root, ds, c):
            todo.append(c)
    if not todo:
        return
    log(f"  generating {ds.name} ({ds.n_pixels / 1e6:.0f} Mpx) for {', '.join(todo)}")
    inten = make_intensity(ds)
    mask = make_mask(ds)
    for c in todo:
        ip, sp = file_paths(root, ds, c)
        shutil.rmtree(container_dir(root, ds, c), ignore_errors=True)
        _write(ip, inten, c)
        if mask is not None:
            _write(sp, mask, c)
        (container_dir(root, ds, c) / "ready.json").write_text(
            json.dumps({"spec": _spec(ds)}, indent=1))
        log(f"    wrote {c}")


def estimated_bytes(ds: Dataset) -> int:
    """Bytes of one container's image and mask, uncompressed."""
    per = np.dtype(ds.dtype).itemsize + (np.dtype(ds.mask_dtype).itemsize if ds.has_mask else 0)
    return ds.n_pixels * per
