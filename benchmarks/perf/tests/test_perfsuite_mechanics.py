"""Tests of the perf harness itself (python -m pytest benchmarks/perf/tests).

Everything but the end-to-end tests at the bottom runs without nyxus. Those featurize tiny
datasets through real worker processes and are skipped when nyxus cannot be imported (an error
instead with PERFSUITE_REQUIRE_NYXUS set). The tests of the built executables run when
PERFSUITE_BENCH_EXE names bench_feature_kernels and PERFSUITE_CLI a CHECKTIMING nyxus CLI, and the
instruction-count test when valgrind is installed too.
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perfsuite import compare, datasets, presets, registry, runner  # noqa: E402
from perfsuite.datasets import Dataset  # noqa: E402
from perfsuite.stats import loglog_slope, mad  # noqa: E402


# --- datasets -----------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "2d-4096x4096-grid1024-uint16-s0", "2d-512x768-giant-float32-s3", "2d-64x64-whole-uint8-s1",
    "3d-32x64x64-grid8-uint16-s0",
])
def test_perfsuite_name_round_trips_mechanics(name):
    assert Dataset.parse(name).name == name


@pytest.mark.parametrize("bad", ["2d-64-grid4-uint16-s0", "3d-64x64-grid4-uint16-s0", "2d-64x64-grid4-int7-s0", "x",
                                 "2d-256x256-grid0-uint16-s0", "2d-0x64-whole-uint16-s0", "3d-8x0x8-giant-uint16-s0"])
def test_perfsuite_bad_names_are_refused_mechanics(bad):
    with pytest.raises(ValueError):
        Dataset.parse(bad)


def test_perfsuite_pixels_are_reproducible_mechanics():
    ds = Dataset.parse("2d-300x200-grid12-uint16-s5")
    a, b = datasets.make_intensity(ds), datasets.make_intensity(ds)
    assert a.dtype == np.uint16 and a.shape == (300, 200)
    assert np.array_equal(a, b)
    assert np.array_equal(datasets.make_mask(ds), datasets.make_mask(ds))
    other = datasets.make_intensity(Dataset.parse("2d-300x200-grid12-uint16-s6"))
    assert not np.array_equal(a, other)


@pytest.mark.parametrize("name", ["2d-512x512-grid16-uint16-s0", "2d-500x500-grid10-uint16-s0",
                                  "3d-48x48x48-grid8-uint16-s0", "3d-40x60x60-grid27-uint16-s0"])
def test_perfsuite_grid_has_exactly_n_rois_mechanics(name):
    ds = Dataset.parse(name)
    m = datasets.make_mask(ds)
    labels = np.unique(m)
    assert list(labels) == list(range(0, ds.n_rois + 1))


def test_perfsuite_grid_rois_are_not_convex_mechanics():
    """A disk would leave hull/Feret features nothing to do; the blobs must be indented."""
    from scipy.spatial import ConvexHull

    m = datasets.make_mask(Dataset.parse("2d-512x512-grid4-uint16-s0"))
    ys, xs = np.nonzero(m == 1)
    hull = ConvexHull(np.c_[xs, ys])
    assert len(xs) / hull.volume < 0.97


def test_perfsuite_giant_and_whole_mechanics():
    g = datasets.make_mask(Dataset.parse("2d-256x256-giant-uint16-s0"))
    assert set(np.unique(g)) == {0, 1} and 0.3 < g.mean() < 0.7
    assert datasets.make_mask(Dataset.parse("2d-64x64-whole-uint16-s0")) is None
    g3 = datasets.make_mask(Dataset.parse("3d-64x64x64-giant-uint16-s0"))
    assert set(np.unique(g3)) == {0, 1}


def test_perfsuite_too_many_rois_for_the_image_is_refused_mechanics():
    with pytest.raises(ValueError):
        datasets.make_mask(Dataset.parse("2d-64x64-grid1024-uint16-s0"))


def test_perfsuite_mask_dtype_widens_past_65535_labels_mechanics():
    assert Dataset.parse("2d-16384x16384-grid65535-uint16-s0").mask_dtype == "uint16"
    assert Dataset.parse("2d-16384x16384-grid65536-uint16-s0").mask_dtype == "uint32"


def test_perfsuite_nifti_header_mechanics(tmp_path):
    arr = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)  # Z, Y, X
    p = tmp_path / "v.nii"
    datasets.write_nifti(p, arr)
    raw = p.read_bytes()
    assert struct.unpack_from("<i", raw, 0)[0] == 348
    assert struct.unpack_from("<4h", raw, 40) == (3, 4, 3, 2)  # ndim, X, Y, Z
    assert struct.unpack_from("<hh", raw, 70) == (512, 16)
    assert raw[344:348] == b"n+1\x00"
    vox = np.frombuffer(raw[352:], dtype="<u2")
    assert np.array_equal(vox, arr.ravel())


def test_perfsuite_ensure_writes_once_and_marks_ready_mechanics(tmp_path):
    ds = Dataset.parse("2d-128x128-grid4-uint16-s0")
    seen = []
    datasets.ensure(tmp_path, ds, ["numpy"], log=seen.append)
    assert datasets.is_ready(tmp_path, ds, "numpy")
    ip, sp = datasets.file_paths(tmp_path, ds, "numpy")
    assert np.array_equal(np.load(ip), datasets.make_intensity(ds))
    assert np.array_equal(np.load(sp), datasets.make_mask(ds))
    seen.clear()
    datasets.ensure(tmp_path, ds, ["numpy"], log=seen.append)
    assert seen == []  # cached


def test_perfsuite_3d_has_no_numpy_container_mechanics(tmp_path):
    with pytest.raises(ValueError):
        datasets.ensure(tmp_path, Dataset.parse("3d-16x16x16-grid1-uint16-s0"), ["numpy"])


# --- presets and registry -----------------------------------------------------------------------

@pytest.mark.parametrize("tier", presets.TIERS)
def test_perfsuite_every_tier_builds_unique_ids_mechanics(tier):
    cases = presets.build(tier, presets.SUITES)
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        Dataset.parse(c["dataset"])  # every dataset name is well formed


def test_perfsuite_feature_suite_covers_every_registry_code_once_mechanics():
    codes = {(r["dim"], r["feature"]) for r in registry.load()}
    cases = presets.feature_cases("smoke")
    assert len(cases) == len(codes)
    assert {c["api"] for c in cases} == {"2d", "3d", "imq"}


def test_perfsuite_load_cases_use_the_api_s_own_group_mechanics():
    for c in presets.load_cases("smoke"):
        assert c["features"] == presets.LOAD_FEATURES[c["api"]]


def test_perfsuite_ooc_mode_sets_the_ram_limit_mechanics():
    c = presets.case("load", "2d", Dataset.parse("2d-64x64-grid4-uint16-s0"), "tiff-tiled", ["MEAN"], "smoke",
                     mode="ooc")
    assert c["params"]["ram_limit"] == presets.OOC_RAM_LIMIT_MB == 0


@pytest.mark.parametrize("tier", presets.TIERS)
def test_perfsuite_ooc_cases_stay_off_the_paths_nyxus_refuses_mechanics(tier):
    ooc = [c for c in presets.load_cases(tier) if c["mode"] == "ooc"]
    assert {c["api"] for c in ooc} == {"2d", "3d"}
    for c in ooc:
        ds = Dataset.parse(c["dataset"])
        assert c["container"] not in ("numpy", "nifti"), c["id"]
        assert not (ds.dim == 2 and ds.layout == "whole"), c["id"]
        assert ds.expected_rois() <= presets.OOC_MAX_ROIS, c["id"]
        assert ds.dim == 3 or ds.n_pixels <= presets.OOC_MAX_PIXELS_2D, c["id"]
    # every out-of-core case has an in-RAM twin to agree with
    inram = {(c["dataset"], c["container"]) for c in presets.load_cases(tier) if c["mode"] == "inram"}
    assert all((c["dataset"], c["container"]) in inram for c in ooc)


def test_perfsuite_resolve_paths_leaves_the_case_alone_mechanics(tmp_path):
    c = presets.case("load", "2d", Dataset.parse("2d-64x64-grid4-float32-s0"), "tiff-tiled", ["MEAN"], "smoke")
    before = json.dumps(c, sort_keys=True)
    r = runner.resolve_paths(c, tmp_path)
    assert r["params"]["max_intensity"] == 1
    assert json.dumps(c, sort_keys=True) == before


# --- stats, compare, digests --------------------------------------------------------------------

def test_perfsuite_loglog_slope_recovers_exponents_mechanics():
    assert loglog_slope([(x, 3 * x ** 2) for x in (1, 2, 4, 8)])[0] == pytest.approx(2.0)
    assert loglog_slope([(x, 5 / x) for x in (1, 2, 4)])[0] == pytest.approx(-1.0)
    assert loglog_slope([(1, 1)]) is None


def test_perfsuite_mad_mechanics():
    assert mad([1, 2, 3, 4, 100]) == 1
    assert mad([5, 5, 5]) == 0


def _results(path, cases, finished=True):
    meta = {"host": "h", "git_sha": "x", **({"finished": "2026-01-01T00:00:00"} if finished else {})}
    path.write_text(json.dumps({"meta": meta, "cases": cases}))
    return path


def _c(cid, med, mad_=0.001, status="ok", rss=100.0, fresh=True):
    return {"id": cid, "status": status, "time": {"median": med, "mad": mad_}, "peak_rss_mb": rss,
            "fresh_worker": fresh}


def test_perfsuite_compare_flags_real_slowdowns_only_mechanics(tmp_path):
    old = _results(tmp_path / "o.json", [_c("a", 1.0), _c("b", 1.0), _c("c", 1.0, mad_=0.2), _c("d", 0.001),
                                         _c("e", 1.0), _c("f", 1.0, rss=100)])
    new = _results(tmp_path / "n.json", [_c("a", 1.5), _c("b", 1.05), _c("c", 1.3, mad_=0.2), _c("d", 0.005),
                                         _c("e", 1.0, status="crashed"), _c("f", 1.0, rss=300)])
    r = compare.compare(old, new)
    ids = [x["id"] for x in r["regressions"]]
    assert "a" in ids  # 50 % slower, beyond noise
    assert "b" not in ids  # within tolerance
    assert "c" not in ids  # within 3 MAD
    assert "d" not in ids  # 5x but under the absolute floor
    assert "f [peak RSS MB]" in ids
    assert [b[0] for b in r["broken"]] == ["e"]


def test_perfsuite_compare_kernel_floor_and_value_changes_mechanics(tmp_path):
    def k(cid, med, value):
        return {"id": cid, "suite": "kernels", "status": "ok", "time": {"median": med, "mad": 0.0}, "value": value}

    old = _results(tmp_path / "o.json", [k("kernels/gabor/side64", 0.003, 1.0), k("kernels/glcm/side64", 0.002, 5.0)])
    new = _results(tmp_path / "n.json", [k("kernels/gabor/side64", 0.006, 1.0), k("kernels/glcm/side64", 0.002, 6.0)])
    r = compare.compare(old, new)
    # 3 ms -> 6 ms is under the 20 ms default floor but over the kernels floor
    assert [x["id"] for x in r["regressions"]] == ["kernels/gabor/side64"]
    assert r["value_changes"] == [("kernels/glcm/side64", 5.0, 6.0)]


def test_perfsuite_compare_reports_improvements_mechanics(tmp_path):
    old = _results(tmp_path / "o.json", [_c("a", 2.0)])
    new = _results(tmp_path / "n.json", [_c("a", 1.0)])
    r = compare.compare(old, new)
    assert [x["id"] for x in r["improvements"]] == ["a"] and not r["regressions"]


def _d(cid, container, digest, mode="inram"):
    return {"id": cid, "suite": "load", "dataset": "2d-64x64-grid4-uint16-s0", "container": container,
            "features": ["*ALL_INTENSITY*"], "params": {"n_feature_calc_threads": 4}, "mode": mode,
            "digest": digest}


def test_perfsuite_digests_catch_disagreement_across_containers_and_modes_mechanics():
    rs = [_d("t", "tiff-tiled", {"MEAN": 10.0, "MAX": 5.0}),
          _d("z", "ome-zarr", {"MEAN": 10.0, "MAX": 5.0}),
          _d("o", "tiff-tiled", {"MEAN": 10.0, "MAX": 6.0}, mode="ooc")]
    bad = runner.check_digests(rs)
    assert [(b[1], b[2]) for b in bad] == [("o", "MAX")]


def test_perfsuite_digests_honor_documented_differences_mechanics():
    rs = [_d("t", "tiff-tiled", {"COVERED_IMAGE_INTENSITY_RANGE": 199.0}),
          _d("n", "numpy", {"COVERED_IMAGE_INTENSITY_RANGE": 0.0})]
    assert runner.check_digests(rs) == []
    rs[1]["container"] = "ome-zarr"
    assert len(runner.check_digests(rs)) == 1


def test_perfsuite_digests_catch_a_feature_one_side_lacks_mechanics():
    rs = [_d("t", "tiff-tiled", {"MEAN": 10.0, "MAX": 5.0}),
          _d("o", "tiff-tiled", {"MEAN": 10.0, "MIN": 1.0}, mode="ooc")]
    bad = sorted((b[2], b[3], b[4]) for b in runner.check_digests(rs))
    assert bad == [("MAX", 5.0, None), ("MIN", None, 1.0)]


def test_perfsuite_digest_counts_nonfinite_values_mechanics():
    import pandas as pd
    from perfsuite.worker import _digest

    zeros = _digest(pd.DataFrame({"ROI_label": [1, 2], "SKEWNESS": [0.0, 0.0]}))
    nan = _digest(pd.DataFrame({"ROI_label": [1, 2], "SKEWNESS": [float("nan"), 0.0]}))
    assert zeros == {"SKEWNESS": 0.0, "SKEWNESS#by_label": 0.0}
    assert nan == {"SKEWNESS": 0.0, "SKEWNESS#by_label": 0.0, "SKEWNESS#nonfinite": 1.0}
    rs = [_d("t", "tiff-tiled", zeros), _d("o", "tiff-tiled", nan, mode="ooc")]
    assert [b[2] for b in runner.check_digests(rs)] == ["SKEWNESS#nonfinite"]


def test_perfsuite_digests_catch_values_given_to_the_wrong_labels_mechanics():
    import pandas as pd
    from perfsuite.worker import _digest

    right = _digest(pd.DataFrame({"ROI_label": [1, 2, 3], "MEAN": [10.0, 20.0, 30.0]}))
    swapped = _digest(pd.DataFrame({"ROI_label": [1, 2, 3], "MEAN": [20.0, 10.0, 30.0]}))
    reordered = _digest(pd.DataFrame({"ROI_label": [3, 1, 2], "MEAN": [30.0, 10.0, 20.0]}))
    assert right["MEAN"] == swapped["MEAN"]  # the plain sum cannot tell
    rs = [_d("t", "tiff-tiled", right), _d("z", "ome-zarr", swapped), _d("o", "tiff-tiled", reordered, mode="ooc")]
    # rows in another order are the same values; labels 1 and 2 swapped are not
    assert [(b[1], b[2]) for b in runner.check_digests(rs)] == [("z", "MEAN#by_label")]


# --- worker timeout, kernels output -------------------------------------------------------------

HANGING_WORKER = """
import json, sys, time
cases = json.load(open(sys.argv[1]))
with open(sys.argv[2], "a") as f:
    f.write(json.dumps({"id": cases[0]["id"], "status": "ok", "times": [0.1], "peak_rss": [1]}) + "\\n")
    f.flush()
time.sleep(120)
"""


def test_perfsuite_timeout_is_per_case_not_per_batch_mechanics(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text(HANGING_WORKER)
    batch = [{"id": f"c{i}"} for i in range(10)]
    done, rc, _, elapsed = runner._run_worker(batch, 2, tmp_path, poll=0.1, worker=[sys.executable, str(script)])
    assert rc == "timeout"
    assert list(done) == ["c0"]
    assert elapsed < 15  # one case's budget after the first case finished, not 10 x 2 s


def test_perfsuite_kernel_rows_survive_a_failed_run_mechanics():
    from perfsuite import kernels

    out = (kernels.HEADER + "\n"
           "gabor,64,3000,20,1.5,0.05,1.4,1.7,0.12345678901234567\n"
           "gabor,256,48000,5,9.0,0.2,8.5,9")  # killed mid-row
    rows = kernels.parse(out)
    assert [r["id"] for r in rows] == ["kernels/gabor/side64"]
    assert rows[0]["value"] == 0.12345678901234567
    # the spread compare's noise gate needs, in seconds like the median
    assert rows[0]["time"]["median"] == 0.0015 and rows[0]["time"]["mad"] == 0.05 / 1000
    assert kernels.parse("") is None  # no header: not this executable's output
    assert kernels.parse(kernels.HEADER + "\n") == []


def test_perfsuite_kernel_values_parse_in_any_runtime_spelling_mechanics():
    import math

    from perfsuite import kernels

    out = (kernels.HEADER + "\n"
           "a,64,1,1,1,0,1,1,-nan(ind)\n"   # MSVC
           "b,64,1,1,1,0,1,1,nan(ind)\n"
           "c,64,1,1,1,0,1,1,-nan\n"        # gcc, clang
           "d,64,1,1,1,0,1,1,inf\n"
           "e,64,1,1,1,0,1,1,-inf\n")
    v = [r["value"] for r in kernels.parse(out)]
    assert all(math.isnan(x) for x in v[:3])
    assert v[3:] == [math.inf, -math.inf]
    with pytest.raises(ValueError):
        kernels.number("garbage")


def _k(cid, value=1.0, status="ok"):
    return {"id": cid, "suite": "kernels", "status": status, "time": {"median": 0.001, "mad": 0.0},
            "value": value}


def test_perfsuite_compare_fails_a_kernels_run_that_died_mechanics(tmp_path):
    from perfsuite.__main__ import main

    old = _results(tmp_path / "o.json", [_k("kernels/gabor/side64"), _k("kernels/gabor/side256")])
    new = _results(tmp_path / "n.json", [_k("kernels/gabor/side64"),
                                         {"id": "kernels/RUN", "suite": "kernels", "status": "error",
                                          "error": "exit 3; 1 rows finished before it", "time": {"n": 0}}])
    r = compare.compare(old, new)
    assert [b[0] for b in r["broken"]] == ["kernels/RUN"]
    assert r["only_old"] == ["kernels/gabor/side256"]
    assert main(["compare", str(old), str(new)]) == 1
    # a new case that is ok is listed, not judged
    ok_new = _results(tmp_path / "n2.json", [_k("kernels/gabor/side64"), _k("kernels/gabor/side256"),
                                             _k("kernels/zernike/side64")])
    assert main(["compare", str(old), str(ok_new)]) == 0


def test_perfsuite_compare_nan_kernel_value_is_no_change_mechanics(tmp_path):
    old = _results(tmp_path / "o.json", [_k("kernels/a/side64", float("nan"))])
    new = _results(tmp_path / "n.json", [_k("kernels/a/side64", float("nan"))])
    assert compare.compare(old, new)["value_changes"] == []
    new = _results(tmp_path / "n.json", [_k("kernels/a/side64", 1.0)])
    assert len(compare.compare(old, new)["value_changes"]) == 1


def _mem(cid, peak, delta=None, fresh=True):
    c = _c(cid, 1.0, rss=peak, fresh=fresh)
    if delta is not None:
        c["delta_rss_mb"] = delta
    return c


def _mem_flags(tmp_path, old, new):
    r = compare.compare(_results(tmp_path / "o.json", old), _results(tmp_path / "n.json", new))
    return [x["id"] for x in r["regressions"]]


def test_perfsuite_memory_is_judged_only_where_the_case_had_its_worker_to_itself_mechanics(tmp_path):
    # a: alone in its worker both times, peak 300 -> 500: judged
    # b, c: batched; what they start from depends on the cases before them, so neither their peak
    #       nor their growth over that start is theirs to be judged on
    old = [_mem("a", 300.0), _mem("b", 300.0, 40.0, fresh=False), _mem("c", 300.0, 40.0, fresh=False)]
    new = [_mem("a", 500.0), _mem("b", 600.0, 41.0, fresh=False), _mem("c", 320.0, 90.0, fresh=False)]
    assert _mem_flags(tmp_path, old, new) == ["a [peak RSS MB]"]


def test_perfsuite_memory_never_compares_one_file_s_peak_with_the_other_s_growth_mechanics(tmp_path):
    # the same 100 MB peak both times; only the new file lacks the growth figure
    assert _mem_flags(tmp_path, [_mem("a", 100.0, 50.0)], [_mem("a", 100.0)]) == []
    assert _mem_flags(tmp_path, [_mem("a", 100.0)], [_mem("a", 100.0, 50.0)]) == []


def test_perfsuite_memory_ignores_a_growth_below_the_starting_point_mechanics(tmp_path):
    # the peak is 100 MB both times, so nothing grew, whatever the growth figures say
    assert _mem_flags(tmp_path, [_mem("a", 100.0, -30.0)], [_mem("a", 100.0, 10.0)]) == []
    case = {k: None for k in ("id", "suite", "api", "dataset", "container", "features", "params", "mode",
                              "sweep", "warmup", "repeats")}
    row = runner._finish(case, {"times": [0.1], "peak_rss": [90 * 2**20], "base_rss": 100 * 2**20})
    assert row["delta_rss_mb"] == 0


def test_perfsuite_run_records_which_cases_had_a_fresh_worker_mechanics(tmp_path, monkeypatch):
    ds = Dataset.parse("2d-64x64-grid4-uint16-s0")
    cases = [presets.case("features", "2d", ds, "tiff-tiled", [f], "smoke") for f in ("MEAN", "MAX", "MIN", "MEDIAN")]
    calls = []

    def fake(batch, *a, **k):
        calls.append(len(batch))
        first = len(calls) == 1  # the first worker finishes one case and dies on the second
        done = {c["id"]: {"id": c["id"], "status": "ok", "times": [0.1], "peak_rss": [2**20]}
                for c in (batch[:1] if first else batch)}
        return done, (3 if first else 0), tmp_path / "w.log", 0.1

    monkeypatch.setattr(runner, "_run_worker", fake)
    res = runner.run(cases, tmp_path / "data", tmp_path / "r.json", batch=4, timeout=5, log=lambda *_: None)
    assert calls == [4, 2]
    assert [(r["status"], r["fresh_worker"]) for r in res] == [("ok", True), ("crashed", True),
                                                             ("ok", True), ("ok", False)]


def test_perfsuite_compare_fails_an_incomplete_or_partial_run_mechanics(tmp_path):
    from perfsuite.__main__ import main

    old = _results(tmp_path / "o.json", [_c("a", 1.0), _c("b", 1.0)])
    unfinished = _results(tmp_path / "u.json", [_c("a", 1.0), _c("b", 1.0)], finished=False)
    partial = _results(tmp_path / "p.json", [_c("a", 1.0)])
    assert compare.compare(old, unfinished)["incomplete"]
    assert main(["compare", str(old), str(unfinished)]) == 1
    assert compare.compare(old, partial)["missing"] == ["b"]
    assert main(["compare", str(old), str(partial)]) == 1
    assert main(["compare", str(old), str(partial), "--allow-missing"]) == 0
    assert main(["compare", str(old), str(old)]) == 0


def test_perfsuite_a_truncated_result_line_is_a_crash_not_an_abort_mechanics(tmp_path):
    script = tmp_path / "w.py"
    script.write_text("import json, sys\n"
                      "f = open(sys.argv[2], 'a')\n"
                      "f.write(json.dumps({'id': 'c0', 'status': 'ok', 'times': [0.1], 'peak_rss': [1]}) + '\\n')\n"
                      "f.write('{\"id\": \"c1\", \"sta'); f.flush(); sys.exit(3)\n")
    done, rc, _, _ = runner._run_worker([{"id": "c0"}, {"id": "c1"}], 30, tmp_path, poll=0.1,
                                        worker=[sys.executable, str(script)])
    assert list(done) == ["c0"] and rc == 3


def test_perfsuite_run_writes_its_results_file_atomically_mechanics(tmp_path, monkeypatch):
    saved = []
    real = runner._save
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    monkeypatch.setattr(runner, "_save", lambda *a: saved.append(a[0]) or real(*a))
    _run_main(tmp_path, "--suite", "kernels")  # no --bench-exe: nothing runs, the file is still written
    assert saved and json.loads((tmp_path / "r.json").read_text())["meta"]["finished"]


# --- profile (CHECKTIMING CLI) -------------------------------------------------------------------

def test_perfsuite_profile_gpu_pass_needs_a_device_mechanics(monkeypatch):
    import subprocess

    from perfsuite import checktiming

    assert checktiming.gpu_device_present({"gpu_available": True})

    def smi(out, rc=0):
        return lambda *a, **k: subprocess.CompletedProcess(a[0], rc, stdout=out, stderr="")

    monkeypatch.setattr(checktiming.subprocess, "run", smi("GPU 0: NVIDIA X (UUID: GPU-1)\n"))
    assert checktiming.gpu_device_present({"gpu_available": False})
    monkeypatch.setattr(checktiming.subprocess, "run", smi("", rc=9))
    assert not checktiming.gpu_device_present({"gpu_available": False})


def test_perfsuite_profile_gpu_pass_skipped_on_a_host_without_a_device_mechanics(tmp_path, monkeypatch):
    from perfsuite import checktiming

    seen = {}
    monkeypatch.setattr(runner, "machine_info", lambda: {"gpu_available": False})
    monkeypatch.setattr(checktiming, "cli_has_gpu", lambda cli: True)
    monkeypatch.setattr(checktiming, "require_checktiming_cli", lambda cli: Path(cli))
    monkeypatch.setattr(checktiming, "gpu_device_present", lambda info: False)
    monkeypatch.setattr(checktiming, "profile", lambda *a, **k: seen.update(k) or [])
    _run_main(tmp_path, "--suite", "profile", "--cli", __file__)
    assert seen["gpu"] is False


def test_perfsuite_filter_on_a_profile_group_runs_and_keeps_that_group_mechanics(tmp_path, monkeypatch):
    import argparse

    from perfsuite import checktiming
    from perfsuite.__main__ import _profile_plan

    name = presets.profile_datasets("smoke")[0]
    args = argparse.Namespace(tier="smoke", filter="profile/.*/cpu/Texture")
    assert _profile_plan(args, ["profile"]) == [(name, "cpu")]
    monkeypatch.setattr(checktiming, "run_cli",
                        lambda *a, **k: (1.0, {"Texture/GLCM": 0.2, "Morphology/Hull": 0.1}, 0))
    rows = checktiming.profile(Path(__file__), [name], tmp_path, repeats=1, log_dir=tmp_path,
                               keep_row=lambda cid: "cpu/Texture" in cid, log=lambda *_: None)
    assert [r["id"] for r in rows] == [f"profile/{name}/cpu/Texture/GLCM"]

def test_perfsuite_cli_gpu_support_is_read_from_its_help_mechanics(monkeypatch):
    import subprocess

    from perfsuite import checktiming

    def fake(help_text):
        return lambda *a, **k: subprocess.CompletedProcess(a[0], 1, stdout=help_text, stderr="")

    monkeypatch.setattr(checktiming.subprocess, "run", fake("Usage:\n [--useGpu=<true or false>]\n"))
    assert checktiming.cli_has_gpu(__file__)
    monkeypatch.setattr(checktiming.subprocess, "run", fake("Usage:\n --features=...\n"))
    assert not checktiming.cli_has_gpu(__file__)


def test_perfsuite_profile_gpu_pass_follows_the_cli_not_the_python_module_mechanics(tmp_path, monkeypatch):
    from perfsuite import __main__ as cli_main
    from perfsuite import checktiming

    seen = {}
    monkeypatch.setattr(runner, "machine_info", lambda: {"gpu_available": True})
    monkeypatch.setattr(checktiming, "cli_has_gpu", lambda cli: False)
    monkeypatch.setattr(checktiming, "require_checktiming_cli", lambda cli: Path(cli))
    monkeypatch.setattr(checktiming, "profile", lambda *a, **k: seen.update(k) or [])
    cli_main.main(["run", "--suite", "profile", "--no-synth", "--data", str(tmp_path),
                   "--out", str(tmp_path / "r.json"), "--cli", __file__])
    assert seen["gpu"] is False


def test_perfsuite_profile_refuses_a_dataset_without_a_mask_mechanics(tmp_path):
    from perfsuite import checktiming

    for name in ("2d-64x64-whole-uint16-s0", "3d-8x64x64-grid8-uint16-s0"):
        with pytest.raises(ValueError, match="2D dataset with a mask"):
            checktiming.run_cli(Path(__file__), Dataset.parse(name), tmp_path)


def test_perfsuite_profile_names_a_cli_without_checktiming_mechanics(tmp_path, monkeypatch):
    from perfsuite import checktiming

    monkeypatch.setattr(checktiming, "run_cli", lambda *a, **k: (1.0, {}, 0))
    rows = checktiming.profile(Path(__file__), ["2d-64x64-grid4-uint16-s0"], tmp_path, repeats=1,
                               log_dir=tmp_path, log=lambda *_: None)
    assert rows[0]["status"] == "error"
    assert "-DCHECKTIMING=ON" in rows[0]["error"]


def test_perfsuite_profile_timeout_reads_as_a_timeout_mechanics(tmp_path, monkeypatch):
    from perfsuite import checktiming

    monkeypatch.setattr(checktiming, "run_cli", lambda *a, **k: (5.0, {}, "timeout"))
    rows = checktiming.profile(Path(__file__), ["2d-64x64-grid4-uint16-s0"], tmp_path, repeats=1,
                               log_dir=tmp_path, timeout=5, log=lambda *_: None)
    assert rows[0]["status"] == "timeout"
    assert rows[0]["error"] == "timeout after 5 s"


# --- --filter reaches every suite ----------------------------------------------------------------

def _run_main(tmp_path, *extra):
    from perfsuite.__main__ import main

    return main(["run", "--no-synth", "--data", str(tmp_path), "--out", str(tmp_path / "r.json"), *extra])


def test_perfsuite_filter_skips_profile_runs_it_does_not_match_mechanics(tmp_path, monkeypatch):
    from perfsuite import checktiming

    calls = []
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    monkeypatch.setattr(checktiming, "cli_has_gpu", lambda cli: True)
    monkeypatch.setattr(checktiming, "require_checktiming_cli", lambda cli: Path(cli))
    monkeypatch.setattr(checktiming, "profile", lambda *a, **k: calls.append(k) or [])
    _run_main(tmp_path, "--suite", "profile", "--filter", "^load/", "--cli", __file__)
    assert calls == []
    _run_main(tmp_path, "--suite", "profile", "--filter", "/cpu/", "--cli", __file__)
    (k,) = calls
    name = presets.profile_datasets("smoke")[0]
    assert k["select"](name, "cpu") and not k["select"](name, "gpu")


def test_perfsuite_filter_leaves_unmatched_profile_datasets_ungenerated_mechanics():
    import argparse

    from perfsuite.__main__ import _needed, _profile_plan

    args = argparse.Namespace(tier="smoke", filter="^load/")
    assert dict(_needed([], _profile_plan(args, ["profile"]))) == {}
    args.filter = None
    assert set(_needed([], _profile_plan(args, ["profile"]))) == set(presets.profile_datasets("smoke"))


def _fake_bench(monkeypatch, *, stdout=None, rc=0):
    """bench_feature_kernels stand-in: lists gabor (max side 128) and glcm, prints one row per
    kernel and side asked for, or `stdout` when given. Returns the argument lists it was run with."""
    import subprocess

    from perfsuite import kernels

    calls = []

    def run(args, **kw):
        calls.append(list(args))
        if "--help" in args:
            return subprocess.CompletedProcess(args, 0, stdout="Usage: x\nKernels:\n  gabor 128\n  glcm 4096\n",
                                               stderr="")
        if stdout is not None:
            return subprocess.CompletedProcess(args, rc, stdout=stdout, stderr="")
        ks = [args[i + 1] for i, a in enumerate(args) if a == "--kernel"] or ["gabor", "glcm"]
        ss = [int(args[i + 1]) for i, a in enumerate(args) if a == "--side"]
        top = {"gabor": 128, "glcm": 4096}
        body = "".join(f"{k},{s},10,1,1.0,0.1,1.0,1.0,0.5\n" for k in ks for s in ss if s <= top[k])
        return subprocess.CompletedProcess(args, rc, stdout=kernels.HEADER + "\n" + body, stderr="")

    monkeypatch.setattr(kernels.subprocess, "run", run)
    return calls


def test_perfsuite_filtered_kernels_run_only_the_selected_pairs_mechanics(monkeypatch):
    from perfsuite import kernels

    calls = _fake_bench(monkeypatch)
    rows = kernels.run(Path(__file__), "smoke", select=lambda cid: cid in ("kernels/gabor/side64", "kernels/glcm/side256"),
                       log=lambda *_: None)
    assert [r["id"] for r in rows] == ["kernels/gabor/side64", "kernels/glcm/side256"]
    runs = [c[1:] for c in calls if "--help" not in c]
    assert runs == [["--kernel", "gabor", "--side", "64"], ["--kernel", "glcm", "--side", "256"]]


def test_perfsuite_a_bench_without_a_kernel_list_is_a_failure_row_mechanics(tmp_path, monkeypatch):
    import subprocess

    from perfsuite import kernels

    monkeypatch.setattr(kernels.subprocess, "run",
                        lambda a, **k: subprocess.CompletedProcess(a, 0, stdout="usage: other tool\n", stderr=""))
    rows = kernels.run(Path(__file__), "smoke", select=lambda cid: True, log=lambda *_: None)
    assert [(r["id"], r["status"]) for r in rows] == [("kernels/RUN", "error")]
    # and `run` refuses it before the long suites start
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    with pytest.raises(RuntimeError, match="lists no kernels"):
        _run_main(tmp_path, "--suite", "kernels", "--filter", "x", "--bench-exe", __file__)


def test_perfsuite_filter_narrows_the_kernels_run_mechanics(monkeypatch):
    from perfsuite import kernels

    calls = _fake_bench(monkeypatch)
    rows = kernels.run(Path(__file__), "smoke", select=lambda cid: "glcm/side64" in cid, log=lambda *_: None)
    assert [r["id"] for r in rows] == ["kernels/glcm/side64"]
    assert ["--kernel", "glcm"] == calls[-1][1:3] and "gabor" not in calls[-1]
    assert kernels.run(Path(__file__), "smoke", select=lambda cid: False, log=lambda *_: None) == []
    # the plan honours each kernel's largest side (gabor stops at 128, smoke runs 64 and 256)
    assert kernels.plan(Path(__file__), "smoke") == [("gabor", 64), ("glcm", 64), ("glcm", 256)]


@pytest.mark.parametrize("stdout", ["", "kernel,side\n", None])
def test_perfsuite_a_kernels_run_without_rows_is_a_failure_mechanics(monkeypatch, stdout):
    from perfsuite import kernels

    _fake_bench(monkeypatch, stdout=kernels.HEADER + "\n" if stdout is None else stdout)
    rows = kernels.run(Path(__file__), "smoke", log=lambda *_: None)
    assert [(r["id"], r["status"]) for r in rows] == [("kernels/RUN", "error")]


def test_perfsuite_executables_resolve_without_their_extension_mechanics(tmp_path, monkeypatch):
    import os

    from perfsuite import kernels
    from perfsuite.checktiming import find_executable

    exe = tmp_path / ("tool.exe" if os.name == "nt" else "tool")
    exe.write_text("")
    exe.chmod(0o755)
    assert find_executable(tmp_path / "tool", "tool").name.lower() == exe.name
    with pytest.raises(FileNotFoundError, match="tool not found"):
        find_executable(tmp_path / "nothing", "tool")
    seen = []
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    monkeypatch.setattr(kernels, "run", lambda exe_, *a, **k: seen.append(exe_) or [])
    monkeypatch.setattr(kernels, "names", lambda exe_: {})  # an empty file lists no kernels
    _run_main(tmp_path, "--suite", "kernels", "--bench-exe", str(tmp_path / "tool"))
    assert seen == [str(tmp_path / "tool")]


def test_perfsuite_compare_fails_on_a_changed_kernel_value_mechanics(tmp_path):
    from perfsuite.__main__ import main

    old = _results(tmp_path / "o.json", [_k("kernels/glcm/side64", 5.0)])
    new = _results(tmp_path / "n.json", [_k("kernels/glcm/side64", 6.0)])
    assert main(["compare", str(old), str(new)]) == 1
    assert main(["compare", str(old), str(old)]) == 0


def test_perfsuite_a_bench_that_cannot_start_is_a_failure_row_mechanics(tmp_path):
    from perfsuite import kernels

    exe = tmp_path / ("bench.exe" if sys.platform == "win32" else "bench")
    exe.write_text("not a program\n")  # a file, so it resolves, but not an executable one
    exe.chmod(0o644)
    rows = kernels.run(exe, "smoke", log=lambda *_: None)
    assert [(r["id"], r["status"]) for r in rows] == [("kernels/RUN", "error")]
    assert "cannot start" in rows[0]["error"]


def test_perfsuite_a_stray_output_line_keeps_the_rows_around_it_mechanics(monkeypatch):
    from perfsuite import kernels

    out = (kernels.HEADER + "\n"
           "gabor,64,10,1,1.0,0.1,1.0,1.0,0.5\n"
           "warning: something the runtime printed\n"
           "glcm,64,10,1,1.0,0.1,1.0,1.0,0.5\n")
    _fake_bench(monkeypatch, stdout=out)
    rows = kernels.run(Path(__file__), "smoke", log=lambda *_: None)
    assert [(r["id"], r["status"]) for r in rows] == [("kernels/gabor/side64", "ok"), ("kernels/glcm/side64", "ok"),
                                                      ("kernels/RUN", "error")]
    assert "warning: something" in rows[-1]["error"]
    with pytest.raises(ValueError, match="not a kernel row"):
        kernels.parse(out)


def _clocked_bench(monkeypatch, cost):
    """A bench whose kernel `k` takes cost[k] seconds on a fake clock; returns the timeouts asked for."""
    import subprocess

    from perfsuite import kernels

    now = [0.0]
    asked = {}
    monkeypatch.setattr(kernels.time, "monotonic", lambda: now[0])

    def run(args, timeout=None, **kw):
        if "--help" in args:
            return subprocess.CompletedProcess(args, 0, stdout="Kernels:\n  gabor 4096\n  glcm 4096\n", stderr="")
        k = args[args.index("--kernel") + 1]
        asked[k] = timeout
        if cost[k] > timeout:
            now[0] += timeout
            raise subprocess.TimeoutExpired(args, timeout, output=kernels.HEADER + "\n", stderr="")
        now[0] += cost[k]
        return subprocess.CompletedProcess(args, 0, stdout=kernels.HEADER + f"\n{k},64,10,1,1.0,0.1,1.0,1.0,0.5\n",
                                           stderr="")

    monkeypatch.setattr(kernels.subprocess, "run", run)
    return asked


def test_perfsuite_a_filtered_kernels_run_keeps_to_its_budget_mechanics(monkeypatch):
    from perfsuite import kernels

    sel = lambda cid: cid.endswith("side64")  # noqa: E731
    asked = _clocked_bench(monkeypatch, {"gabor": 9.5, "glcm": 5.0})
    rows = kernels.run(Path(__file__), "smoke", timeout=10, select=sel, log=lambda *_: None)
    assert asked == {"gabor": 10, "glcm": 0.5}  # glcm gets what is left, not a fresh second
    assert rows[-1]["status"] == "timeout" and "--kernel glcm" in rows[-1]["error"]
    assert "budget ran out during it" in rows[-1]["error"]
    # gabor used it all: glcm is not started, and the row says so
    asked = _clocked_bench(monkeypatch, {"gabor": 10.0, "glcm": 0.1})
    rows = kernels.run(Path(__file__), "smoke", timeout=10, select=sel, log=lambda *_: None)
    assert "glcm" not in asked
    assert rows[-1]["status"] == "timeout" and "ran out before --kernel glcm started" in rows[-1]["error"]


def test_perfsuite_child_env_puts_perfsuite_on_the_path_once_mechanics(monkeypatch):
    import os

    from perfsuite import nyxenv

    parent = str(Path(nyxenv.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([parent, "elsewhere"]))
    assert nyxenv.child_env()["PYTHONPATH"].split(os.pathsep) == [parent, "elsewhere"]
    monkeypatch.delenv("PYTHONPATH")
    assert nyxenv.child_env()["PYTHONPATH"] == parent


def test_perfsuite_report_shows_peak_and_growth_and_marks_shared_peaks_mechanics(tmp_path, capsys):
    from perfsuite.__main__ import main

    def case(suite, cid, fresh):
        return {"id": cid, "suite": suite, "status": "ok", "time": {"median": 0.5, "mad": 0.01},
                "features": ["MEAN"], "dataset": "d", "peak_rss_mb": 250.0, "delta_rss_mb": 12.5,
                "fresh_worker": fresh}

    path = _results(tmp_path / "r.json", [case("load", "load/x", True), case("features", "features/d/MEAN/fo", False)])
    assert main(["report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "| growth MB |" in out
    assert "| 250 | 12.5 | ok |" in out      # load: own worker, unmarked
    assert "| 250* | 12.5 |" in out          # features: batched, marked


# --- rss monitor --------------------------------------------------------------------------------

def test_perfsuite_monitor_credits_a_sample_only_to_the_repeat_it_was_read_in_mechanics():
    from perfsuite.rss_monitor import Monitor

    level = [100]
    mon = None

    def rss():
        if level[0] == 999:  # the read is slow: the worker stops this repeat and starts the next
            level[0] = 100
            mon.stop()
            mon.start()
            return 999
        return level[0]

    mon = Monitor(rss)
    mon.start()
    level[0] = 999
    assert mon.sample_once()
    peak, n = mon.stop()
    assert (peak, n) == (100, 0)  # the 999 belongs to the repeat that had already stopped


def test_perfsuite_a_dead_monitor_costs_one_repeat_s_memory_not_the_batch_mechanics(monkeypatch):
    import pandas as pd

    from perfsuite import worker

    sampler = worker.RssSampler()
    calls = []

    def call(obj, case, arrays):
        calls.append(case["id"])
        if len(calls) == 1:
            sampler._mon.kill()  # the monitor dies while the first repeat runs
            sampler._mon.wait()
        return pd.DataFrame({"ROI_label": [1], "MEAN": [2.0]})

    monkeypatch.setattr(worker, "_build", lambda case: object())
    monkeypatch.setattr(worker, "_call", call)
    try:
        first = worker.run_case({"id": "a", "container": "tiff-tiled", "repeats": 2}, sampler)
        second = worker.run_case({"id": "b", "container": "tiff-tiled", "repeats": 1}, sampler)
    finally:
        sampler.close()
    assert first["status"] == "ok" and first["rss_lost"] == 1 and len(first["peak_rss"]) == 1
    assert second["status"] == "ok" and "rss_lost" not in second and second["peak_rss"][0] > 0
    assert sampler.restarts == 1
    # the case whose peak may be missing its highest repeat has no peak at all
    case = {k: None for k in ("id", "suite", "api", "dataset", "container", "features", "params", "mode",
                              "sweep", "warmup", "repeats")}
    row = runner._finish(case, {"times": [0.1, 0.1], "peak_rss": [2**20], "rss_lost": 1, "base_rss": 1})
    assert row["peak_rss_mb"] is None and "delta_rss_mb" not in row and row["rss_lost"] == 1


def test_perfsuite_skipped_suites_generate_no_data_mechanics(tmp_path, monkeypatch):
    from perfsuite import __main__ as cli_main

    made = []
    monkeypatch.setattr(runner, "machine_info", lambda: {"gpu_available": False})
    monkeypatch.setattr(cli_main.datasets, "ensure", lambda root, ds, cs, **k: made.append((ds.name, tuple(cs))))
    monkeypatch.setattr(runner, "run", lambda cases, *a, **k: [])
    # no GPU for the gpu suite, no --cli for the profile suite: nothing of theirs is generated
    cli_main.main(["run", "--tier", "full", "--suite", "gpu", "--suite", "profile", "--data", str(tmp_path),
                   "--out", str(tmp_path / "results" / "r.json")])
    assert made == []
    # a suite that runs still gets its data
    cli_main.main(["run", "--tier", "smoke", "--suite", "load", "--filter", "numpy", "--data", str(tmp_path),
                   "--out", str(tmp_path / "results" / "r.json")])
    assert made and all("numpy" in cs for _, cs in made)


def test_perfsuite_profile_logs_go_with_the_run_s_logs_and_start_fresh_mechanics(tmp_path, monkeypatch):
    from perfsuite import checktiming

    seen = {}
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    monkeypatch.setattr(checktiming, "cli_has_gpu", lambda cli: False)
    monkeypatch.setattr(checktiming, "require_checktiming_cli", lambda cli: Path(cli))
    monkeypatch.setattr(checktiming, "profile", lambda *a, **k: seen.update(k) or [])
    _run_main(tmp_path, "--suite", "profile", "--cli", __file__)
    assert seen["log_dir"] == tmp_path / "r-logs"
    # each run's log replaces the last one's instead of growing across runs
    log = tmp_path / "cli-2d-64x64-grid4-uint16-s0-cpu.log"
    log.write_text("an earlier run\n")
    written = []

    def run_cli(*a, log_path=None, **k):
        written.append(log_path.exists())
        return 1.0, {"Texture/GLCM": 0.1}, 0

    monkeypatch.undo()
    monkeypatch.setattr(checktiming, "run_cli", run_cli)
    checktiming.profile(Path(__file__), ["2d-64x64-grid4-uint16-s0"], tmp_path, repeats=1, log_dir=tmp_path,
                        log=lambda *_: None)
    assert written == [False]


def test_perfsuite_profile_rows_are_the_ones_a_2d_run_can_report_mechanics():
    from perfsuite import checktiming

    cpu, gpu = set(checktiming.run_groups("cpu")), set(checktiming.run_groups("gpu"))
    assert "Texture/GLCM" in cpu and "Texture/GLCM" in gpu
    assert not any(g.startswith(("3D", "ImageQuality/")) for g in cpu | gpu)
    assert "GPU-Gabor/GPU-Gabor" in gpu and not any(g.startswith(("GPU-", "Upload2Gpu/")) for g in cpu)


def test_perfsuite_monitor_sleeps_its_period_between_samples_mechanics():
    import threading

    from perfsuite.rss_monitor import Monitor, sample_loop

    mon, wake, done = Monitor(lambda: 1), threading.Event(), threading.Event()
    slept = []

    def sleep(s):  # stands in for time.sleep, the call that keeps to the period on Windows too
        slept.append(s)
        if len(slept) == 3:
            done.set()

    mon.start()
    t = threading.Thread(target=sample_loop, args=(mon, 0.005, wake, done, sleep), daemon=True)
    t.start()
    t.join(timeout=5)
    done.set()
    wake.set()
    assert slept == [0.005] * 3 and mon.samples == 3


def test_perfsuite_monitor_samples_at_its_period_mechanics():
    import time

    from perfsuite import worker

    # 100 periods; a timer that rounds up to the 15.6 ms Windows tick gets about 32. The best of
    # three windows is taken, so one window in which a busy machine descheduled the monitor passes.
    best = 0
    sampler = worker.RssSampler(period=0.005)
    try:
        for _ in range(3):
            sampler.start()
            time.sleep(0.5)
            sampler.stop()
            best = max(best, sampler.samples)
            if best >= 60:
                break
    finally:
        sampler.close()
    assert best >= 60


BENCH_EXE = os.environ.get("PERFSUITE_BENCH_EXE")


@pytest.mark.skipif(not BENCH_EXE, reason="set PERFSUITE_BENCH_EXE to a built bench_feature_kernels")
def test_perfsuite_bench_refuses_an_unknown_kernel_and_lists_side_limits_mechanics():
    import subprocess

    from perfsuite import kernels

    p = subprocess.run([BENCH_EXE, "--kernel", "Gabor", "--side", "64"], capture_output=True, text=True)
    assert p.returncode != 0 and "unknown kernel Gabor" in p.stderr
    top = kernels.names(BENCH_EXE)
    assert top["gabor"] == 1024 and "glcm" in top
    p = subprocess.run([BENCH_EXE, "--kernel", "gabor", "--side", "2048"], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == kernels.HEADER  # past gabor's largest side


@pytest.mark.skipif(not BENCH_EXE, reason="set PERFSUITE_BENCH_EXE to a built bench_feature_kernels")
@pytest.mark.parametrize("depth", ["0", "-64", "64"])
def test_perfsuite_bench_takes_every_grey_depth_nyxus_takes_mechanics(depth):
    import subprocess

    from perfsuite import kernels

    p = subprocess.run([BENCH_EXE, "--kernel", "glcm", "--side", "64", "--greydepth", depth],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    rows = kernels.parse(p.stdout)
    assert [r["id"] for r in rows] == ["kernels/glcm/side64"]
    assert 0 <= rows[0]["time"]["mad"] <= rows[0]["time"]["max"] - rows[0]["time"]["min"]


@pytest.mark.skipif(not BENCH_EXE, reason="set PERFSUITE_BENCH_EXE to a built bench_feature_kernels")
def test_perfsuite_bench_refuses_a_bad_side_or_depth_mechanics():
    import subprocess

    for args in (["--side", "0"], ["--side", "-64"], ["--side", "3000000000"], ["--greydepth", "-3000000000"],
                 ["--reps", "4294967297"], ["--greydepth", "x"], ["--greydepth", ""]):
        p = subprocess.run([BENCH_EXE, "--kernel", "glcm", *args], capture_output=True, text=True)
        assert p.returncode == 1 and "integer" in p.stderr, args


# --- dataset generation --------------------------------------------------------------------------

def test_perfsuite_intensity_does_not_depend_on_the_chunk_size_mechanics(monkeypatch):
    for name in ("2d-300x200-grid12-uint16-s5", "3d-20x30x40-grid8-float32-s1"):
        ds = Dataset.parse(name)
        ref = datasets.make_intensity(ds)
        monkeypatch.setattr(datasets, "CHUNK_VALUES", 1000)
        assert np.array_equal(datasets.make_intensity(ds), ref)
        monkeypatch.undo()


def test_perfsuite_3d_generation_memory_stays_near_the_output_mechanics(monkeypatch):
    import tracemalloc

    ds = Dataset.parse("3d-64x96x96-grid8-uint16-s0")
    out_bytes = ds.n_pixels * 2
    chunk = 96 * 96  # one slice
    monkeypatch.setattr(datasets, "CHUNK_VALUES", chunk, raising=False)
    tracemalloc.start()
    datasets.make_intensity(ds)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # the output plus a few float64-sized temporaries of one chunk; one whole-volume float64 noise
    # array alone would be 4x the output
    assert peak < out_bytes + 4 * 8 * chunk


def test_perfsuite_disk_estimate_counts_each_dtype_mechanics():
    assert datasets.estimated_bytes(Dataset.parse("2d-100x100-grid4-uint16-s0")) == 100 * 100 * (2 + 2)
    assert datasets.estimated_bytes(Dataset.parse("2d-100x100-grid4-float32-s0")) == 100 * 100 * (4 + 2)
    assert datasets.estimated_bytes(Dataset.parse("2d-100x100-whole-uint32-s0")) == 100 * 100 * 4
    big = Dataset.parse("2d-2048x2048-grid70000-uint16-s0")
    assert datasets.estimated_bytes(big) == big.n_pixels * (2 + 4)  # uint32 mask past 65,535 labels


def test_perfsuite_generation_does_not_sort_the_mask_mechanics(tmp_path, monkeypatch):
    calls = []
    real = np.unique
    monkeypatch.setattr(np, "unique", lambda *a, **k: calls.append(1) or real(*a, **k))
    datasets.ensure(tmp_path, Dataset.parse("2d-64x64-grid4-uint16-s0"), ["tiff-tiled", "ome-tiff", "numpy"],
                    log=lambda *_: None)
    assert calls == []


# --- worker --------------------------------------------------------------------------------------

def test_perfsuite_sampler_stops_when_the_featurize_call_raises_mechanics(monkeypatch):
    from perfsuite import worker

    monkeypatch.setattr(worker, "_build", lambda case: object())

    def boom(*_):
        raise RuntimeError("feature failed")

    monkeypatch.setattr(worker, "_call", boom)
    sampler = worker.RssSampler()
    try:
        r = worker.run_case({"id": "x", "container": "tiff-tiled", "repeats": 1}, sampler)
        assert r["status"] == "error" and "feature failed" in r["error"]
        assert not sampler.active
        sampler.start()  # and it still answers for the next case
        assert sampler.stop() > 0
    finally:
        sampler.close()


def _lock_holding_sleep():
    """A C function that blocks for `ms` milliseconds holding the interpreter lock, as a Nyxus
    featurize call does. Looked up before use: loading the library lets other threads run."""
    import ctypes

    if sys.platform == "win32":
        return ctypes.PyDLL("kernel32").Sleep
    usleep = ctypes.PyDLL(None).usleep
    return lambda ms: usleep(ms * 1000)


def test_perfsuite_sampler_sees_memory_while_the_interpreter_lock_is_held_mechanics():
    from perfsuite import worker

    n = 200 * 2**20
    sampler = worker.RssSampler()
    old = sys.getswitchinterval()
    try:
        base = sampler.rss()
        sys.setswitchinterval(30)  # nothing else in this process gets to run until we block
        sleep = _lock_holding_sleep()
        sampler.start()
        block = b"\x01" * n  # touched, so resident
        sleep(1000)  # 200 sampling periods: a monitor descheduled for most of them still sees it
        del block
        peak = sampler.stop()
    finally:
        sys.setswitchinterval(old)
        sampler.close()
    assert peak - base > 0.8 * n


# --- end to end ---------------------------------------------------------------------------------

def _nyxus_importable():
    """Whether the end-to-end tests can run. With PERFSUITE_REQUIRE_NYXUS set (CI), an import
    failure is an error rather than a reason to skip them."""
    from perfsuite import nyxenv

    try:
        nyxenv.prepare()
        import nyxus  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        if os.environ.get("PERFSUITE_REQUIRE_NYXUS"):
            raise
        return False


@pytest.mark.skipif(not _nyxus_importable(), reason="nyxus not importable")
def test_perfsuite_worker_end_to_end_mechanics(tmp_path):
    ds = Dataset.parse("2d-256x256-grid16-uint16-s0")
    cases = [presets.case("load", "2d", ds, c, ["*ALL_INTENSITY*"], "smoke", mode=m, digest=True, repeats=(0, 2))
             for c, m in (("tiff-tiled", "inram"), ("tiff-tiled", "ooc"), ("ome-zarr", "ooc"), ("numpy", "inram"))]
    cases.append(presets.case("features", "2d", ds, "tiff-tiled", ["NOT_A_FEATURE"], "smoke", repeats=(0, 1)))
    datasets.ensure(tmp_path / "data", ds, ["tiff-tiled", "ome-zarr", "numpy"], log=lambda *_: None)
    res = runner.run(cases, tmp_path / "data", tmp_path / "r.json", batch=8, timeout=300, log=lambda *_: None)
    by = {r["id"]: r for r in res}
    ok = [r for r in res if r["status"] == "ok"]
    assert len(ok) == 4, [(r["id"], r.get("error")) for r in res]
    for r in ok:
        assert r["rows"] == 16 and r["time"]["n"] == 2 and r["peak_rss_mb"] > 0
    assert by[cases[-1]["id"]]["status"] == "error"
    assert runner.check_digests(res) == []
    saved = json.loads((tmp_path / "r.json").read_text())
    assert len(saved["cases"]) == len(cases)


@pytest.mark.skipif(not _nyxus_importable(), reason="nyxus not importable")
def test_perfsuite_worker_end_to_end_3d_and_imq_mechanics(tmp_path):
    ds3 = Dataset.parse("3d-24x32x32-grid8-uint16-s0")
    ds2 = Dataset.parse("2d-128x128-grid4-uint16-s0")
    cases = [presets.case("load", "3d", ds3, c, presets.LOAD_FEATURES["3d"], "smoke", digest=True, repeats=(0, 1))
             for c in datasets.CONTAINERS_3D]
    cases.append(presets.case("groups", "imq", ds2, "tiff-tiled", ["*ALL_IMQ*"], "smoke", repeats=(0, 1)))
    datasets.ensure(tmp_path / "data", ds3, datasets.CONTAINERS_3D, log=lambda *_: None)
    datasets.ensure(tmp_path / "data", ds2, ["tiff-tiled"], log=lambda *_: None)
    res = runner.run(cases, tmp_path / "data", tmp_path / "r.json", batch=8, timeout=300, log=lambda *_: None)
    for r in res:
        assert r["status"] == "ok", (r["id"], r.get("error"))
    assert [r["rows"] for r in res[:3]] == [8, 8, 8]
    assert runner.check_digests(res) == []  # the three 3D readers give the same values


def _oversized_per_case(log_dir: Path) -> dict:
    """{case id: number of "oversized ROI <label>" lines} from the worker logs (verbose >= 2)."""
    import re

    out = {}
    for log in log_dir.glob("*.log"):
        cur = None
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"\[perfsuite\] (\S+)", line)
            if m:
                cur = m.group(1)
                out[cur] = 0
            elif cur and re.match(r"oversized ROI \d+ \(", line):
                out[cur] += 1
    return out


@pytest.mark.skipif(not _nyxus_importable(), reason="nyxus not importable")
@pytest.mark.parametrize("name,container,features", [
    ("2d-256x256-grid16-uint16-s0", "tiff-tiled", ["*ALL_INTENSITY*"]),
    ("3d-24x32x32-grid8-uint16-s0", "ome-tiff", ["*3D_ALL_INTENSITY*"]),
])
def test_perfsuite_ooc_cases_go_out_of_core_mechanics(tmp_path, name, container, features):
    """Every ROI of an out-of-core case takes the oversized-ROI path, and none of an in-RAM case."""
    ds = Dataset.parse(name)
    api = "2d" if ds.dim == 2 else "3d"
    cases = [presets.case("load", api, ds, container, features, "smoke", mode=m, params={"verbose": 2},
                          repeats=(0, 1)) for m in ("inram", "ooc")]
    datasets.ensure(tmp_path / "data", ds, [container], log=lambda *_: None)
    res = runner.run(cases, tmp_path / "data", tmp_path / "r.json", timeout=300, log=lambda *_: None)
    assert all(r["status"] == "ok" for r in res), [r.get("error") for r in res]
    seen = _oversized_per_case(tmp_path / "r-logs")
    assert seen == {cases[0]["id"]: 0, cases[1]["id"]: ds.n_rois}


# --- gpu suite, value checks, profile groups, provenance, report encoding ------------------------

def test_perfsuite_gpu_suite_runs_only_what_python_runs_on_the_gpu_mechanics():
    for tier in presets.TIERS:
        gpu = [c for c in presets.build(tier, ["gpu"]) if c["params"]["use_gpu_device"] >= 0]
        assert gpu
        # GPU Gabor from Python has an empty filter bank (docs/source/gpu.rst)
        assert not any("GABOR" in c["features"] for c in gpu)
        # GPU erosion needs a moment feature in the same run (#520)
        for c in gpu:
            if "EROSIONS_2_VANISH" in c["features"]:
                assert any("GEOMOMS" in f for f in c["features"]), c["id"]


def _gpu_pair(digest_cpu, digest_gpu):
    def r(cid, dev, d):
        return {"id": cid, "suite": "gpu", "dataset": "2d-64x64-grid4-uint16-s0", "container": "tiff-tiled",
                "features": ["*SGEOMOMS*"], "params": {"use_gpu_device": dev, "n_feature_calc_threads": 16},
                "mode": "inram", "digest": d}
    return [r("cpu", -1, digest_cpu), r("gpu", 0, digest_gpu)]


def test_perfsuite_documented_gpu_moment_differences_are_not_disagreements_mechanics():
    cpu = {"CENTRAL_MOMENT_01": 1.0, "WT_NORM_CTR_MOM_02": 1.0, "IMOM_WCM_03": 1.0, "WEIGHTED_HU_M1": 1.0,
           "IMOM_HU1#by_label": 1.0, "IMOM_WNCM_02": 1.0, "SPAT_MOMENT_00": 1.0, "IMOM_NRM_00": 1.0,
           "EROSIONS_2_VANISH": 7.0}
    gpu = {k: v * 3 for k, v in cpu.items()}
    bad = sorted(b[2] for b in runner.check_digests(_gpu_pair(cpu, gpu)))
    # raw and normalized spatial moments and erosion are not in the documented set: they must still agree
    assert bad == ["EROSIONS_2_VANISH", "IMOM_NRM_00", "SPAT_MOMENT_00"]
    # and only on the GPU: the same columns between two CPU paths are checked
    rs = _gpu_pair(cpu, gpu)
    rs[1]["params"]["use_gpu_device"] = -1
    assert len(runner.check_digests(rs)) == len(cpu)


def test_perfsuite_documented_gpu_difference_names_only_moment_features_mechanics():
    from perfsuite import registry

    pat = runner.DOCUMENTED_DIFFERENCES["gpu"]
    hit = {r["feature"] for r in registry.load() if r["dim"] == "2D" and pat.fullmatch(r["feature"])}
    assert hit and all("MOM" in f or "HU_M" in f or f.startswith("IMOM_") for f in hit), sorted(hit)


@pytest.mark.parametrize("tier", presets.TIERS)
def test_perfsuite_thread_count_is_value_checked_without_a_gpu_mechanics(tier):
    groups = {}
    for c in presets.build(tier, ["load", "groups", "features", "sweep"]):
        if c["digest"]:
            p = {k: v for k, v in c["params"].items() if k != "n_feature_calc_threads"}
            key = (c["suite"], c["dataset"], c["container"], c["mode"], tuple(c["features"]),
                   json.dumps(p, sort_keys=True))
            groups.setdefault(key, set()).add(c["params"]["n_feature_calc_threads"])
    assert any(len(t) > 1 for t in groups.values())


def test_perfsuite_profile_groups_come_from_the_stopwatches_in_the_sources_mechanics(tmp_path):
    from perfsuite import checktiming

    groups = checktiming.stopwatch_groups()
    for g in ("GPU-Gabor/GPU-Gabor", "GPU-Morphology/GPU-Erosion", "Intensity/IntensityHistogram",
              "ImageQuality/Sharpness", "Texture/GLCM", "prescan/p0"):
        assert g in groups
    assert not any(g.startswith("sz01/") for g in groups)  # commented out in glszm.cpp
    assert "profile/d/gpu/GPU-Gabor/GPU-Gabor" in checktiming.row_ids("d", "gpu")
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.cpp").write_text('{ STOPWATCH("Cat/Name/X/#fff", "=");\n'
                               '//STOPWATCH("Gone/Gone/X/#fff", "=");\n'
                               '#define STOPWATCH(H,T) Stopwatch stopWatch(H,T);\n')
    assert checktiming.stopwatch_groups(src) == ("Cat/Name",)
    with pytest.raises(FileNotFoundError, match="STOPWATCH"):
        checktiming.stopwatch_groups(tmp_path / "nothing")


def test_perfsuite_a_failed_profile_repeat_is_one_failure_row_mechanics(tmp_path, monkeypatch):
    from perfsuite import checktiming

    runs = iter([(1.0, {"Texture/GLCM": 0.2, "Morphology/Hull": 0.1}, 0), (2.0, {}, "timeout")])
    monkeypatch.setattr(checktiming, "run_cli", lambda *a, **k: next(runs))
    rows = checktiming.profile(Path(__file__), ["2d-64x64-grid4-uint16-s0"], tmp_path, repeats=3,
                               log_dir=tmp_path, timeout=5, log=lambda *_: None)
    assert [(r["id"].rsplit("/", 1)[1], r["status"]) for r in rows] == [("WALL", "timeout")]
    assert rows[0]["error"] == "repeat 2 of 3: timeout after 5 s"


def test_perfsuite_untracked_files_make_a_checkout_dirty_mechanics(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("git"):
        pytest.skip("git not installed")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    assert runner.git_dirty(tmp_path) is False
    (tmp_path / "new.py").write_text("x = 1\n")  # a new source file, not yet added
    assert runner.git_dirty(tmp_path) is True


def test_perfsuite_report_redirected_to_a_file_is_utf8_mechanics(tmp_path):
    import subprocess

    from perfsuite import nyxenv

    case = {"id": "load/x", "suite": "load", "status": "ok", "time": {"median": 0.5, "mad": 0.01},
            "features": ["MEAN"], "dataset": "d", "peak_rss_mb": 250.0, "fresh_worker": True}
    path = _results(tmp_path / "r.json", [case])
    env = {k: v for k, v in nyxenv.child_env().items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    out = tmp_path / "r.md"
    with open(out, "wb") as f:
        subprocess.run([sys.executable, "-m", "perfsuite", "report", str(path)], stdout=f, env=env, check=True,
                       cwd=tmp_path)
    assert "·" in out.read_bytes().decode("utf-8")


# --- harness failures, early checks, report, floors, logs ----------------------------------------

def test_perfsuite_a_worker_that_cannot_measure_stops_the_run_mechanics(tmp_path, monkeypatch):
    from perfsuite import worker
    from perfsuite.__main__ import main

    # the monitor never starts: the worker says so with its own exit status, not as a feature error
    monkeypatch.setattr(worker.RssSampler, "start", lambda self: (_ for _ in ()).throw(worker.MonitorGone("no")))
    cases = tmp_path / "c.json"
    cases.write_text(json.dumps([{"id": "a", "container": "tiff-tiled", "repeats": 1}]))
    monkeypatch.setattr(worker, "_build", lambda case: object())
    assert worker.main([str(cases), str(tmp_path / "r.jsonl")]) == worker.HARNESS_FAILURE
    assert not (tmp_path / "r.jsonl").read_text().strip()
    # and the runner stops instead of charging case after case with it
    script = tmp_path / "w.py"
    script.write_text(f"import sys\nsys.exit({worker.HARNESS_FAILURE})\n")
    monkeypatch.setattr(runner, "WORKER", [sys.executable, str(script)])
    ds = Dataset.parse("2d-64x64-grid4-uint16-s0")
    many = [presets.case("features", "2d", ds, "tiff-tiled", [f], "smoke") for f in ("MEAN", "MAX")]
    with pytest.raises(runner.HarnessFailure):
        runner.run(many, tmp_path / "data", tmp_path / "r.json", batch=2, timeout=30, log=lambda *_: None)
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    assert main(["run", "--no-synth", "--suite", "load", "--filter", "numpy/inram", "--data", str(tmp_path / "data"),
                 "--out", str(tmp_path / "m.json")]) == 2
    assert "finished" not in json.loads((tmp_path / "m.json").read_text())["meta"]


def test_perfsuite_cases_measured_before_a_harness_failure_are_kept_mechanics(tmp_path, monkeypatch):
    from perfsuite import worker

    ds = Dataset.parse("2d-64x64-grid4-uint16-s0")
    cases = [presets.case("features", "2d", ds, "tiff-tiled", [f], "smoke") for f in ("MEAN", "MAX", "MIN")]

    def fake(batch, *a, **k):  # the worker measured the first two, then its monitor would not start
        done = {c["id"]: {"id": c["id"], "status": "ok", "times": [0.1], "peak_rss": [2**20]} for c in batch[:2]}
        return done, worker.HARNESS_FAILURE, tmp_path / "w.log", 0.1

    monkeypatch.setattr(runner, "_run_worker", fake)
    with pytest.raises(runner.HarnessFailure) as e:
        runner.run(cases, tmp_path / "data", tmp_path / "r.json", batch=3, timeout=5, log=lambda *_: None)
    assert [r["id"] for r in e.value.results] == [c["id"] for c in cases[:2]]
    saved = json.loads((tmp_path / "r.json").read_text())
    assert [r["id"] for r in saved["cases"]] == [c["id"] for c in cases[:2]] and "finished" not in saved["meta"]


def test_perfsuite_run_refuses_a_cli_without_checktiming_before_it_starts_mechanics(tmp_path, monkeypatch):
    import subprocess

    from perfsuite import checktiming

    plain = "Usage: nyxus --intDir=... --features=...\n"
    timed = plain + "\t\t(optional) --exclusivetiming=<false or true>\n"
    monkeypatch.setattr(checktiming.subprocess, "run",
                        lambda a, **k: subprocess.CompletedProcess(a, 1, stdout=help_text[0], stderr=""))
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("the Python suites started"))
    help_text = [plain]
    with pytest.raises(RuntimeError, match="not a CHECKTIMING build"):
        _run_main(tmp_path, "--cli", __file__)
    help_text[0] = timed
    assert checktiming.require_checktiming_cli(__file__) == Path(__file__)


def test_perfsuite_run_refuses_a_bench_that_is_not_one_before_it_starts_mechanics(tmp_path, monkeypatch):
    import subprocess

    from perfsuite import kernels

    monkeypatch.setattr(kernels.subprocess, "run",
                        lambda a, **k: subprocess.CompletedProcess(a, 0, stdout="usage: other tool\n", stderr=""))
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("the Python suites started"))
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    with pytest.raises(RuntimeError, match="lists no kernels"):
        _run_main(tmp_path, "--bench-exe", __file__)  # no --filter: refused all the same


def test_perfsuite_report_lists_every_case_that_is_not_ok_mechanics(tmp_path, capsys):
    from perfsuite.__main__ import main

    bad = {"id": "sweep/2d-x/tiff-tiled/inram/t4/all_glcm", "suite": "sweep", "status": "timeout",
           "error": "worker exit timeout | log w.log", "time": {"n": 0}, "features": ["*ALL_GLCM*"],
           "dataset": "2d-x", "sweep": {"name": "roi-count/all_glcm", "x": 65536}}
    path = _results(tmp_path / "r.json", [bad])
    assert main(["report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "## not ok (1)" in out and "`sweep/2d-x/tiff-tiled/inram/t4/all_glcm` | timeout" in out
    assert "worker exit timeout \\| log" in out  # a | in the error does not break the table


def test_perfsuite_kernel_floor_is_its_own_option_mechanics(tmp_path):
    from perfsuite.__main__ import main

    def k(med):
        return {"id": "kernels/gabor/side64", "suite": "kernels", "status": "ok", "value": 1.0,
                "time": {"median": med, "mad": 0.0}}

    old = _results(tmp_path / "o.json", [k(0.003)])
    new = _results(tmp_path / "n.json", [k(0.006)])
    assert main(["compare", str(old), str(new)]) == 1                           # 3 ms over the 0.5 ms floor
    assert main(["compare", str(old), str(new), "--kernel-floor", "0.005"]) == 0  # under a 5 ms one
    assert main(["compare", str(old), str(new), "--floor", "0.005"]) == 1       # --floor is the other suites'


def test_perfsuite_every_stage_logs_to_the_run_s_log_directory_mechanics(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "machine_info", lambda: {})
    monkeypatch.setattr(runner, "run", lambda cases, data, out, **k: seen.append((Path(out).name, k["log_dir"])) or [])
    _run_main(tmp_path, "--suite", "load", "--suite", "features", "--filter", "grid64")
    assert {name for name, _ in seen} == {"r.json", "r-features.json"}
    assert {d for _, d in seen} == {tmp_path / "r-logs"}


# --- host baselines -----------------------------------------------------------------------------

def test_perfsuite_baselines_live_on_the_host_outside_the_repository_mechanics(tmp_path, monkeypatch):
    from perfsuite import baseline
    from perfsuite.__main__ import main

    monkeypatch.setenv("PERFSUITE_BASELINES", str(tmp_path / "store"))

    def results(name, med, finished=True, status="ok", dirty=False):
        p = tmp_path / name
        meta = {"host": "box/1", "tier": "smoke", "git_sha": "x", "git_dirty": dirty,
                **({"finished": "t"} if finished else {})}
        p.write_text(json.dumps({"meta": meta, "cases": [_c("a", med, status=status)]}))
        return p

    first = results("first.json", 1.0)
    assert main(["baseline", "save", str(first)]) == 0
    stored = tmp_path / "store" / "box%2F1" / "smoke.json"
    assert stored.exists() and baseline.listing() == [("box/1", "smoke", stored)]
    # compare with one file judges it against its host's baseline
    assert main(["compare", str(results("slow.json", 2.0))]) == 1
    assert main(["compare", str(results("same.json", 1.0))]) == 0
    # a run that is no baseline is refused, unless forced; the one it replaces is kept
    for bad in (results("u.json", 1.0, finished=False), results("e.json", 1.0, status="error"),
                results("d.json", 1.0, dirty=True)):
        with pytest.raises(ValueError, match="not saved as a baseline"):
            baseline.save(bad)
    baseline.save(results("second.json", 1.5), force=True)
    assert json.loads(stored.read_text())["cases"][0]["time"]["median"] == 1.5
    assert json.loads((stored.parent / "prev" / "smoke.json").read_text())["cases"][0]["time"]["median"] == 1.0
    assert [h for h, _, _ in baseline.listing()] == ["box/1"]  # the kept one is not listed as a baseline
    # another host or tier has none
    other = tmp_path / "o.json"
    other.write_text(json.dumps({"meta": {"host": "elsewhere", "tier": "smoke", "finished": "t"}, "cases": []}))
    with pytest.raises(FileNotFoundError, match="no baseline for host 'elsewhere'"):
        baseline.find(other)
    # git that cannot read the checkout is not a clean one
    with pytest.raises(ValueError, match="git could not tell"):
        baseline.save(results("g.json", 1.0, dirty=None))


def test_perfsuite_host_names_never_share_a_baseline_mechanics(tmp_path, monkeypatch):
    from perfsuite import baseline

    monkeypatch.setenv("PERFSUITE_BASELINES", str(tmp_path / "store"))
    names = ["box/1", "box_1", "box%2F1", "a:b", "a_b", ".", "..", "", "ü", "BuildBox", "buildbox", "CON", "con.x",
             "nul", "build", "build.", "build ", "build.."]
    paths = {baseline.path_for(h, "smoke") for h in names}
    assert len(paths) == len(names) and all(p.parent.parent == tmp_path / "store" for p in paths)
    # distinct even where file names ignore case, and none a Windows device name
    assert len({str(p).lower() for p in paths}) == len(names)
    assert all(p.parent.name.split(".")[0].lower() not in ("con", "nul") for p in paths)
    assert not any(p.parent.name.endswith((".", " ")) for p in paths)  # Windows would drop it
    # a baseline file that names another host is refused, not compared across machines
    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps({"meta": {"host": "box_1", "tier": "smoke", "finished": "t"}, "cases": []}))
    p = baseline.path_for("box_1", "smoke")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"meta": {"host": "box/1", "tier": "smoke"}, "cases": []}))
    with pytest.raises(ValueError, match="compare only on one host"):
        baseline.find(mine)


def test_perfsuite_git_that_cannot_read_the_checkout_is_not_clean_mechanics(tmp_path):
    import shutil

    if not shutil.which("git"):
        pytest.skip("git not installed")
    assert runner.git_dirty(tmp_path) is None  # not a checkout: git status exits 128


def test_perfsuite_atomic_writes_do_not_share_a_temporary_file_mechanics(tmp_path, monkeypatch):
    import os
    import threading

    from perfsuite import fileio

    target = tmp_path / "r.json"
    blobs = [("x" * 200_000 + str(i)).encode() for i in range(8)]
    ts = [threading.Thread(target=fileio.write_atomic, args=(target, b)) for b in blobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert target.read_bytes() in blobs  # one writer's data whole, not a mix
    assert list(tmp_path.iterdir()) == [target]  # no temporary file left
    # a replace that keeps failing (Windows: the file is open elsewhere) is retried, then raised
    calls = []
    real = os.replace

    def busy(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(fileio.os, "replace", busy)
    fileio.write_atomic(target, b"new")
    assert target.read_bytes() == b"new" and len(calls) == 3
    monkeypatch.setattr(fileio.os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError("in use")))
    with pytest.raises(PermissionError):
        fileio.write_atomic(target, b"never", retry_s=0.1)
    assert target.read_bytes() == b"new" and list(tmp_path.iterdir()) == [target]


def test_perfsuite_results_and_baselines_stay_out_of_git_mechanics():
    from perfsuite import baseline

    here = Path(__file__).resolve().parents[1]
    assert "results/" in (here / ".gitignore").read_text().split()
    assert here not in baseline.store().resolve().parents  # the default store is outside the checkout


# --- instruction counts ---------------------------------------------------------------------------

def _counts(counts, compiler="gcc 13.3.0", libc="glibc 2.39", finished=True):
    meta = {"kind": "instructions", "toolchain": {"compiler": compiler, "libc": libc, "machine": "x86_64"},
            **({"finished": "t"} if finished else {})}
    return {"meta": meta, "counts": {k: {"instructions": n, "library_instructions": 0, "value": v}
                                     for k, (n, v) in counts.items()}}


def test_perfsuite_instruction_counts_compare_tightly_mechanics(tmp_path):
    from perfsuite import icount
    from perfsuite.__main__ import main

    old = _counts({"kernels/glcm/side64": (1_000_000, 5.0), "kernels/gabor/side64": (2_000_000, 1.0),
                   "kernels/hull/side64": (500_000, 2.0)})
    new = _counts({"kernels/glcm/side64": (1_005_000, 5.0), "kernels/gabor/side64": (2_100_000, 1.0),
                   "kernels/hull/side64": (450_000, 2.0)})
    # counts repeat exactly, so by default every change is one: +0.5 % and +5 % rise, -10 % falls
    r = icount.compare(old, new)
    assert [x["id"] for x in r["regressions"]] == ["kernels/gabor/side64", "kernels/glcm/side64"]
    assert [x["id"] for x in r["improvements"]] == ["kernels/hull/side64"]  # a fall fails too, until committed
    assert icount.failed(r)
    loose = icount.compare(old, new, tol=0.01)  # a tolerance passes the +0.5 % only
    assert [x["id"] for x in loose["regressions"]] == ["kernels/gabor/side64"]
    o, n = tmp_path / "o.json", tmp_path / "n.json"
    icount.save(o, old)
    icount.save(n, _counts({k: (v["instructions"], v["value"]) for k, v in old["counts"].items()}))
    assert main(["compare", str(o), str(n)]) == 0
    icount.save(n, _counts({**{k: (v["instructions"], v["value"]) for k, v in old["counts"].items()},
                            "kernels/glcm/side64": (1_000_001, 5.0)}))
    assert main(["compare", str(o), str(n)]) == 1  # the command's default too: one instruction more fails
    # a kernel with no committed count fails: nothing would ever judge it
    icount.save(n, _counts({**{k: (v["instructions"], v["value"]) for k, v in old["counts"].items()},
                            "kernels/new/side64": (10, 1.0)}))
    assert main(["compare", str(o), str(n)]) == 1
    # a changed value, a missing kernel, an unfinished count each fail
    for bad in (_counts({**{k: (v["instructions"], v["value"]) for k, v in old["counts"].items()},
                         "kernels/glcm/side64": (1_000_000, 6.0)}),
                _counts({"kernels/glcm/side64": (1_000_000, 5.0)}),
                _counts({k: (v["instructions"], v["value"]) for k, v in old["counts"].items()}, finished=False)):
        assert icount.failed(icount.compare(old, bad))
    # timings and counts are not compared with each other
    with pytest.raises(SystemExit, match="instruction counts and the other timings"):
        main(["compare", str(o), str(_results(tmp_path / "t.json", [_c("a", 1.0)]))])


def test_perfsuite_counts_from_another_toolchain_are_not_compared_mechanics():
    from perfsuite import icount

    old = _counts({"kernels/glcm/side64": (1_000_000, 5.0)})
    new = _counts({"kernels/glcm/side64": (1_000_000, 5.0)}, compiler="gcc 14.1.0")
    r = icount.compare(old, new)
    assert r["toolchain_change"] == {"compiler": ("gcc 13.3.0", "gcc 14.1.0")} and icount.failed(r)
    assert not icount.failed(icount.compare(old, new, allow_toolchain_change=True))


def _row(args, value):
    """The kernel row a benchmark run with these arguments prints: its own kernel and side."""
    k, s = args[args.index("--kernel") + 1], args[args.index("--side") + 1]
    return {"id": f"kernels/{k}/side{s}", "value": value}


def test_perfsuite_a_count_is_one_repetition_in_one_run_mechanics(monkeypatch):
    from perfsuite import icount, kernels

    monkeypatch.setattr(kernels, "names", lambda exe: {"glcm": 4096, "gabor": 64})
    monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir: {"compiler": "gcc"})
    runs = []

    def instructions(args, timeout):
        runs.append((args[args.index("--kernel") + 1], int(args[args.index("--side") + 1]),
                     args[args.index("--reps") + 1], "--no-warmup" in args and "--no-timing" in args))
        return 1_000, 40, _row(args, 0.5)

    monkeypatch.setattr(icount, "_instructions", instructions)
    data = icount.count(Path(__file__), Path(__file__).parent, log=lambda *_: None)
    one = {"instructions": 1_000, "library_instructions": 40, "value": 0.5}
    assert data["counts"] == {"kernels/glcm/side64": one, "kernels/glcm/side256": one, "kernels/gabor/side64": one}
    # one run per kernel and side, one repetition, no warm-up and no timing (or it does not repeat)
    assert sorted(runs) == [("gabor", 64, "1", True), ("glcm", 64, "1", True), ("glcm", 256, "1", True)]
    assert data["meta"]["kind"] == "instructions" and "finished" in data["meta"]
    # the valgrind options collect inside the repetition function only, from no earlier point
    assert "--collect-atstart=no" in icount.TOOL and f"--toggle-collect={icount.REPETITION}*" in icount.TOOL


def test_perfsuite_a_failed_valgrind_run_keeps_the_counts_that_finished_mechanics(tmp_path, monkeypatch):
    import threading

    from perfsuite import icount, kernels
    from perfsuite.__main__ import main

    monkeypatch.setattr(kernels, "names", lambda exe: {f"k{i}": 4096 for i in range(8)})
    monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir: {"compiler": "gcc"})
    started, gate = [], threading.Event()

    def instructions(args, timeout):
        k = args[args.index("--kernel") + 1]
        started.append(k)
        if k == "k0":
            raise RuntimeError("valgrind failed")
        return 10, 0, _row(args, 1.0)

    monkeypatch.setattr(icount, "_instructions", instructions)
    out = tmp_path / "icount.json"
    rc = main(["icount", "--bench-exe", __file__, "--build-dir", str(Path(__file__).parent), "--out", str(out),
               "--side", "64", "--jobs", "1"])
    data = json.loads(out.read_text())
    assert rc == 1 and "finished" not in data["meta"] and "valgrind failed" in data["meta"]["error"]
    assert len(started) < 8  # the queued runs were cancelled, not waited for
    assert all(v["instructions"] == 10 for v in data["counts"].values())  # what finished is saved


def test_perfsuite_committed_counts_name_their_toolchain_mechanics():
    from perfsuite import icount

    path = Path(__file__).resolve().parents[1] / "baselines" / "instructions.json"
    data = json.loads(path.read_text())
    assert icount.is_counts(data) and "finished" in data["meta"]
    assert {"git_sha", "git_dirty"} <= set(data["meta"])  # made by this counter, provenance and all
    assert set(icount.COMPARED) | set(icount.BUILD) <= set(data["meta"]["toolchain"])
    assert "binutils=" in data["meta"]["toolchain"]["packages"]
    assert data["meta"]["toolchain"]["gpu"] == "off" and data["meta"]["toolchain"]["build_type"] == "Release"
    assert data["counts"] and all(isinstance(v["instructions"], int) and v["instructions"] > 0
                                  for v in data["counts"].values())


def test_perfsuite_a_count_that_falls_fails_until_committed_mechanics():
    from perfsuite import icount

    old = _counts({"kernels/gabor/side256": (917_000_000, 1.0)})
    faster = _counts({"kernels/gabor/side256": (640_000_000, 1.0)})
    r = icount.compare(old, faster)
    assert [x["id"] for x in r["improvements"]] == ["kernels/gabor/side256"] and icount.failed(r)
    assert "commit the new counts" in icount.format_report(r)
    # committed, the next rise is judged against the new count, not the stale one
    assert icount.failed(icount.compare(faster, _counts({"kernels/gabor/side256": (900_000_000, 1.0)})))


def test_perfsuite_toolchain_covers_how_the_benchmark_was_built_mechanics(tmp_path):
    from perfsuite import icount

    old = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    # the toolchain proper: another one is not compared at all
    for key, value in (("packages", "libc6=2.39-0ubuntu8.13"), ("valgrind", "valgrind-3.24.0"),
                       ("libc", "glibc 2.40"), ("compiler", "gcc 13.3.1"), ("machine", "aarch64")):
        new = _counts({"kernels/glcm/side64": (1_000, 5.0)})
        new["meta"]["toolchain"][key] = value
        assert key in icount.compare(old, new)["toolchain_change"], key
    # the build configuration: part of the change, reported, and the counts still compared
    for key, value in (("compile_hash", "0" * 64), ("gpu", "on"), ("checktiming", "off"), ("build_type", "Debug")):
        new = _counts({"kernels/glcm/side64": (1_000, 5.0)})
        new["meta"]["toolchain"][key] = value
        r = icount.compare(old, new)
        assert key in r["build_change"] and not r["toolchain_change"] and not icount.failed(r), key
    # the CPU features are recorded, not compared: the executable's own instructions do not depend on
    # them, and the library instructions run with glibc's routine choice pinned
    new = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    new["meta"]["toolchain"]["cpu_features"] = "avx"
    assert not icount.compare(old, new)["toolchain_change"]
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t\t: fpu sse2 avx avx2 fma bmi2 avx512f\n")
    assert icount.cpu_features(cpuinfo) == "avx avx2 fma bmi2"
    assert icount.cpu_features(tmp_path / "none") == "unknown"


def test_perfsuite_executable_and_library_instructions_are_counted_apart_mechanics(tmp_path):
    from perfsuite import icount

    exe = tmp_path / "bench_feature_kernels"
    exe.write_text("")
    out = (f"version: 1\ncreator: callgrind-3.22.0\npositions: line\nevents: Ir\nsummary: 999\n\n"
           f"ob={exe}\nfl=???\nfn=kernel\n0 100\n0 20\n"
           "cob=/usr/lib/x86_64-linux-gnu/libm.so.6\ncfn=exp\ncalls=3 0\n0 5000\n"  # inclusive: libm's own
           "0 7\n\n"
           "ob=/usr/lib/x86_64-linux-gnu/libm.so.6\nfl=???\nfn=exp\n0 5000\n\n"
           f"ob={exe}\nfn=other\n0 3\n")
    assert icount.split_counts(out, exe) == (100 + 20 + 7 + 3, 5000)
    with pytest.raises(RuntimeError, match="recorded no code"):
        icount.split_counts(out, tmp_path / "another")


def test_perfsuite_work_moved_into_a_library_is_caught_mechanics():
    from perfsuite import icount

    def c(own, lib):
        d = _counts({"kernels/gabor/side64": (own, 1.0)})
        d["counts"]["kernels/gabor/side64"]["library_instructions"] = lib
        return d

    base = c(53_597_823, 1_063_967)
    # exp() called twice as often: the executable's count barely moves, the libraries' doubles
    r = icount.compare(base, c(53_597_823, 2_127_934))
    assert not r["regressions"] and [x["id"] for x in r["library_changes"]] == ["kernels/gabor/side64"]
    assert icount.failed(r)
    # a residue under 0.5 % of the whole is absorbed; the 2 % an unpinned glibc moved it by is not
    assert not icount.failed(icount.compare(base, c(53_597_823, 1_063_967 + 200_000)))
    assert icount.failed(icount.compare(base, c(53_597_823, 1_063_967 - 443_000)))
    assert "x86_non_temporal_threshold" in icount.GLIBC_PIN and "-SSE4_2" in icount.GLIBC_PIN


def test_perfsuite_valgrind_runs_with_glibc_s_routine_choice_pinned_mechanics(tmp_path, monkeypatch):
    import subprocess

    from perfsuite import icount, kernels

    exe = tmp_path / "bench_feature_kernels"
    exe.write_text("")
    seen = {}

    def run(args, env=None, **kw):
        seen["env"], seen["args"] = env, args
        out = next(a.split("=", 1)[1] for a in args if a.startswith("--callgrind-out-file="))
        Path(out).write_text(f"summary: 9\nob={exe}\nfn=bench_kernel\n0 9\n")
        return subprocess.CompletedProcess(args, 0, stdout=kernels.HEADER + "\nglcm,64,1,1,0,0,0,0,1.5\n", stderr="")

    monkeypatch.setattr(icount, "_valgrind", lambda: "valgrind")
    monkeypatch.setattr(icount.subprocess, "run", run)
    own, lib, row = icount._instructions([str(exe), "--kernel", "glcm"], 60)
    assert (own, lib, row["value"]) == (9, 0, 1.5)
    assert seen["env"]["GLIBC_TUNABLES"] == icount.GLIBC_PIN
    assert all(t in seen["args"] for t in icount.TOOL)


def test_perfsuite_rise_and_fall_thresholds_mirror_each_other_mechanics():
    from perfsuite import icount

    old = _counts({"kernels/a/side64": (1_000, 1.0)})
    up = icount.compare(old, _counts({"kernels/a/side64": (2_001, 1.0)}), tol=1.0)
    down = icount.compare(old, _counts({"kernels/a/side64": (499, 1.0)}), tol=1.0)
    assert up["regressions"] and down["improvements"]  # past 2x either way
    assert not icount.failed(icount.compare(old, _counts({"kernels/a/side64": (501, 1.0)}), tol=1.0))


def test_perfsuite_a_count_must_be_positive_mechanics(monkeypatch):
    from perfsuite import icount

    monkeypatch.setattr(icount, "_instructions", lambda args, timeout: (0, 50, _row(args, 1.0)))
    with pytest.raises(RuntimeError, match="no instructions collected"):
        icount._one(Path(__file__), "glcm", 64, 60)
    r = icount.compare(_counts({"kernels/a/side64": (0, 1.0)}), _counts({"kernels/a/side64": (0, 1.0)}))
    assert r["invalid"] == [("kernels/a/side64", 0, 0)] and not r["regressions"] and icount.failed(r)


def _build_dir(root, src, flags="-O2", backend=False):
    root.mkdir(parents=True)
    (root / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={src}\n")
    cmds = [{"directory": str(root), "file": str(src / "src" / "nyx" / f),
             "command": f"/usr/bin/c++ -I{src}/src {flags} -o CMakeFiles/nyxus_core.dir/{f}.o -c {src}/src/nyx/{f}"}
            for f in ("a.cpp", "b.cpp")]
    if backend:  # the Python backend compiles the same sources with its own definitions
        cmds += [{"directory": str(root), "file": str(src / "src" / "nyx" / f),
                  "command": f"/usr/bin/c++ -DWITH_PYTHON_H {flags} -o CMakeFiles/backend.dir/{f}.o -c {src}/src/nyx/{f}"}
                 for f in ("a.cpp", "b.cpp")]
    (root / "compile_commands.json").write_text(json.dumps(cmds))
    return root


def test_perfsuite_the_compile_hash_follows_flags_not_paths_mechanics(tmp_path):
    from perfsuite import icount

    a = icount.compile_hash(_build_dir(tmp_path / "one" / "build", tmp_path / "one"))
    b = icount.compile_hash(_build_dir(tmp_path / "two" / "build", tmp_path / "two"))
    c = icount.compile_hash(_build_dir(tmp_path / "three" / "build", tmp_path / "three", flags="-O2 -flto"))
    d = icount.compile_hash(_build_dir(tmp_path / "four" / "build", tmp_path / "four", backend=True))
    assert a == b and a != c  # another checkout path hashes the same; another flag does not
    assert d == a  # other targets compiling the same sources are not the code a count runs
    with pytest.raises(FileNotFoundError, match="compile_commands.json"):
        icount.compile_hash(tmp_path)


def test_perfsuite_a_build_configuration_change_is_judged_by_its_counts_mechanics(tmp_path):
    from perfsuite import icount
    from perfsuite.__main__ import main

    o, n = tmp_path / "o.json", tmp_path / "n.json"
    old = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    old["meta"]["toolchain"]["compile_hash"] = "a" * 64
    icount.save(o, old)
    same = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    same["meta"]["toolchain"]["compile_hash"] = "b" * 64  # a new definition that changes no code
    icount.save(n, same)
    assert main(["compare", str(o), str(n)]) == 0
    moved = _counts({"kernels/glcm/side64": (1_010, 5.0)})
    moved["meta"]["toolchain"]["compile_hash"] = "b" * 64
    icount.save(n, moved)
    assert main(["compare", str(o), str(n)]) == 1  # a regression, not "toolchain changed" (3)
    assert "build configuration differs" in icount.format_report(icount.compare(old, moved))


def test_perfsuite_the_compile_hash_reads_paths_as_written_and_resolved_mechanics(tmp_path):
    from perfsuite import icount

    plain = icount.compile_hash(_build_dir(tmp_path / "a" / "build", tmp_path / "a"))
    # configured through a path that resolves elsewhere (a symlink, or here a ".."): the commands
    # carry that spelling, which must be taken out as surely as the resolved one
    roundabout = tmp_path / "b" / "build" / ".."
    (tmp_path / "b").mkdir()
    assert icount.compile_hash(_build_dir(tmp_path / "b" / "build", Path(str(roundabout)))) == plain


def test_perfsuite_the_bench_must_come_from_the_build_directory_mechanics(tmp_path, monkeypatch):
    from perfsuite import icount, kernels

    monkeypatch.setattr(kernels, "names", lambda exe: {"glcm": 4096})
    with pytest.raises(ValueError, match="is not in"):
        icount.count(Path(__file__), tmp_path, log=lambda *_: None)


def test_perfsuite_count_errors_are_reported_in_plan_order_mechanics(monkeypatch):
    from perfsuite import icount, kernels

    monkeypatch.setattr(kernels, "names", lambda exe: {"a": 4096, "b": 4096, "c": 4096})
    monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir: {"compiler": "gcc"})

    import threading
    import time

    started = threading.Barrier(3, timeout=5)

    def instructions(args, timeout):
        k = args[args.index("--kernel") + 1]
        if jobs[0] == 3:
            started.wait()  # all three running before any fails
        if k == "b":
            time.sleep(0.2)  # b fails last by the clock, though it is first in the plan
            raise RuntimeError("b broke")
        if k == "c":
            raise RuntimeError("c broke")
        return 10, 0, _row(args, 1.0)

    monkeypatch.setattr(icount, "_instructions", instructions)
    here = Path(__file__)
    jobs = [1]
    # one at a time, c never starts once b has failed
    data = icount.count(here, here.parent, sides=(64,), jobs=1, log=lambda *_: None)
    assert data["meta"]["error"] == "kernels/b/side64: b broke"
    # three at once, c fails first and b after it: b, the first failure in the plan, is reported
    jobs[0] = 3
    data = icount.count(here, here.parent, sides=(64,), jobs=3, log=lambda *_: None)
    assert data["meta"]["error"] == "kernels/b/side64: b broke (and 1 more)"
    assert list(data["counts"]) == ["kernels/a/side64"]


def test_perfsuite_counts_name_the_commit_they_count_mechanics(tmp_path):
    import shutil
    import subprocess

    from perfsuite import icount

    if not shutil.which("git"):
        pytest.skip("git not installed")
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=src, check=True, capture_output=True)
    build = _build_dir(src / "build", src)
    sha, dirty = icount.provenance(build)
    assert sha is None and dirty is True  # a checkout with no commit yet, and new files in it
    assert icount.provenance(tmp_path) == (None, None)  # no CMake cache: nothing to tell


def test_perfsuite_icount_setup_problems_are_messages_mechanics(tmp_path, monkeypatch):
    from perfsuite import icount, kernels
    from perfsuite.__main__ import main

    monkeypatch.setattr(kernels, "names", lambda exe: {"glcm": 4096})
    for err in (FileNotFoundError("valgrind not found"), RuntimeError("--build-info printed no compiler")):
        monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir, e=err: (_ for _ in ()).throw(e))
        with pytest.raises(SystemExit, match="icount: "):
            main(["icount", "--bench-exe", __file__, "--build-dir", str(Path(__file__).parent),
                  "--out", str(tmp_path / "x.json")])


def test_perfsuite_ci_python_packages_are_pinned_mechanics():
    req = Path(__file__).resolve().parents[1] / "requirements-ci.txt"
    pins = {ln.split("==")[0].lower(): ln for ln in req.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    assert pins and all("==" in ln for ln in pins.values())
    # the complete set, dependencies too (the job installs it with --no-deps), each file by its hash
    assert {"numcodecs", "donfig", "pluggy", "iniconfig", "python-dateutil", "tzdata"} <= set(pins)
    assert all(" --hash=sha256:" in ln for ln in pins.values())
    wf = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "build_and_test_ubuntu.yml").read_text()
    assert "--no-deps --require-hashes -r" in wf


def test_perfsuite_store_files_that_are_not_baselines_are_skipped_mechanics(tmp_path, monkeypatch):
    from perfsuite import baseline

    monkeypatch.setenv("PERFSUITE_BASELINES", str(tmp_path))
    (tmp_path / "h").mkdir()
    for name, text in (("list.json", "[]"), ("nullmeta.json", '{"meta": null}'), ("bad.json", "{"),
                       ("good.json", '{"meta": {"host": "h", "tier": "smoke"}, "cases": []}')):
        (tmp_path / "h" / name).write_text(text)
    assert [(h, t) for h, t, _ in baseline.listing()] == [("h", "smoke")]


def test_perfsuite_git_provenance_is_none_outside_a_checkout_mechanics(tmp_path):
    import shutil

    if not shutil.which("git"):
        pytest.skip("git not installed")
    assert runner.git_sha(tmp_path) is None and runner.git_dirty(tmp_path) is None
    here = runner.git_sha(Path(__file__).parent)
    assert here is None or len(here) == 40  # the full SHA, as instruction counts record it


def test_perfsuite_the_compile_hash_covers_the_link_settings_mechanics(tmp_path):
    from perfsuite import icount

    def build(name, cache_extra="", ninja_flags=None):
        b = _build_dir(tmp_path / name / "build", tmp_path / name)
        with open(b / "CMakeCache.txt", "a") as f:
            f.write("CMAKE_EXE_LINKER_FLAGS:STRING=\n" + cache_extra)
        if ninja_flags is not None:
            (b / "build.ninja").write_text("build bench_feature_kernels: CXX_EXECUTABLE_LINKER__x a.o\n"
                                           f"  LINK_FLAGS = {ninja_flags}\n  LINK_PATH =\n\nbuild other: phony\n")
        return icount.compile_hash(b)

    base = build("a", ninja_flags="")
    assert base == build("b", ninja_flags="")
    assert base != build("c", cache_extra="CMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=ON\n", ninja_flags="")
    assert base != build("d", ninja_flags="-flto=auto")
    assert icount.link_config(tmp_path / "d" / "build") == ["CMAKE_EXE_LINKER_FLAGS=", "LINK_FLAGS=-flto=auto"]


@pytest.mark.skipif(os.name == "nt", reason="the umask is POSIX")
def test_perfsuite_atomic_writes_respect_the_umask_mechanics(tmp_path):
    from perfsuite import fileio

    old = os.umask(0o077)
    try:
        fileio.write_atomic(tmp_path / "new.json", b"x")
    finally:
        os.umask(old)
    assert (tmp_path / "new.json").stat().st_mode & 0o777 == 0o600
    (tmp_path / "kept.json").write_bytes(b"x")
    os.chmod(tmp_path / "kept.json", 0o640)
    fileio.write_atomic(tmp_path / "kept.json", b"y")
    assert (tmp_path / "kept.json").stat().st_mode & 0o777 == 0o640  # a replaced file keeps its mode


def test_perfsuite_counts_made_another_way_are_not_compared_mechanics():
    from perfsuite import icount

    old = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    new = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    old["meta"]["method"] = "valgrind callgrind ...; GLIBC_TUNABLES=a"
    new["meta"]["method"] = "valgrind callgrind ...; GLIBC_TUNABLES=b"  # the pin or the options moved
    r = icount.compare(old, new)
    assert r["method_change"] and not r["toolchain_change"] and icount.failed(r)
    assert "COUNTING METHOD CHANGED" in icount.format_report(r) and "TOOLCHAIN" not in icount.format_report(r)


def test_perfsuite_link_settings_hash_alike_from_any_build_directory_mechanics(tmp_path):
    from perfsuite import icount

    def build(name):
        # an out-of-tree build: the build directory is not under the source directory
        b = _build_dir(tmp_path / name / "out", tmp_path / name / "src")
        (b / "build.ninja").write_text("build bench_feature_kernels: CXX_EXECUTABLE_LINKER__x a.o\n"
                                       f"  LINK_FLAGS = -Wl,-Map,{b}/bench.map -Wl,-rpath,{tmp_path / name / 'src'}/lib\n\n")
        return icount.compile_hash(b)

    assert build("head") == build("base")  # the build and source paths in the link line are taken out


def test_perfsuite_a_stored_baseline_that_is_not_results_is_refused_mechanics(tmp_path, monkeypatch):
    from perfsuite import baseline
    from perfsuite.__main__ import main

    monkeypatch.setenv("PERFSUITE_BASELINES", str(tmp_path / "store"))
    mine = _results(tmp_path / "r.json", [_c("a", 1.0)])
    stored = baseline.path_for("h", "?")  # _results writes host "h" and no tier
    stored.parent.mkdir(parents=True)
    stored.write_text("{}")
    monkeypatch.setattr(baseline, "path_for", lambda host, tier: stored)
    with pytest.raises(SystemExit, match="not a perfsuite results file"):
        main(["compare", str(mine)])


def test_perfsuite_an_unusable_count_fails_before_its_toolchain_is_judged_mechanics(tmp_path):
    from perfsuite import icount
    from perfsuite.__main__ import main

    o, n = tmp_path / "o.json", tmp_path / "n.json"
    icount.save(o, _counts({"kernels/glcm/side64": (1_000, 5.0)}))
    other = _counts({"kernels/glcm/side64": (1_000, 5.0)}, compiler="gcc 14.1.0", finished=False)
    icount.save(n, other)
    assert main(["compare", str(o), str(n)]) == 1  # unfinished: a failure, not "another toolchain" (3)
    other["meta"]["finished"] = "t"
    icount.save(n, other)
    assert main(["compare", str(o), str(n)]) == 3


def test_perfsuite_a_count_without_its_kernel_row_fails_mechanics(tmp_path, monkeypatch):
    import subprocess

    from perfsuite import icount

    exe = tmp_path / "bench_feature_kernels"
    exe.write_text("")

    def run_printing(stdout):
        def run(args, env=None, **kw):
            out = next(a.split("=", 1)[1] for a in args if a.startswith("--callgrind-out-file="))
            Path(out).write_text(f"summary: 9\nob={exe}\nfn=bench_kernel\n0 9\n")
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
        return run

    monkeypatch.setattr(icount, "_valgrind", lambda: "valgrind")
    monkeypatch.setattr(icount.subprocess, "run", run_printing("a banner, then nothing\n"))
    with pytest.raises(RuntimeError, match="printed no kernel row"):
        icount._instructions([str(exe), "--kernel", "glcm", "--side", "64"], 60)
    # a row for another kernel or side is not this one's count
    from perfsuite import kernels
    monkeypatch.setattr(icount.subprocess, "run", run_printing(kernels.HEADER + "\ngabor,64,1,1,0,0,0,0,1.5\n"))
    with pytest.raises(RuntimeError, match="reported kernels/gabor/side64 instead"):
        icount._one(exe, "glcm", 64, 60)


def test_perfsuite_a_forced_baseline_without_host_or_tier_is_found_mechanics(tmp_path, monkeypatch):
    from perfsuite import baseline

    monkeypatch.setenv("PERFSUITE_BASELINES", str(tmp_path / "store"))
    r = tmp_path / "r.json"
    r.write_text(json.dumps({"meta": {"finished": "t", "git_dirty": False}, "cases": [_c("a", 1.0)]}))
    saved = baseline.save(r, force=True)  # refused unforced: no host or tier
    assert baseline.find(r)[0] == saved


def test_perfsuite_counts_name_the_pull_request_head_they_came_from_mechanics(monkeypatch):
    from perfsuite import icount, kernels

    monkeypatch.setattr(kernels, "names", lambda exe: {"glcm": 4096})
    monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir: {"compiler": "gcc"})
    monkeypatch.setattr(icount, "_instructions", lambda args, timeout: (10, 0, _row(args, 1.0)))
    here = Path(__file__)
    assert icount.count(here, here.parent, sides=(64,), source_sha="abc", log=lambda *_: None)["meta"]["source_sha"] == "abc"
    assert "source_sha" not in icount.count(here, here.parent, sides=(64,), log=lambda *_: None)["meta"]


def test_perfsuite_compare_s_failure_rule_lives_with_its_keys_mechanics():
    keys = {"regressions": [], "broken": [], "value_changes": [], "incomplete": False, "missing": []}
    assert not compare.failed(keys)
    for k, v in (("regressions", [1]), ("broken", [1]), ("value_changes", [1]), ("incomplete", True), ("missing", [1])):
        assert compare.failed({**keys, k: v}), k


def test_perfsuite_no_flag_compares_counts_made_another_way_mechanics():
    from perfsuite import icount

    old = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    new = _counts({"kernels/glcm/side64": (1_000, 5.0)}, compiler="gcc 14.1.0")
    old["meta"]["method"], new["meta"]["method"] = "pin a", "pin b"
    r = icount.compare(old, new, allow_toolchain_change=True)
    assert not r["toolchain_change"] and r["method_change"] and icount.failed(r)  # the compiler is waived, the method not


def test_perfsuite_icount_says_when_the_bench_cannot_run_or_answer_mechanics(tmp_path, monkeypatch):
    import subprocess

    from perfsuite import icount, kernels
    from perfsuite.__main__ import main

    monkeypatch.setattr(kernels, "names", lambda exe: {"glcm": 4096})
    for err in (subprocess.TimeoutExpired(["bench", "--build-info"], 60), PermissionError("not executable")):
        monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir, e=err: (_ for _ in ()).throw(e))
        with pytest.raises(SystemExit, match="icount: "):
            main(["icount", "--bench-exe", __file__, "--build-dir", str(Path(__file__).parent),
                  "--out", str(tmp_path / "x.json")])


def test_perfsuite_makefile_link_settings_are_hashed_mechanics(tmp_path):
    from perfsuite import icount

    def build(name, flags):
        b = _build_dir(tmp_path / name / "out", tmp_path / name / "src")
        d = b / "CMakeFiles" / "bench_feature_kernels.dir"
        d.mkdir(parents=True)
        (d / "link.txt").write_text(f"/usr/bin/c++ -O2 {flags} CMakeFiles/bench_feature_kernels.dir/benchmarks/"
                                    "bench_feature_kernels.cpp.o -o bench_feature_kernels libnyxus_core.a -ltiff\n")
        return icount.compile_hash(b)

    assert build("a", "-Wl,-O1") == build("b", "-Wl,-O1")  # the inputs and output are left out
    assert build("c", "") != build("d", "-Wl,-O1")


def test_perfsuite_an_unfinished_reference_count_is_refused_mechanics(tmp_path):
    from perfsuite import icount
    from perfsuite.__main__ import main

    o, n = tmp_path / "o.json", tmp_path / "n.json"
    broken = _counts({"kernels/glcm/side64": (1_000, 5.0)})
    broken["meta"]["error"] = "kernels/gabor/side64: valgrind failed"  # an artifact of a failed run
    icount.save(o, broken)
    icount.save(n, _counts({"kernels/glcm/side64": (1_000, 5.0), "kernels/gabor/side64": (2_000, 1.0)}))
    r = icount.compare(json.loads(o.read_text()), json.loads(n.read_text()))
    assert r["old_incomplete"] and "REFERENCE INCOMPLETE" in icount.format_report(r)
    assert main(["compare", str(o), str(n)]) == 1


def test_perfsuite_the_compile_hash_resolves_each_directory_once_mechanics(tmp_path, monkeypatch):
    from perfsuite import icount

    b = _build_dir(tmp_path / "x" / "build", tmp_path / "x")
    cmds = json.loads((b / "compile_commands.json").read_text()) * 50  # 100 entries, one directory
    (b / "compile_commands.json").write_text(json.dumps(cmds))
    calls = [0]
    real = icount.os.path.realpath

    def counting(p):
        calls[0] += 1
        return real(p)

    monkeypatch.setattr(icount.os.path, "realpath", counting)
    icount.compile_hash(b)
    assert calls[0] < 10


def test_perfsuite_profile_rows_are_filtered_once_each_mechanics(tmp_path, monkeypatch):
    from perfsuite import checktiming

    monkeypatch.setattr(checktiming, "run_cli", lambda *a, **k: (1.0, {f"G/{i}": 0.1 for i in range(20)}, 0))
    seen = []
    names = [f"2d-64x64-grid4-uint16-s{i}" for i in range(5)]
    rows = checktiming.profile(Path(__file__), names, tmp_path, repeats=1, log_dir=tmp_path,
                               keep_row=lambda cid: seen.append(cid) or True, log=lambda *_: None)
    assert len(rows) == 5 * 21 and len(seen) == len(rows) and len(set(seen)) == len(seen)


def test_perfsuite_workers_started_together_keep_their_own_logs_mechanics(tmp_path, monkeypatch):
    script = tmp_path / "w.py"
    script.write_text("import sys; print('worker', sys.argv[1]); sys.exit(3)\n")
    monkeypatch.setattr(runner.time, "time", lambda: 1_700_000_000.0)  # the same millisecond, every time
    logs = [runner._run_worker([{"id": "c"}], 30, tmp_path, poll=0.05, worker=[sys.executable, str(script)])[2]
            for _ in range(2)]
    assert logs[0] != logs[1] and all(p.exists() for p in logs)


def test_perfsuite_the_committed_counts_are_of_the_code_users_build_mechanics():
    data = json.loads((Path(__file__).resolve().parents[1] / "baselines" / "instructions.json").read_text())
    assert data["meta"]["toolchain"]["checktiming"] == "off"
    wf = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "build_and_test_ubuntu.yml").read_text()
    bench = next(ln for ln in wf.splitlines() if ln.strip().startswith("BENCH_CMAKE:"))
    assert "-DCHECKTIMING=OFF" in bench


def test_perfsuite_a_relative_build_directory_hashes_as_an_absolute_one_mechanics(tmp_path, monkeypatch):
    from perfsuite import icount

    b = _build_dir(tmp_path / "src" / "build", tmp_path / "src")
    with open(b / "CMakeCache.txt", "a") as f:
        f.write("CMAKE_EXE_LINKER_FLAGS:STRING=-Wl,--build-id\n")  # holds the word "build"
    absolute = icount.compile_hash(b)
    monkeypatch.chdir(tmp_path / "src")
    assert icount.compile_hash("build") == absolute
    assert "-Wl,--build-id" in "\n".join(icount.link_config("build"))


def test_perfsuite_callgrind_object_names_are_resolved_once_each_mechanics(tmp_path, monkeypatch):
    from perfsuite import icount

    exe = tmp_path / "bench_feature_kernels"
    exe.write_text("")
    out = "".join(f"ob={exe}\nfn=f{i}\n0 1\nob=/usr/lib/libm.so.6\nfn=g{i}\n0 1\n" for i in range(500))
    calls = [0]
    real = icount.os.path.realpath
    monkeypatch.setattr(icount.os.path, "realpath", lambda p: calls.__setitem__(0, calls[0] + 1) or real(p))
    assert icount.split_counts(out, exe) == (500, 500)
    assert calls[0] <= 3  # the executable, and each of the two object names once


def test_perfsuite_results_are_saved_now_and_then_and_at_the_end_mechanics(tmp_path, monkeypatch):
    ds = Dataset.parse("2d-64x64-grid4-uint16-s0")
    cases = [presets.case("features", "2d", ds, "tiff-tiled", [f"F{i}"], "smoke") for i in range(40)]
    monkeypatch.setattr(runner, "_run_worker", lambda batch, *a, **k: (
        {c["id"]: {"id": c["id"], "status": "ok", "times": [0.1], "peak_rss": [1]} for c in batch}, 0,
        tmp_path / "w.log", 0.0))
    clock = [0.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 1.0) or clock[0])
    saves = []
    monkeypatch.setattr(runner, "_save", lambda path, meta, results: saves.append(("finished" in meta, len(results))))
    res = runner.run(cases, tmp_path / "data", tmp_path / "r.json", batch=1, timeout=5, log=lambda *_: None)
    assert len(res) == 40
    assert len(saves) < 40 / 2 and saves[-1] == (True, 40)  # not after every case; the last one finished


def test_perfsuite_a_filter_that_selects_nothing_is_refused_mechanics(tmp_path, monkeypatch):
    from perfsuite import icount, kernels
    from perfsuite.__main__ import main

    monkeypatch.setattr(kernels, "names", lambda exe: {"glcm": 4096})
    monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir: {"compiler": "gcc"})
    out = tmp_path / "icount.json"
    with pytest.raises(SystemExit, match="select none"):
        main(["icount", "--bench-exe", __file__, "--build-dir", str(Path(__file__).parent), "--out", str(out),
              "--filter", "glmc"])
    assert not out.exists()  # no empty, finished count to compare a subset with


def test_perfsuite_a_count_without_its_library_instructions_is_invalid_mechanics():
    from perfsuite import icount

    old = _counts({"kernels/a/side64": (1_000, 1.0)})
    del old["counts"]["kernels/a/side64"]["library_instructions"]
    r = icount.compare(old, _counts({"kernels/a/side64": (1_000, 1.0)}))
    assert r["invalid"] and not r["library_changes"] and icount.failed(r)


def test_perfsuite_compare_takes_parsed_results_too_mechanics(tmp_path):
    old = {"meta": {"host": "h", "finished": "t"}, "cases": [_c("a", 1.0)]}
    new = {"meta": {"host": "h", "finished": "t"}, "cases": [_c("a", 2.0)]}
    assert [x["id"] for x in compare.compare(old, new)["regressions"]] == ["a"]


def test_perfsuite_package_versions_come_from_dpkg_mechanics(monkeypatch):
    import subprocess

    from perfsuite import icount

    monkeypatch.setattr(icount.shutil, "which", lambda name: "/usr/bin/dpkg-query" if name == "dpkg-query" else None)
    seen = {}

    def run(a, **k):
        seen["args"] = a
        return subprocess.CompletedProcess(a, 1, stderr="", stdout=(
            "ii libc6=2.39-0ubuntu8.12\nii gcc-13=13.3.0-6ubuntu2~24.04.1\nii binutils=2.42-4ubuntu2.5\n"
            "un gcc-12=\nrc gcc-11=11.4.0-9ubuntu1\nii valgrind=\n"))  # not installed / removed / empty

    monkeypatch.setattr(icount.subprocess, "run", run)
    assert icount.package_versions() == "binutils=2.42-4ubuntu2.5 gcc-13=13.3.0-6ubuntu2~24.04.1 libc6=2.39-0ubuntu8.12"
    assert {"gcc-[0-9]*", "binutils", "cmake"} <= set(seen["args"])  # any compiler version, assembler, linker
    monkeypatch.setattr(icount.shutil, "which", lambda name: None)
    assert icount.package_versions() == "unknown"


def test_perfsuite_compare_says_when_only_the_toolchain_differs_mechanics(tmp_path):
    from perfsuite import icount
    from perfsuite.__main__ import TOOLCHAIN_CHANGED, main

    o, n = tmp_path / "o.json", tmp_path / "n.json"
    icount.save(o, _counts({"kernels/glcm/side64": (1_000, 5.0)}))
    icount.save(n, _counts({"kernels/glcm/side64": (1_000, 5.0)}, compiler="gcc 13.3.1"))
    assert main(["compare", str(o), str(n)]) == TOOLCHAIN_CHANGED == 3  # the CI job's cue to count the base
    icount.save(n, _counts({"kernels/glcm/side64": (1_100, 5.0)}))
    assert main(["compare", str(o), str(n)]) == 1


def test_perfsuite_a_narrowed_count_compares_with_allow_missing_mechanics(tmp_path):
    from perfsuite import icount
    from perfsuite.__main__ import main

    o, n = tmp_path / "o.json", tmp_path / "n.json"
    icount.save(o, _counts({"kernels/glcm/side64": (1_000, 5.0), "kernels/gabor/side64": (2_000, 1.0)}))
    icount.save(n, _counts({"kernels/gabor/side64": (2_000, 1.0)}))
    assert main(["compare", str(o), str(n)]) == 1
    assert main(["compare", str(o), str(n), "--allow-missing"]) == 0


def test_perfsuite_count_files_are_not_host_baselines_mechanics(tmp_path, monkeypatch, capsys):
    from perfsuite import icount
    from perfsuite.__main__ import main

    monkeypatch.setenv("PERFSUITE_BASELINES", str(tmp_path / "store"))
    n = tmp_path / "icount.json"
    icount.save(n, _counts({"kernels/glcm/side64": (1_000, 5.0)}))
    for argv in (["compare", str(n)], ["baseline", "save", str(n)]):
        with pytest.raises(SystemExit, match="baselines/instructions.json"):
            main(argv)


def test_perfsuite_parallel_counts_equal_serial_ones_mechanics(monkeypatch):
    import threading
    import time

    from perfsuite import icount, kernels

    monkeypatch.setattr(kernels, "names", lambda exe: {f"k{i}": 4096 for i in range(6)})
    monkeypatch.setattr(icount, "toolchain", lambda exe, build_dir: {"compiler": "gcc"})
    live, most = [0], [0]
    lock = threading.Lock()

    def instructions(args, timeout):
        with lock:
            live[0] += 1
            most[0] = max(most[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        k = args[args.index("--kernel") + 1]
        return 100 * (int(k[1:]) + 1), int(k[1:]), _row(args, float(k[1:]))

    monkeypatch.setattr(icount, "_instructions", instructions)
    here = Path(__file__)
    serial = icount.count(here, here.parent, jobs=1, log=lambda *_: None)["counts"]
    assert most[0] == 1
    parallel = icount.count(here, here.parent, jobs=4, log=lambda *_: None)["counts"]
    assert most[0] > 1 and parallel == serial and list(parallel) == list(serial)


CLI_EXE = os.environ.get("PERFSUITE_CLI")


@pytest.mark.skipif(not BENCH_EXE, reason="set PERFSUITE_BENCH_EXE to a built bench_feature_kernels")
def test_perfsuite_bench_repetitions_and_build_info_mechanics():
    import subprocess

    from perfsuite import icount, kernels

    p = subprocess.run([BENCH_EXE, "--kernel", "glcm", "--side", "64", "--reps", "2", "--no-warmup"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and kernels.parse(p.stdout)[0]["repeats"] == 2
    info = icount.build_info(BENCH_EXE)
    assert info["compiler"] not in ("", "unknown") and info["gpu"] in ("on", "off")
    assert info["checktiming"] in ("on", "off") and info["build_type"] not in ("", "unknown")  # set by CMake
    assert (Path(BENCH_EXE).parent / "compile_commands.json").is_file()  # exported for icount's compile hash
    p = subprocess.run([BENCH_EXE, "--kernel", "glcm", "--side", "64", "--reps", "1", "--no-timing"],
                       capture_output=True, text=True)
    row = kernels.parse(p.stdout)[0]
    assert row["time"]["median"] == 0 and row["time"]["max"] == 0  # no clock read


@pytest.mark.skipif(not (BENCH_EXE and __import__("shutil").which("valgrind")),
                    reason="needs PERFSUITE_BENCH_EXE and valgrind")
def test_perfsuite_instruction_counts_repeat_exactly_mechanics():
    from perfsuite import icount

    sel = lambda cid: cid == "kernels/glcm/side64"  # noqa: E731
    build = Path(BENCH_EXE).parent
    a = icount.count(BENCH_EXE, build, sides=(64,), select=sel, log=lambda *_: None)
    b = icount.count(BENCH_EXE, build, sides=(64,), select=sel, log=lambda *_: None)
    ca, cb = a["counts"]["kernels/glcm/side64"], b["counts"]["kernels/glcm/side64"]
    assert ca["instructions"] > 0 and ca == cb  # own and library instructions, exactly


@pytest.mark.skipif(not CLI_EXE, reason="set PERFSUITE_CLI to a nyxus CLI built with -DCHECKTIMING=ON")
def test_perfsuite_profile_with_a_real_checktiming_cli_mechanics(tmp_path):
    from perfsuite import checktiming

    name = "2d-256x256-grid16-uint16-s0"
    datasets.ensure(tmp_path / "data", Dataset.parse(name), ["tiff-tiled"], log=lambda *_: None)
    checktiming.require_checktiming_cli(CLI_EXE)
    rows = checktiming.profile(CLI_EXE, [name], tmp_path / "data", repeats=1, threads=1, log_dir=tmp_path,
                               log=lambda *_: None)
    assert all(r["status"] == "ok" for r in rows), [r.get("error") for r in rows]
    groups = {r["id"].split("/", 3)[3] for r in rows}
    assert "WALL" in groups and "Texture/GLCM" in groups and groups - {"WALL"} <= set(checktiming.run_groups("cpu"))
