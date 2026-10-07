# 3D mesh volume vs PyRadiomics — comparison report

`3MESH_VOLUME` and PyRadiomics `MeshVolume` agree bit for bit on every mask tested whose surface
crosses no ambiguous lattice face — a face with two diagonal corners inside and two outside. On masks
with such faces they differ, by 0.42% on a pitted ball and 2.5–5.1% on random noise, because
PyRadiomics' table handles those faces differently from Nyxus' and in places leaves its surface open.
Mirroring such a mask changes PyRadiomics' volume by up to 8.5% and Nyxus' by up to 0.55%. PyRadiomics
is therefore a usable third oracle for `3MESH_VOLUME`, after MIRP and the analytic solids, only for
ROIs whose surface has no ambiguous face, at unit spacing. No assertion in the tree pins a PyRadiomics
value.

## Tool and configuration

| | |
|---|---|
| Tool | PyRadiomics 3.0.1 (SimpleITK 2.3.1, NumPy 1.23.5, Python 3.8.20) |
| PyRadiomics call | `radiomics.cShape.calculate_coefficients`, the C routine `shape.py` calls, on a `[z, y, x]` mask padded by one voxel, unit spacing — the array the extractor builds after cropping the mask to its bounding box |
| Nyxus side | `MC_TRIANGLES` read from `src/nyx/features/3d_mesh.cpp`, walked as `march_roi_surface()` walks it and integrated from the first vertex as `roi_mesh_volume()` does |
| Script | `tests/vetting/audit/compare_mesh_volume_pyradiomics.py` |

```
python tests/vetting/audit/compare_mesh_volume_pyradiomics.py
```

The script reads the shipped table, so it always checks the table in the tree, but it does not run
the Nyxus binary. It runs PyRadiomics at unit spacing, so both sides report volume in voxel units.
PyRadiomics (BSD-3-Clause) is a reference only, never a CI runtime dependency (SPEC §4).

## What both implementations share

- Marching cubes over the binary mask at the 0.5 isolevel, with one voxel of empty padding, so the
  surface also closes where the ROI touches its bounding box.
- Every vertex at the midpoint of a cube edge; a binary mask needs no interpolation.
- Volume by the divergence theorem: the signed volumes of the tetrahedra each triangle spans with a
  reference point, summed and divided by 6.

PyRadiomics also integrates the mesh area, as half the norm of each triangle's cross product. Nyxus
computes no mesh area: `3AREA` counts exposed voxel faces. The script computes the same mesh area on
the Nyxus triangles only to compare the two meshes.

## Where they differ

| aspect | PyRadiomics 3.0.1 | Nyxus | consequence |
|---|---|---|---|
| case table | classic 128-entry table; a cube whose corner 7 (in `cshape.c`'s numbering) is inside is served by complementing its mask and flipping the volume's sign | all 256 masks, derived by `derive_marching_cubes_table.py`; an ambiguous face always separates its inside corners | a complemented PyRadiomics cube joins the inside corners of an ambiguous face, where Nyxus separates them; on faces without ambiguity the contour loops are the same, though the diagonals that triangulate them may differ |
| volume reference point | the corner of the padded array, i.e. one voxel outside the ROI's bounding box | the first vertex of the surface | over a closed surface the choice does not matter; over PyRadiomics' open one it does (see Result) |
| voxel spacing | each vertex scaled by the physical spacing; volume in physical units (mm³) | never physical units: anisotropic data is resampled by the `--aniso*` factors as given or, with physical spacing on, by the spacing ratios with the smallest axis 1, and the volume is in units of the resampled voxel | equal only at unit spacing; at an isotropic spacing *s*, PyRadiomics' volume is *s*³ times Nyxus' |
| convex-hull volume | no 3D hull feature | `3VOLUME_CONVEXHULL`, the hull of the mesh vertices | no PyRadiomics counterpart |
| surface area and its ratios | `SurfaceArea` is the mesh area; `Sphericity`, `SurfaceVolumeRatio` and the deprecated `Compactness1`, `Compactness2`, `SphericalDisproportion` use mesh area and mesh volume | `3AREA` counts exposed voxel faces; `3AREA_2_VOLUME`, `3COMPACTNESS1/2`, `3SPHERICAL_DISPROPORTION`, `3SPHERICITY` use `3AREA` and `3VOXEL_VOLUME` | different definitions, not comparable |

## Result

Volumes (V) and areas (A) in voxel units. "Mirrored" is the range of the volume over the mask
mirrored along each of its three axes in turn; a single value means all three agree with each other.
"Moved" calls `cShape` directly on the mask placed 7 voxels further from the array's corner along
each axis — something PyRadiomics' extractor never does, since it crops first — to expose its open
surface. The mesh area is listed for both sides because the script computes it, but the Nyxus
`3AREA` feature is the face count, not this mesh area.

| mask | voxels | Nyxus V | Nyxus V mirrored | PyRadiomics V | PyRadiomics V mirrored | PyRadiomics V moved | Nyxus mesh A | PyRadiomics A | Nyxus mesh closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| single voxel | 1 | 0.166667 | 0.166667 | 0.166667 | 0.166667 | 0.166667 | 1.732051 | 1.732051 | yes |
| box 3×4×5 | 60 | 54.666667 | 54.666667 | 54.666667 | 54.666667 | 54.666667 | 79.187895 | 79.187895 | yes |
| ball r=6 | 925 | 911.500000 | 911.500000 | 911.500000 | 911.500000 | 911.500000 | 499.688711 | 499.688711 | yes |
| ellipsoid, semi-axes 8/5/3 | 477 | 465.166667 | 465.166667 | 465.166667 | 465.166667 | 465.166667 | 357.821622 | 357.821622 | yes |
| two voxels, edge contact | 2 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | 3.464102 | 3.464102 | yes |
| two voxels, corner contact | 2 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | 3.464102 | 3.464102 | yes |
| pitted ball r=10 | 4018 | 3994.625000 | 3994.375000–3995.666667 | 3977.708333 | 3961.708333–3967.958333 | 3984.708333 | 1535.250855 | 1514.763251 | yes |
| random 30% fill, 8³ | 164 | 92.208333 | 91.750000–92.166667 | 96.833333 | 93.250000–102.500000 | 78.166667 | 385.967632 | 419.125472 | yes |
| random 50% fill, 8³ | 249 | 198.208333 | 198.000000–199.291667 | 203.166667 | 208.166667–220.416667 | 185.666667 | 562.485921 | 570.553568 | yes |
| random 70% fill, 8³ | 361 | 346.791667 | 347.041667–348.083333 | 364.416667 | 355.416667–358.833333 | 376.083333 | 594.229507 | 562.946064 | yes |

The pitted ball is the r=10 digital ball (4169 voxels) with each voxel of its outer one-voxel shell
removed with probability 0.15, which removed 151.

**Masks without ambiguous faces agree.** The four solid shapes cross no ambiguous face. Their volumes
agree bit for bit, in place and mirrored, and their areas to 1e-14 relative. The two-voxel masks
agree too. The edge-contact pair crosses one ambiguous face, and PyRadiomics serves both cubes that
share it without complementing, so it separates the corners just as Nyxus does. The corner-contact
pair touches only across a cube's body diagonal and has no ambiguous face.

**Masks with ambiguous faces differ.** On the pitted ball PyRadiomics sits 0.42% below Nyxus. On the
noise masks it sits 2.5–5.1% above. Two things contribute:

- Where both cubes sharing an ambiguous face are complemented, PyRadiomics joins the inside corners
  and Nyxus separates them. Both surfaces are closed there; they just enclose different volumes.
- Where one of the two cubes is complemented and the other is not, they cut the face differently and
  PyRadiomics' surface has a hole.

The script does not separate the two contributions.

**PyRadiomics' open surface makes its volume depend on orientation.** With holes, the volume depends
on the integration reference point. In normal use that point is fixed relative to the ROI, so
moving the ROI within the image does not change the reported value. The "moved" column shows the
effect directly: placed 7 voxels further out, the pitted ball's volume changes by 7.0 voxel³ and the
noise masks by up to 15%. Mirroring the mask, though, changes the value through the extractor as
well: by up to 0.40% on the pitted ball and up to 8.5% on the noise masks.

**Nyxus' volume changes much less under mirroring, but it does change.** It is closed on every mask,
so the reference point does not matter, and translating the ROI cannot change it. Mirroring changes
the triangles, though, and on the pitted ball and the noise masks the volume moves by up to 0.55%. On
the solid shapes it does not move. The cause has not been traced further.

**How PyRadiomics' openness was established.** This was checked once outside the script, which cannot
do it: the pip package ships `cShape` compiled, without the C source that holds its tables.
Rebuilding PyRadiomics' triangles from the `gridAngles`, `triTable` and `vertList` tables in 3.0.1's
`radiomics/src/cshape.c` reproduces `cShape`'s volume exactly on all ten masks.

- The rebuilt mesh is closed on the four solid shapes and the two two-voxel masks.
- On the pitted ball and the 30%, 50% and 70% noise masks, 184, 184, 172 and 240 directed edges have
  no reverse: 4 for each of the 46, 46, 43 and 60 ambiguous faces shared by a complemented cube and a
  plain one.
- A further 17, 21, 26 and 24 ambiguous faces are shared by two complemented cubes. These are the
  closed but different ones above.

"Closed" in the table means every directed edge of the Nyxus mesh is traversed as often in reverse.
Where two sheets of the surface touch along a lattice edge, four triangles share that edge, two in
each direction. In the Nyxus mesh this happens on 2 edges of the 50% noise mask and 12 of the 70% one,
and leaves the surface closed; every other edge of every Nyxus mesh is shared by exactly two
triangles.

## Performance

Measured in one session with a standalone harness that links `src/nyx/features/3d_mesh.cpp` and
PyRadiomics 3.0.1's `cshape.c`, both built with MSVC 2022 `/O2`, on one Windows 11 workstation. The
inputs are digital balls. Each number is the best of 3–10 runs, except PyRadiomics as called by
`shape.py` at r=40, which ran once. PyRadiomics' C routine always computes the mesh diameters as
well, so it was timed with and without that step. These numbers are not in a test and will drift
with hardware and compiler.

| ball | voxels | Nyxus volume only | Nyxus volume + hull points | PyRadiomics mesh only | PyRadiomics as called by `shape.py` |
|---|---:|---:|---:|---:|---:|
| r=10 | 4,169 | 0.12 ms | 0.64 ms | 0.10 ms | 3.7 ms |
| r=40 | 267,761 | 4.2 ms | 12.4 ms | 3.6 ms | 760 ms |
| r=100 | 4,187,857 | 61 ms | 135 ms | 50 ms | not run |

- The mesh passes are within about 20% of each other, PyRadiomics' being the faster. PyRadiomics
  reads a dense, padded mask; Nyxus starts from the voxel list, keeps two z-planes in memory and
  hands each triangle to a callback. The same walker then serves in-memory and out-of-core ROIs.
- `roi_mesh_volume()` with hull points, the call `3d_surface.cpp` makes, costs 2.2–5.3× the volume
  alone over balls of r=10 to 100. The extra work is collecting the two ends of each lattice row of
  vertices, which `RowEnds` keeps in a `std::map`.
- PyRadiomics' diameter step compares every pair of surface vertices, so its cost grows with the
  square of the surface size. With it, the call takes 37× the mesh pass at r=10 and 214× at r=40.
