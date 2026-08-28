# phenolapseUAV

[![Tests](https://github.com/Rocsg/phenolapseUAV/actions/workflows/tests.yml/badge.svg)](https://github.com/Rocsg/phenolapseUAV/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/phenolapseUAV.svg)](https://pypi.org/project/phenolapseUAV/)
[![Python](https://img.shields.io/pypi/pyversions/phenolapseUAV.svg)](https://pypi.org/project/phenolapseUAV/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**phenolapseUAV** is an open-source napari plugin for interactive and automatic
registration of large georeferenced multispectral and infrared UAV
orthomosaics. It is designed for temporal phenotyping series in which image
resolution, pixel size, contrast, and reconstruction quality vary between
acquisitions or sensors.

## Installation for napari users

After the first PyPI release, no terminal is required:

1. Install and launch the official napari application.
2. Open **Plugins > Install/Uninstall Plugins**.
3. Search for **phenolapseUAV**.
4. Select **Install**.
5. Open **Plugins > phenolapseUAV Registration**.

Alternatively, install into an existing napari Python environment:

```bash
python -m pip install phenolapseUAV
```

Automatic command-line registration is optional:

```bash
python -m pip install "phenolapseUAV[automatic]"
```

AROSICS and GDAL availability depends on the operating system. The interactive
napari plugin does not require AROSICS.

## Quick start

Open **Plugins > phenolapseUAV Registration**, then either select one FIXED and
one MOVING GeoTIFF or open a directory following the
`<scene>_<MS|IR>.tif` naming convention. All controls and full-resolution
exports are available from the dock widget.

## Pairwise automatic registration

```bash
phenolapseUAV-register-pair --ref reference.tif --mov moving.tif --outdir output
```

### Options

| Option | Description |
|---|---|
| `--ref PATH` | Reference image |
| `--mov PATH` | Moving image |
| `--outdir PATH` | Output directory |
| `--band INT` | 1-based band index; defaults to a robust 2nd–98th percentile mean |
| `--method {arosics,fallback,auto}` | Registration method; `auto` tries AROSICS, then the fallback |
| `--debug` | Enable verbose logging |
| `--overwrite` | Allow existing outputs to be overwritten |

### Outputs

- `<moving>_registered.tif`: registered GeoTIFF with all spectral bands preserved
- `registration_report.txt`: detailed text report
- `registration_vectors.geojson`: displacement vectors when available
- `quicklook_before.png` and `quicklook_after.png`
- `difference_before.png` and `difference_after.png`

### Methods

1. **AROSICS**: global `COREG`, followed by `COREG_LOCAL` when successful.
2. **Fallback**: ORB keypoint extraction and matching, RANSAC estimation of a conservative translation or Euclidean model, then band-by-band reprojection with rasterio.

Registration is intentionally conservative to preserve the geometry of biological field plots; no unconstrained deformation is applied.

## Manual landmark registration in napari

The phenolapseUAV napari widget registers two large GeoTIFF files from explicit landmark pairs. During editing, it loads only the displayed band and a downsampled preview (factor 8 by default).

```bash
phenolapseUAV \
  --fixed /path/to/reference_MS.tif \
  --moving /path/to/moving_MS.tif \
  --preview-factor 8
```

Open a complete series and its registration graph directly with:

```bash
phenolapseUAV \
  --dir /path/to/Input/STR \
  --preview-factor 6
```

`--dir` and the `--fixed`/`--moving` pair are mutually exclusive.

When opened from napari, the plugin first presents two workflows: alignment of two images or orthoimages, and alignment of a time-lapse series. Selecting a workflow dismisses the centered welcome message and reveals only the corresponding file controls. In standalone two-image mode, opening FIXED and MOVING immediately creates a deterministic JSON state file under the sibling `<moving-folder>_REGISTERED/STATE` directory. Selecting the same ordered pair later restores its landmarks and settings automatically; every subsequent edit is saved atomically.

### Landmark input

Input works anywhere on the canvas, independently of the displayed view or active layer:

- **FIXED is red** and uses the left mouse button.
- **MOVING is green** and uses the right mouse button.
- The recommended order is FIXED first, then MOVING.
- Landmark clicks never switch views automatically; the buttons and `F`, `M`, and `O` shortcuts remain under user control.
- A middle click on a FIXED or MOVING landmark immediately deletes its entire pair and table row.
- A click without movement adds a landmark; a drag retains normal canvas navigation.
- Holding **Space** completely disables left- and right-click landmark input so those buttons can pan the napari view.
- Hovering over a landmark displays **Click point XX to delete the pair**. A left click on that point removes the complete pair, refreshes the table, and recomputes the preview.
- **Delete latest pair**, `Delete`, or `Shift+Delete` always removes the newest pair, regardless of the current view, active layer, or selection.
- In the opposite view or overlay, clicks are mapped back through the current transformation, or through GeoTIFF georeferencing before the first registration.
- Right-button dragging never zooms; use the mouse wheel for zooming.

### Shortcuts

- `N`: explicitly reserve the next pair
- `F`: FIXED-only view
- `M`: MOVING-only view
- `O`: additive red FIXED / green registered MOVING overlay

The application starts in the red/green overlay view.

Every landmark has a stable visible ID. FIXED markers are slightly red-tinted white and MOVING markers are slightly green-tinted white. Both are semi-transparent and include a fully saturated red or green center. Their initial diameter is 3 and can be adjusted with **Landmark size** for precise placement.

Deleting a landmark removes its complete pair without silently reassigning later IDs. Moving, deleting, or disabling a complete pair starts an asynchronous preview update. An isolated FIXED click does not trigger registration: computation starts only after a complete pair is validated. A large **UPDATING VIEW…** message appears at the center of the canvas during computation. Landmark placement and movement are locked during this update, while view navigation remains available. Stale worker results are ignored. The point table automatically scrolls to the latest coordinates.

Stored MOVING coordinates always refer to the raw image. In the overlay, green MOVING pixels have already been warped onto the FIXED grid; right clicks are therefore inverse-mapped to the original MOVING coordinates before updating the model. Once a registration is available, the MOVING-only view also displays the registered image on the FIXED grid. Clicks and moved landmarks in that view are mapped back to raw MOVING coordinates.

Before the first pair, the MOVING preview is reprojected onto the geographic FIXED grid using GeoTIFF georeferencing and pixel sizes. Lower-resolution IR images therefore retain their correct geographic extent instead of appearing artificially smaller. The green-only view and red/green overlay use the exact same registered MOVING array and FIXED grid, both before and after landmark fitting; switching views cannot change image placement.

Because scikit-image does not provide an analytical TPS inverse, the application explicitly estimates two TPS models: FIXED→MOVING for resampling and MOVING→FIXED for interactive landmark display.

### Registration-series graph

The **Manual registration series** panel below the landmark table manages multiple registrations in one session:

1. **Open directory** discovers GeoTIFF files and automatically reloads existing state.
2. Filenames are parsed explicitly as `<scene>_<MS|IR>.tif`, for example `2022_13h_STR_MS.tif`.
3. Scenes are displayed as rows and MS/IR modalities as columns.
4. Left-clicking a node makes it FIXED and gives it a red frame. The graph automatically selects MOVING and immediately loads the pair into the main viewer.
5. The automatic MOVING choice is the next image in alphanumeric filename order within the same modality. If there is none, it uses the same scene in the next available modality.
6. Right-clicking another node overrides MOVING while preserving the current FIXED image. The selected MOVING node receives a green frame.
7. Switching edges stores the previous edge's landmarks in memory and restores the new edge if it was edited earlier.

Potential edges are dark. An edited edge changes color according to its number of complete pairs: 1, 4 (similarity), 7 (residual affine), and 10 (residual TPS). Every landmark edit and edge change automatically saves the complete state. **Save state** forces the same write without asking for a filename.

For an input directory such as `Input/STR`, every generated file is stored in the deterministic sibling directory `Input/STR_REGISTERED`. Registration state is saved under `STR_REGISTERED/STATE/manual_registration_state.json`. The file contains image metadata, directed FIXED→MOVING edges, and full-resolution coordinates. Existing state from the former `OUTPUT` layout is still read for backward compatibility and is migrated to the new location on the next save.

Manual registration always starts with GeoTIFF georeferencing and pixel sizes. Landmark corrections are then fitted successively on the remaining residuals: a fallback translation for 1–3 pairs, similarity from 4 pairs, affine after similarity from 7 pairs, and TPS after similarity and affine from 10 pairs. The same composed mapping is used for preview, interaction, and full-resolution export. Full-resolution export requires 10 pairs and processes spectral bands separately with one shared spatial transformation and no interpolation between channels.

**Apply / Export TPS** first displays the complete image list and asks which image must define the series reference geometry, with the current FIXED image preselected. It then writes automatically to the sibling `<input>_REGISTERED` directory. In series mode it follows complete directed registration edges from the selected reference and exports every image onto that grid. Export is refused if any series image is unreachable or if a required edge has fewer than 10 complete enabled pairs.

The **Export resolution** selector offers full resolution, ×2, and ×4 outputs, with ×4 selected by default. Subsampled source crops are created in memory before registration, reducing both the number of output pixels and the memory held by each active warped band. Results are kept separately under `REGISTERED_IMAGES/FULL`, `REGISTERED_IMAGES/SUBSAMPLE_2`, or `REGISTERED_IMAGES/SUBSAMPLE_4`.

**Crop export to useful data** is enabled by default. A pixel is considered useful when at least one band is valid according to the raster mask/nodata metadata and differs from the minimum and maximum value of its datatype. The FIXED useful-data bounding box receives a 5-pixel margin and defines the shared output grid. For every MOVING image, a dense sampling of this FIXED region is transformed into source coordinates and intersected with that image's own padded useful-data bounds. Thus a source crop never extends beyond its detected data. Pixels on the FIXED output grid that have no source coverage receive the source nodata value, or zero when nodata is undefined. The exported affine transform includes both the FIXED crop offset and the selected subsampling factor.

The **Workers** selector controls how many independent source bands are warped concurrently; it defaults to at most 4 or the available logical CPU count. A useful starting point is one worker per simultaneously exported band or modality, provided enough RAM is available. Memory use grows approximately with the worker count because every active warped band loads its downsampled source crop. Export progress reports completed and active bands separately, distinguishes fast reference copies from actual warps, and estimates the remaining warp time only after a warped band has completed. Multiband assembly is shown as a separate phase. **Stop export** requests cooperative cancellation: active workers finish their current raster block, temporary files are removed, and the worker count can then be changed before restarting. Existing completed outputs are not replaced until every new band and multiband product is ready.

Each file in the selected resolution's `BANDS/` directory is an ordinary two-dimensional, one-band GeoTIFF containing exactly one TIFF IFD/page (`C=1`, `Z=1`, `T=1`). Outputs are deliberately written as compressed TIFF strips rather than tiles so Fiji can use its native TIFF opener; a tiled GeoTIFF would normally be delegated to Bio-Formats, which displays stack/hyperstack import choices and ignores ImageJ display ranges. These files are computed independently and in parallel. Only after all bands succeed are they assembled serially into one all-bands GeoTIFF per source image in the adjacent `MULTIBAND/` directory. Completion metadata, resolution, and crop bounds are stored in `STATE/phenolapseuav_export.json`.

Every TIFF produced by the manual workflow also contains an official ImageJ `ImageDescription` plus private `IJMetadata` display-range hints. The description explicitly declares one non-hyperstack image and duplicates the first-band display minimum and maximum for compatibility. A robust lower/upper range is estimated independently for each source band with the selected contrast percentiles, excluding zero, integer saturation, NaN, and infinity. These values only tell Fiji how to initialize **Brightness/Contrast**: exported scientific pixel values, datatype, georeferencing, and nodata are not clipped, normalized, or otherwise changed. Multiband files store one lower/upper pair per channel, and the sample stack propagates the range of each contributing first band. If Bio-Formats is explicitly selected, its importer can ignore ImageJ-private display hints; Fiji's native TIFF opener reads the ImageJ metadata directly.

All TIFF products from the manual workflow also carry the ImageJ spatial
calibration of the shared FIXED output grid. Every registered image is first
written with the FIXED CRS and affine transform; the same resulting pixel size
is therefore propagated to every sensor, date, band, ROI patch, mask, and
pseudo-RGB product. Projected CRS units are converted to centimetres and written
as TIFF `XResolution`, `YResolution`, and `ResolutionUnit=cm` tags together with
`unit=cm` in the ImageJ description. Thus a 0.005 m FIXED pixel is opened as
0.5 cm per pixel. Cropping changes only the origin, while ×2/×4 subsampling
multiplies the common FIXED pixel size by 2 or 4. Multiband files and sample
results preserve the same calibration without changing GeoTIFF georeferencing.

To continue an existing test, open the original input directory again, not its output directory. The plugin automatically finds the sibling `<input>_REGISTERED/STATE/manual_registration_state.json` and restores the graph and landmarks. To restart only the full-resolution export, keep `STATE/manual_registration_state.json` and launch **Apply / Export TPS** again; completed files are replaced after the new export succeeds. To simulate a completely new series, rename or remove the sibling `<input>_REGISTERED` directory. Exports created with older or different control points are detected as obsolete and are never reused for sample creation. Cancelling a rerun preserves a previous export only when its registration fingerprint still matches the current points; otherwise every output is recomputed on the next run.

After a successful export, **Create 512×512 sample** becomes available. The button is also restored automatically in a later session when the completion manifest and all declared files are present. Clicking it displays a preview of the selected series reference and asks for a crop center. The next left click creates `PATCHES/SAMPLE_RESULT.tif`. This file contains one 512×512 band per exported image, using the first source band, and is immediately opened as a stack in napari for visual inspection across the complete series and all modalities. Crops touching an image boundary are padded with nodata.

### Shapefile ROI analysis

After at least one complete export, **Shapefile ROI analysis** lists every
available full, ×2, and ×4 result whose files still exist and whose registration
fingerprint matches the current landmarks. Completion manifests are retained per
resolution under `STATE/EXPORTS`, while the latest-export manifest remains
available for backward compatibility.

Select a resolution and load a polygon shapefile. The shapefile must declare a
CRS; its polygons are reprojected to the selected registered raster. Choose the
attributes used as region ID and region name. The ID selector lists only fields
having one non-empty, distinct value per polygon; that value directly names the
polygon's patch folder. Duplicate identifiers are refused before any output can
be overwritten. **Display polygons** then shows only the exported reference's
first band, yellow polygon outlines, and compact labels placed at representative
interior points. Polygon holes and multipolygon parts are preserved.

**Export ROI statistics** writes the UTF-8 tab-separated file
`ROI_ANALYSIS/DATA_FOR_SHAPEFILE_<shapefile>/<resolution>/roi_statistics.tsv`,
with one row per region, source image, and band. A pixel is included when its
center lies inside the polygon and the raster mask marks it valid. Explicit
nodata, NaN, and infinity are excluded; zero remains a valid scientific value.
The table contains the valid count, minimum, maximum, mean, population standard
deviation (`ddof=0`), median, and raw median absolute deviation without
normalization.

**Export ROI patches + masks** uses one shared polygon bounding window for all
registered sources, enlarged by the configurable margin (5 pixels by default).
Each region receives its own `PATCHES/<region_id>` directory containing:

- one unchanged scientific 2-D, one-page, stripped GeoTIFF per source band,
  retaining dtype, CRS, affine transform, nodata, and ImageJ display range;
- one `uint8` mask whose values are 255 inside and 0 outside the polygon;
- for MS sources with at least three bands, one masked 8-bit pseudo-RGB image
  with `R=B03`, `G=B02`, and `B=B01`, scaled independently from the registered
  bands' ImageJ display ranges.

During statistics or patch generation, live progress is shown both in the top
status area of the right-hand panel and over the image canvas. It reports the
current region, source image and band, masks and pseudo-RGB products, TSV writing,
completed steps, percentage, and elapsed time.

Contrast limits are estimated from raw source-pixel samples. Non-finite values, zero, and exact saturation are excluded. The default 0.5th and 99.5th percentiles can be changed in the panel. Landmarks and session metadata can be saved and reloaded as JSON.

### Future synchronized three-view layout

napari currently has no stable public API for inserting three independent canvases into one `Viewer`. Modifying private Qt/Vispy viewer objects would be fragile. A safe future implementation would use three public `Viewer` instances (FIXED, MOVING, and OVERLAY), share read-only preview arrays, and synchronize their cameras through public `camera.events` with a recursion guard. Qt can place the windows side by side without duplicating full-resolution rasters. The current prototype intentionally retains one canvas with instant view switching until this feature can be isolated and tested.

## Series registration

The optional `phenolapseUAV-register-series` command registers an entire orthomosaic series, manages matching multispectral (MS) and infrared (IR) acquisitions, and exports every image in the geometry and native resolution of the reference MS image. IR images are therefore upsampled onto the reference MS grid.

Expected naming is `<prefix>_MS.tif` and `<prefix>_IR.tif`, for example `2022_13h_STR_MS.tif` and `2022_13h_STR_IR.tif`. Each shared prefix identifies one scene. MS images are matched to MS images, and IR images to IR images.

```bash
phenolapseUAV-register-series --indir /path/to/input --outdir /path/to/output
```

### Options

| Option | Description |
|---|---|
| `--indir PATH` | Directory containing the GeoTIFF series |
| `--outdir PATH` | Output directory |
| `--reference NAME.tif` | Reference **MS** filename; when omitted, select it from an interactive downsampled montage |
| `--band INT` | 1-based MS matching band; when omitted, select a channel interactively or use the robust mean |
| `--subsample {1,2,3,4}` | Downsampling for a single pass; ignored with `--pyramid` |
| `--pyramid 4,2,1` | Coarse-to-fine registration with composed corrections and one final native-resolution resampling |
| `--warp {rigid,affine,poly2,tps}` | Warp model; defaults to a smooth, lightweight TPS with automatic fallback when landmarks are insufficient |
| `--resample {cubic,linear,nearest}` | Resampling interpolation |
| `--limit N` | Process only the first N moving scenes; 0 processes all scenes |
| `--debug` | Enable verbose logging |

Existing files with the same output names are overwritten automatically.

### Optional `Reference_point` file

When `Reference_point` or `Reference_point.txt` exists in the input directory, each `image_name X Y` line specifies a full-resolution support point in column/row coordinates. The point is added as an anchor to the MS tie points before automatic optimization.

### MS/IR registration topology

1. Register every **MS** image to the reference **MS** image.
2. Register every **IR** image to the reference scene's **IR** image.
3. Place the reference **IR** image on the reference **MS** image using georeferencing only; no cross-modality IR↔MS content matching is performed.
4. Compose each IR transformation as `IR → IR_ref → MS_ref`.

Homogeneous **MS↔MS** and **IR↔IR** matching uses SIFT and RANSAC. With `--pyramid 4,2,1`, level 4 estimates a similarity model, level 2 estimates a lightweight affine model, and level 1 estimates the requested final model (TPS by default). Refinements with overly concentrated tie points are rejected. Transformations are composed, and each original GeoTIFF is resampled only once at full resolution onto the reference MS grid.

### Outputs

- All rasters are written to the `BANDS/` subdirectory.
- Each MS image produces one single-band file per channel: `<name>_B01_registered.tif`, `<name>_B02_registered.tif`, and so on.
- Each IR image produces `<name>_registered.tif`.
- Reference MS bands are extracted without resampling.
- A safety guard rejects transformations covering less than 1% of the reference grid.
- `series_report.txt` records the model, tie-point count, and processing time for each image.

## Development

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/Rocsg/phenolapseUAV.git
cd phenolapseUAV
python -m pip install -e ".[dev]"
python -m pytest
```

Use `.[automatic,dev]` to include the optional automatic-registration stack.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules.

## Release process

1. Update the version in `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md`.
2. Run the test suite and `python -m build`.
3. Validate artifacts with `python -m twine check dist/*`.
4. Push a version tag and create a GitHub release.
5. The release workflow publishes the wheel and source distribution to PyPI
  using trusted publishing.
6. Connect the GitHub repository to Zenodo before the first public release to
  obtain both release and concept DOIs.

The PyPI project owner must configure a trusted publisher for repository
`Rocsg/phenolapseUAV`, workflow `publish.yml`, environment `pypi`.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). A JOSS manuscript
is available under [paper/paper.md](paper/paper.md). Once a DOI or paper citation
exists, update both files before the corresponding release.

## License

phenolapseUAV is distributed under the GNU General Public License version 3.0
only. See [LICENSE](LICENSE).
