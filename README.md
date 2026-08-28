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

The phenolapseUAV napari widget registers two large GeoTIFF files from explicit landmark pairs. During editing, it loads only the displayed band and a downsampled preview (factor 4 by default).

```bash
phenolapseUAV \
  --fixed /path/to/reference_MS.tif \
  --moving /path/to/moving_MS.tif \
  --preview-factor 4
```

Open a complete series and its registration graph directly with:

```bash
phenolapseUAV \
  --dir /path/to/Input/STR \
  --preview-factor 6
```

`--dir` and the `--fixed`/`--moving` pair are mutually exclusive.

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
4. The first selected node becomes FIXED and receives a red frame.
5. The second becomes MOVING, receives a green frame, and immediately loads the pair into the main viewer.
6. Switching edges stores the previous edge's landmarks in memory and restores the new edge if it was edited earlier.

Potential edges are dark. An edited edge changes color according to its number of complete pairs: 1, 4 (similarity), 7 (residual affine), and 10 (residual TPS). Every landmark edit and edge change automatically saves the complete state. **Save state** forces the same write without asking for a filename.

For an `Input/STR` directory, state is stored automatically in `Output/STR/manual_registration_state.json`, or in an `OUTPUT` directory created next to `INPUT`. The file contains image metadata, directed FIXED→MOVING edges, and full-resolution coordinates. These directed edges form the graph required to compose transformations along a chain without losing original coordinates.

Manual registration always starts with GeoTIFF georeferencing and pixel sizes. Landmark corrections are then fitted successively on the remaining residuals: a fallback translation for 1–3 pairs, similarity from 4 pairs, affine after similarity from 7 pairs, and TPS after similarity and affine from 10 pairs. The same composed mapping is used for preview, interaction, and full-resolution export. Full-resolution export requires 10 pairs and processes spectral bands separately with one shared spatial transformation and no interpolation between channels.

**Apply / Export TPS** first displays the complete image list and asks which image must define the series reference geometry, with the current FIXED image preselected. It then asks for an output directory. In series mode it follows complete directed registration edges from the selected reference and exports every image onto that grid. Export is refused if any series image is unreachable or if a required edge has fewer than 10 complete enabled pairs. Every source band is saved as an independent single-band, two-dimensional GeoTIFF in `BANDS/`; one-band source images therefore remain 2-D rasters rather than one-item stacks. In standalone mode, the FIXED and MOVING images are exported in the same per-band format.

After a successful export, **Create 512×512 sample** becomes available. Clicking it displays a preview of the selected series reference and asks for a crop center. The next left click creates `SAMPLE_RESULT.tif` in the selected output directory. This file contains one 512×512 band per exported image, using the first source band, and is immediately opened as a stack in napari for visual inspection across the complete series and all modalities. Crops touching an image boundary are padded with nodata.

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
