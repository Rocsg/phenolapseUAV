# Changelog

All notable changes to phenolapseUAV are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.2] - 2026-08-29

### Added

-Fix test under windows

## [1.0.1] - 2026-08-29

### Added

-Fix test under windows

## [1.0.0] - 2026-08-28

### Added

- Complete shapefile ROI workflow with CRS reprojection, polygon overlays,
	labels, selectable unique region identifiers, and per-resolution analysis.
- UTF-8 TSV statistics per region, scene, sensor, and spectral band, including
	population standard deviation, median, and raw median absolute deviation.
- Georeferenced scientific ROI patches, polygon masks, and MS pseudo-RGB
	products using one shared FIXED-grid window per region.
- ImageJ/Fiji spatial calibration on registered bands, multiband images,
	samples, ROI patches, masks, and pseudo-RGB outputs.
- Resolution-specific completion manifests for simultaneous FULL, ×2, and ×4
	exports.
- Continuous persistence for standalone two-image landmark sessions.

### Changed

- Registration exports are parallel, cancellable, atomically replaced, and
	accompanied by live band-level progress and ETA reporting.
- Exports can crop to useful data and subsample source crops before warping to
	reduce memory use; the default is ×4 with four workers.
- TIFF bands use compressed strips and ImageJ display ranges for reliable
	native opening in Fiji.
- Series graph selection, launch workflow, landmark interaction, and output
	directory organization were redesigned for clearer operation.
- ROI processing reports live region, band, mask, pseudo-RGB, and TSV progress.

### Fixed

- Preserved FIXED-grid CRS, affine transform, pixel size, nodata, datatype, and
	band descriptions throughout transformed and cropped outputs.
- Prevented stale exports and duplicate ROI identifiers from silently reusing
	or overwriting results.
- Restored geographic-grid preview projection and robust navigation in ROI
	display mode.

## [0.1.0] - 2026-08-28

### Added

- npe2 napari plugin for interactive UAV orthomosaic registration.
- Georeference-first residual similarity, affine, and TPS model.
- Directed registration graph for MS/IR image series.
- Per-band full-resolution GeoTIFF export.
- 512 × 512 multi-image validation stack export.
- Automatic pairwise and series registration command-line tools.

[Unreleased]: https://github.com/Rocsg/phenolapseUAV/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Rocsg/phenolapseUAV/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/Rocsg/phenolapseUAV/releases/tag/v0.1.0
