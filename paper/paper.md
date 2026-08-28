---
title: 'phenolapseUAV: interactive registration of multispectral and infrared UAV orthomosaic time series'
tags:
  - Python
  - napari
  - image registration
  - UAV
  - multispectral imaging
  - phenotyping
authors:
  - name: Romain Fernandez
    affiliation: 1
  - name: Serigne Mansour Diene
    affiliation: 1
  - name: Alain Audebert
    affiliation: 1
affiliations:
  - name: CIRAD, Montpellier, France
    index: 1
date: 28 August 2026
bibliography: paper.bib
---

# Summary

UAV image series acquired across dates or sensors require spatial alignment
before plot-level temporal and multispectral analyses. `phenolapseUAV` is an
open-source napari plugin for interactive landmark registration of large,
georeferenced multispectral and infrared orthomosaics. It combines GeoTIFF
georeferencing and pixel sizes with successive residual similarity, affine,
and thin-plate spline corrections. The same composed mapping is used for
interactive previews and band-wise full-resolution export.

The plugin provides explicit FIXED and MOVING landmark pairs, asynchronous
registered previews, additive red/green quality control, stable point
identifiers, persistent directed registration graphs, and a multi-image crop
stack for checking an exported series. It processes spectral bands
independently and places all exported images on a user-selected reference
geometry. Command-line workflows are also provided for automatic pairwise and
series registration.

# Statement of need

Registration errors can confound temporal phenotyping measurements, while
cross-modality MS/IR matching is difficult when sensor resolution, contrast,
and reconstruction quality differ. General-purpose registration packages may
hide correspondences or require programming expertise. `phenolapseUAV` exposes
landmarks and intermediate visual feedback directly in napari [@napari], making
manual correction accessible while preserving geospatial metadata and
multispectral channels. Its directed graph supports chained registrations when
direct matching to one reference is unreliable.

# Method and implementation

Source previews are downsampled so editing does not repeatedly resample full
orthomosaics. Before landmark fitting, MOVING data are reprojected using CRS,
affine transforms, and pixel size. Landmark residuals successively estimate a
similarity transform, an affine transform, and a local thin-plate spline using
scikit-image [@skimage]. Full-resolution GeoTIFFs are read and written with
Rasterio [@rasterio], one spectral band at a time. napari [@napari] provides the
viewer, layers, navigation, and plugin integration.

# Availability

The source code is available at
[https://github.com/Rocsg/phenolapseUAV](https://github.com/Rocsg/phenolapseUAV)
and is distributed under GPL-3.0-only. Releases are intended for PyPI and the
napari plugin manager. Archived releases will be deposited through Zenodo.

# Acknowledgements

The authors thank the napari, scikit-image, and Rasterio communities.

# References
