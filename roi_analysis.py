"""Shapefile-based statistics and patch extraction on registered GeoTIFFs."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

ROI_STATISTICS_COLUMNS = (
    "region_id",
    "region_name",
    "scene",
    "source_image",
    "sensor",
    "band_number",
    "band_description",
    "valid_pixel_count",
    "minimum",
    "maximum",
    "mean",
    "standard_deviation",
    "median",
    "median_absolute_deviation",
)


@dataclass(frozen=True)
class ROIRegion:
    """One polygonal feature reprojected into the selected raster CRS."""

    region_id: str
    region_name: str
    geometry: Any
    properties: dict[str, Any]


@dataclass(frozen=True)
class ROIAnalysisResult:
    """Paths and counts produced by an ROI analysis run."""

    output_directory: Path
    statistics_path: Optional[Path]
    region_count: int
    patch_count: int
    mask_count: int
    rgb_count: int


def _polygonal_geometry(geometry: Any) -> Any:
    """Return only polygonal components from a valid Shapely geometry."""
    from shapely import make_valid
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    geometry = make_valid(geometry)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for item in geometry.geoms:
            if isinstance(item, Polygon):
                polygons.append(item)
            elif isinstance(item, MultiPolygon):
                polygons.extend(item.geoms)
        if polygons:
            return MultiPolygon(polygons)
    raise ValueError("ROI feature has no usable polygonal geometry")


def shapefile_fields(path: Path) -> list[str]:
    """Return attribute names, requiring a declared shapefile CRS."""
    import fiona

    with fiona.open(path) as source:
        if not source.crs_wkt and not source.crs:
            raise ValueError("The ROI shapefile has no declared CRS")
        return list(source.schema.get("properties", {}))


def unique_shapefile_fields(path: Path) -> list[str]:
    """Return fields containing one non-empty, unique value per feature."""
    import fiona

    with fiona.open(path) as source:
        if not source.crs_wkt and not source.crs:
            raise ValueError("The ROI shapefile has no declared CRS")
        fields = list(source.schema.get("properties", {}))
        values = {field: [] for field in fields}
        feature_count = 0
        for feature in source:
            if feature.get("geometry") is None:
                continue
            feature_count += 1
            properties = feature.get("properties") or {}
            for field in fields:
                value = properties.get(field)
                values[field].append("" if value is None else str(value).strip())
    return [
        field
        for field in fields
        if feature_count > 0
        and len(values[field]) == feature_count
        and all(values[field])
        and len(set(values[field])) == feature_count
    ]


def load_shapefile_regions(
    path: Path,
    target_crs: Any,
    id_field: str,
    name_field: str,
) -> list[ROIRegion]:
    """Read, validate, and reproject shapefile polygons to a raster CRS."""
    import fiona
    from pyproj import CRS, Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as transform_geometry

    if target_crs is None:
        raise ValueError("The selected registered raster has no CRS")
    regions: list[ROIRegion] = []
    seen_ids: set[str] = set()
    seen_directories: set[str] = set()
    with fiona.open(path) as source:
        source_crs_value = source.crs_wkt or source.crs
        if not source_crs_value:
            raise ValueError("The ROI shapefile has no declared CRS")
        fields = set(source.schema.get("properties", {}))
        for field in (id_field, name_field):
            if field not in fields:
                raise ValueError(f"Shapefile attribute not found: {field}")
        transformer = Transformer.from_crs(
            CRS.from_user_input(source_crs_value),
            CRS.from_user_input(target_crs),
            always_xy=True,
        )
        for index, feature in enumerate(source, start=1):
            raw_geometry = feature.get("geometry")
            if raw_geometry is None:
                continue
            properties = dict(feature.get("properties") or {})
            geometry = _polygonal_geometry(
                transform_geometry(transformer.transform, shape(raw_geometry))
            )
            if geometry.is_empty:
                continue
            region_id = str(properties.get(id_field, "")).strip()
            region_name = str(properties.get(name_field, "")).strip()
            if not region_id:
                region_id = str(index)
            if not region_name:
                region_name = region_id
            if region_id in seen_ids:
                raise ValueError(
                    f"Region ID attribute '{id_field}' is not unique: value "
                    f"'{region_id}' occurs more than once. Choose a unique ID "
                    "attribute so every polygon receives its own folder."
                )
            directory_name = safe_component(region_id)
            if directory_name in seen_directories:
                raise ValueError(
                    f"Region IDs in attribute '{id_field}' become identical file "
                    f"names ('{directory_name}') after removing unsafe characters. "
                    "Choose another unique ID attribute."
                )
            seen_ids.add(region_id)
            seen_directories.add(directory_name)
            regions.append(
                ROIRegion(region_id, region_name, geometry, properties)
            )
    if not regions:
        raise ValueError("The shapefile contains no usable polygon features")
    return regions


def safe_component(value: str, fallback: str = "region") -> str:
    """Make an attribute value safe for use as one path component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def geometry_window(dataset: Any, geometry: Any, margin: int = 5) -> Any:
    """Return a clipped integer raster window around a geometry."""
    from rasterio.windows import Window, from_bounds

    if margin < 0:
        raise ValueError("ROI margin must be non-negative")
    raw = from_bounds(*geometry.bounds, transform=dataset.transform)
    col_start = max(0, math.floor(raw.col_off) - margin)
    row_start = max(0, math.floor(raw.row_off) - margin)
    col_stop = min(dataset.width, math.ceil(raw.col_off + raw.width) + margin)
    row_stop = min(dataset.height, math.ceil(raw.row_off + raw.height) + margin)
    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError("ROI does not intersect the selected registered raster")
    return Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    )


def polygon_mask(dataset: Any, geometry: Any, window: Any) -> np.ndarray:
    """Rasterize center-in-polygon membership, preserving holes."""
    from rasterio.features import geometry_mask
    from rasterio.windows import transform as window_transform

    return geometry_mask(
        [geometry.__geo_interface__],
        out_shape=(int(window.height), int(window.width)),
        transform=window_transform(window, dataset.transform),
        all_touched=False,
        invert=True,
    )


def valid_polygon_values(
    values: np.ndarray,
    dataset_mask: np.ndarray,
    inside_polygon: np.ndarray,
    nodata: Optional[float],
) -> np.ndarray:
    """Select finite, raster-valid pixel centers inside a polygon."""
    valid = inside_polygon & (dataset_mask != 0)
    if nodata is not None:
        if np.isnan(nodata):
            valid &= ~np.isnan(values)
        else:
            valid &= values != nodata
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.isfinite(values)
    return values[valid]


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    """Compute population statistics and unscaled median absolute deviation."""
    count = int(values.size)
    if count == 0:
        return {
            "valid_pixel_count": 0,
            "minimum": "",
            "maximum": "",
            "mean": "",
            "standard_deviation": "",
            "median": "",
            "median_absolute_deviation": "",
        }
    numeric = values.astype(np.float64, copy=False)
    median = float(np.median(numeric))
    return {
        "valid_pixel_count": count,
        "minimum": float(np.min(numeric)),
        "maximum": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
        "standard_deviation": float(np.std(numeric, ddof=0)),
        "median": median,
        "median_absolute_deviation": float(np.median(np.abs(numeric - median))),
    }


def _source_identity(source_path: Path) -> tuple[str, str]:
    try:
        from series_registration import parse_series_image

        image = parse_series_image(source_path)
        return image.scene, image.modality
    except ValueError:
        return source_path.stem, "UNKNOWN"


def _band_description(source_path: Path, band_number: int) -> str:
    import rasterio

    try:
        with rasterio.open(source_path) as source:
            description = source.descriptions[band_number - 1]
    except (OSError, IndexError):
        description = None
    return description or f"B{band_number:02d}"


def _display_range(path: Path) -> Optional[tuple[float, float]]:
    from napari_tps_register import _read_imagej_display_ranges

    ranges = _read_imagej_display_ranges(path)
    if ranges is None or len(ranges) < 2:
        return None
    return float(ranges[0]), float(ranges[1])


def _strip_profile(profile: dict[str, Any]) -> dict[str, Any]:
    profile = profile.copy()
    for key in ("blockxsize", "blockysize", "interleave", "photometric"):
        profile.pop(key, None)
    profile.update(driver="GTiff", tiled=False, compress="deflate")
    return profile


def _write_band_patch(source: Any, path: Path, window: Any) -> None:
    import rasterio
    from rasterio.windows import transform as window_transform

    from napari_tps_register import (
        DEFAULT_PERCENTILES,
        _write_imagej_display_ranges,
        robust_contrast_limits,
    )

    data = source.read(1, window=window)
    profile = _strip_profile(source.profile)
    profile.update(
        width=int(window.width),
        height=int(window.height),
        count=1,
        transform=window_transform(window, source.transform),
    )
    temporary = path.with_suffix(path.suffix + ".partial")
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(data, 1)
        destination.update_tags(**source.tags())
        destination.update_tags(1, **source.tags(1))
    display_range = _display_range(Path(source.name))
    if display_range is None:
        display_range = robust_contrast_limits(data, DEFAULT_PERCENTILES)
    _write_imagej_display_ranges(temporary, display_range)
    temporary.replace(path)


def _scale_uint8(values: np.ndarray, display_range: Optional[tuple[float, float]]) -> np.ndarray:
    if display_range is None:
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = float(np.min(finite)), float(np.max(finite))
        else:
            low, high = 0.0, 1.0
    else:
        low, high = display_range
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values.astype(np.float64) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _write_rgb_patch(
    band_paths: Sequence[Path], path: Path, window: Any, inside: np.ndarray
) -> None:
    import rasterio
    from rasterio.windows import transform as window_transform

    from napari_tps_register import _write_imagej_display_ranges

    # Source B03/B02/B01 become output R/G/B.
    channels = []
    for band_path in (band_paths[2], band_paths[1], band_paths[0]):
        with rasterio.open(band_path) as source:
            channel = _scale_uint8(source.read(1, window=window), _display_range(band_path))
        channel[~inside] = 0
        channels.append(channel)
    with rasterio.open(band_paths[0]) as source:
        profile = _strip_profile(source.profile)
        profile.update(
            width=int(window.width),
            height=int(window.height),
            count=3,
            dtype="uint8",
            nodata=None,
            transform=window_transform(window, source.transform),
            photometric="RGB",
        )
    temporary = path.with_suffix(path.suffix + ".partial")
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(np.stack(channels))
        destination.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )
    _write_imagej_display_ranges(temporary, (0.0, 255.0))
    temporary.replace(path)


def _write_mask(reference: Any, path: Path, window: Any, inside: np.ndarray) -> None:
    import rasterio
    from rasterio.windows import transform as window_transform

    from napari_tps_register import _write_imagej_display_ranges

    profile = _strip_profile(reference.profile)
    profile.update(
        width=int(window.width),
        height=int(window.height),
        count=1,
        dtype="uint8",
        nodata=0,
        transform=window_transform(window, reference.transform),
    )
    temporary = path.with_suffix(path.suffix + ".partial")
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(np.where(inside, 255, 0).astype(np.uint8), 1)
    _write_imagej_display_ranges(temporary, (0.0, 255.0))
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=ROI_STATISTICS_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def analyze_exported_rois(
    export_result: dict[str, Any],
    regions: Sequence[ROIRegion],
    shapefile_path: Path,
    margin: int = 5,
    write_statistics: bool = True,
    write_patches: bool = True,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> ROIAnalysisResult:
    """Analyze all registered bands on common per-region reference windows."""
    import rasterio

    reference_key = str(Path(export_result["reference_path"]).resolve())
    outputs = export_result["outputs"]
    reference_bands = [Path(path) for path in outputs.get(reference_key, [])]
    if not reference_bands:
        raise ValueError("The selected export has no registered reference band")
    output_directory = (
        Path(export_result["output_directory"])
        / "ROI_ANALYSIS"
        / f"DATA_FOR_SHAPEFILE_{safe_component(shapefile_path.stem, 'shapefile')}"
        / str(export_result["resolution"])
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    patch_count = mask_count = rgb_count = 0
    band_total = sum(len(paths) for paths in outputs.values())
    rgb_source_total = sum(
        1
        for source_name, paths in outputs.items()
        if _source_identity(Path(source_name))[1] == "MS" and len(paths) >= 3
    )
    work_per_region = band_total
    if write_patches:
        work_per_region += 1 + rgb_source_total
    total_work = max(
        1,
        len(regions) * work_per_region + (1 if write_statistics else 0),
    )
    completed_work = 0

    def report(
        phase: str,
        region_index: int = 0,
        region_id: str = "",
        source_image: str = "",
        band_number: int = 0,
        band_count: int = 0,
    ) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": phase,
                    "completed": completed_work,
                    "total": total_work,
                    "region_index": region_index,
                    "region_count": len(regions),
                    "region_id": region_id,
                    "source_image": source_image,
                    "band_number": band_number,
                    "band_count": band_count,
                }
            )

    report("preparing")

    with rasterio.open(reference_bands[0]) as reference:
        if reference.crs is None:
            raise ValueError("The selected registered raster has no CRS")
        for region_index, region in enumerate(regions, start=1):
            window = geometry_window(reference, region.geometry, margin)
            inside = polygon_mask(reference, region.geometry, window)
            region_directory = (
                output_directory / "PATCHES" / safe_component(region.region_id)
            )
            if write_patches:
                region_directory.mkdir(parents=True, exist_ok=True)
                _write_mask(reference, region_directory / "roi_mask.tif", window, inside)
                mask_count += 1
                completed_work += 1
                report("mask", region_index, region.region_id)

            for source_name, exported_names in outputs.items():
                source_path = Path(source_name)
                band_paths = [Path(name) for name in exported_names]
                scene, sensor = _source_identity(source_path)
                for band_number, band_path in enumerate(band_paths, start=1):
                    with rasterio.open(band_path) as band:
                        values = band.read(1, window=window)
                        if write_statistics:
                            raster_mask = band.read_masks(1, window=window)
                            selected = valid_polygon_values(
                                values, raster_mask, inside, band.nodata
                            )
                        if write_patches:
                            patch_name = (
                                f"{safe_component(source_path.stem, 'image')}"
                                f"_B{band_number:02d}.tif"
                            )
                            _write_band_patch(
                                band, region_directory / patch_name, window
                            )
                            patch_count += 1
                    if write_statistics:
                        row = {
                            "region_id": region.region_id,
                            "region_name": region.region_name,
                            "scene": scene,
                            "source_image": source_path.name,
                            "sensor": sensor,
                            "band_number": band_number,
                            "band_description": _band_description(
                                source_path, band_number
                            ),
                        }
                        row.update(summarize_values(selected))
                        rows.append(row)
                    completed_work += 1
                    report(
                        "band",
                        region_index,
                        region.region_id,
                        source_path.name,
                        band_number,
                        len(band_paths),
                    )
                if write_patches and sensor == "MS" and len(band_paths) >= 3:
                    rgb_name = f"{safe_component(source_path.stem, 'image')}_RGB.tif"
                    _write_rgb_patch(
                        band_paths, region_directory / rgb_name, window, inside
                    )
                    rgb_count += 1
                    completed_work += 1
                    report(
                        "rgb",
                        region_index,
                        region.region_id,
                        source_path.name,
                    )

    statistics_path = output_directory / "roi_statistics.tsv" if write_statistics else None
    if statistics_path is not None:
        report("tsv")
        _atomic_tsv(statistics_path, rows)
        completed_work += 1
        report("complete")
    else:
        report("complete")
    return ROIAnalysisResult(
        output_directory,
        statistics_path,
        len(regions),
        patch_count,
        mask_count,
        rgb_count,
    )


def geometry_display_paths(
    regions: Sequence[ROIRegion], transform: Any, scale_yx: tuple[float, float]
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Convert polygon rings and representative points to napari preview Y,X."""
    inverse = ~transform
    paths: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for region in regions:
        polygons = (
            list(region.geometry.geoms)
            if region.geometry.geom_type == "MultiPolygon"
            else [region.geometry]
        )
        for polygon in polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                xy = np.asarray(ring.coords, dtype=np.float64)
                columns, rows = inverse * (xy[:, 0], xy[:, 1])
                paths.append(
                    np.column_stack(
                        (rows / scale_yx[0], columns / scale_yx[1])
                    )
                )
        point = region.geometry.representative_point()
        column, row = inverse * (point.x, point.y)
        labels.append(np.array([row / scale_yx[0], column / scale_yx[1]]))
        names.append(region.region_name)
    return paths, labels, names
