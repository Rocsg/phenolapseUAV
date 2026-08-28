#!/usr/bin/env python3
"""Manual landmark registration for large GeoTIFF files in napari.

The interactive preview is downsampled. Full-resolution data are read only
during export and are warped separately, one spectral band at a time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np


SIMILARITY_MIN_PAIRS = 4
AFFINE_MIN_PAIRS = 7
TPS_MIN_PAIRS = 10
DEFAULT_PERCENTILES = (0.5, 99.5)


@dataclass
class LandmarkPair:
    """Stable pair identity; coordinates are full-resolution (row, column)."""

    pair_id: int
    fixed_yx: Optional[tuple[float, float]] = None
    moving_yx: Optional[tuple[float, float]] = None
    enabled: bool = True

    @property
    def complete(self) -> bool:
        return self.fixed_yx is not None and self.moving_yx is not None


@dataclass(frozen=True)
class ImageInfo:
    path: str
    shape_cyx: tuple[int, int, int]
    dtype: str
    channel_axis: int


class ImageSource(Protocol):
    """Minimal interface for future TIFF, Zarr, or Dask sources."""

    @property
    def info(self) -> ImageInfo: ...

    def read_preview(self, channel: int, factor: int) -> np.ndarray: ...

    def sample_contrast(
        self, channel: int, percentiles: tuple[float, float]
    ) -> tuple[float, float]: ...


class RasterioSource:
    """Window-readable GeoTIFF source with an explicit C,Y,X axis order."""

    def __init__(self, path: Path) -> None:
        import rasterio

        self.path = path.resolve()
        with rasterio.open(self.path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self._info = ImageInfo(
                path=str(self.path),
                shape_cyx=(src.count, src.height, src.width),
                dtype=src.dtypes[0],
                channel_axis=0,
            )

    @property
    def info(self) -> ImageInfo:
        return self._info

    def _validate_channel(self, channel: int) -> None:
        if not 0 <= channel < self.info.shape_cyx[0]:
            raise ValueError(
                f"Channel {channel + 1} is outside 1..{self.info.shape_cyx[0]}"
            )

    def read_preview(self, channel: int, factor: int) -> np.ndarray:
        import rasterio
        from rasterio.enums import Resampling

        self._validate_channel(channel)
        if factor < 1:
            raise ValueError("Preview factor must be greater than or equal to 1")
        _, height, width = self.info.shape_cyx
        out_shape = (max(1, height // factor), max(1, width // factor))
        with rasterio.open(self.path) as src:
            return src.read(
                channel + 1, out_shape=out_shape, resampling=Resampling.average
            )

    def read_preview_on_grid(
        self, channel: int, factor: int, reference: "RasterioSource"
    ) -> np.ndarray:
        """Project a preview onto the reference image's geographic grid."""
        import rasterio
        from affine import Affine
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        self._validate_channel(channel)
        _, ref_height, ref_width = reference.info.shape_cyx
        out_shape = (max(1, ref_height // factor), max(1, ref_width // factor))
        destination = np.zeros(out_shape, dtype=np.float32)
        destination_transform = reference.transform * Affine.scale(
            ref_width / out_shape[1], ref_height / out_shape[0]
        )
        fallback_crs = self.crs or reference.crs or CRS.from_epsg(3857)
        with rasterio.open(self.path) as src:
            reproject(
                source=rasterio.band(src, channel + 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs or fallback_crs,
                dst_transform=destination_transform,
                dst_crs=reference.crs or fallback_crs,
                resampling=Resampling.average,
                src_nodata=src.nodata,
                dst_nodata=0,
            )
        return destination

    def map_preview_point_to(
        self,
        target: "RasterioSource",
        point_yx: Sequence[float],
        source_preview_shape: tuple[int, int],
        target_preview_shape: tuple[int, int],
    ) -> tuple[float, float]:
        """Convert a preview point to another georeferenced grid."""
        from rasterio.warp import transform as transform_crs

        source_full = self.info.shape_cyx[1:]
        target_full = target.info.shape_cyx[1:]
        row = float(point_yx[0]) * source_full[0] / source_preview_shape[0]
        col = float(point_yx[1]) * source_full[1] / source_preview_shape[1]
        x, y = self.transform * (col + 0.5, row + 0.5)
        if self.crs is not None and target.crs is not None and self.crs != target.crs:
            x_values, y_values = transform_crs(self.crs, target.crs, [x], [y])
            x, y = x_values[0], y_values[0]
        target_col, target_row = (~target.transform) * (x, y)
        return (
            float((target_row - 0.5) * target_preview_shape[0] / target_full[0]),
            float((target_col - 0.5) * target_preview_shape[1] / target_full[1]),
        )

    def sample_contrast(
        self,
        channel: int,
        percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
        max_side: int = 1024,
    ) -> tuple[float, float]:
        """Estimate raw limits while excluding sentinel values."""
        import rasterio
        from rasterio.enums import Resampling

        self._validate_channel(channel)
        _, height, width = self.info.shape_cyx
        scale = min(1.0, max_side / max(height, width))
        out_shape = (max(1, round(height * scale)), max(1, round(width * scale)))
        with rasterio.open(self.path) as src:
            sample = src.read(
                channel + 1, out_shape=out_shape, resampling=Resampling.nearest
            )
        return robust_contrast_limits(sample, percentiles)


def robust_contrast_limits(
    sample: np.ndarray,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> tuple[float, float]:
    """Compute display limits without zeros, saturation, NaN, or infinity."""
    low_p, high_p = percentiles
    if not 0 <= low_p < high_p <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    values = np.asarray(sample).ravel()
    keep = np.isfinite(values) & (values != 0)
    if np.issubdtype(values.dtype, np.integer):
        keep &= values != np.iinfo(values.dtype).max
    elif values.size:
        finite = values[np.isfinite(values)]
        if finite.size and finite.min() >= 0 and finite.max() <= 1:
            keep &= values != 1
    values = values[keep]
    if values.size == 0:
        return (0.0, 1.0)
    low, high = np.percentile(values, (low_p, high_p))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(low) if np.isfinite(low) else 0.0
        return (center, center + 1.0)
    return float(low), float(high)


def yx_to_xy(points_yx: np.ndarray) -> np.ndarray:
    """Convert napari (row, column) coordinates to skimage (x, y)."""
    points = np.asarray(points_yx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected an N x 2 coordinate array")
    return points[:, ::-1].copy()


def full_yx_to_preview_yx(
    point_yx: tuple[float, float], full_shape: tuple[int, int], preview_shape: tuple[int, int]
) -> tuple[float, float]:
    sy = full_shape[0] / preview_shape[0]
    sx = full_shape[1] / preview_shape[1]
    return point_yx[0] / sy, point_yx[1] / sx


def preview_yx_to_full_yx(
    point_yx: Sequence[float], full_shape: tuple[int, int], preview_shape: tuple[int, int]
) -> tuple[float, float]:
    sy = full_shape[0] / preview_shape[0]
    sx = full_shape[1] / preview_shape[1]
    return float(point_yx[0] * sy), float(point_yx[1] * sx)


def complete_enabled_pairs(
    pairs: Sequence[LandmarkPair],
) -> list[LandmarkPair]:
    return [pair for pair in pairs if pair.enabled and pair.complete]


class PixelGeoreferenceMap:
    """Map source-grid pixels to target-grid pixels through georeferencing."""

    def __init__(self, source: RasterioSource, target: RasterioSource) -> None:
        self.source_transform = source.transform
        self.target_inverse_transform = ~target.transform
        self.source_crs = source.crs
        self.target_crs = target.crs
        self._crs_transformer: Optional[Any] = None

    def __call__(self, coordinates_xy: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates_xy, dtype=np.float64)
        columns = coordinates[:, 0] + 0.5
        rows = coordinates[:, 1] + 0.5
        transform = self.source_transform
        world_x = transform.a * columns + transform.b * rows + transform.c
        world_y = transform.d * columns + transform.e * rows + transform.f
        if (
            self.source_crs is not None
            and self.target_crs is not None
            and self.source_crs != self.target_crs
        ):
            if self._crs_transformer is None:
                from pyproj import Transformer

                self._crs_transformer = Transformer.from_crs(
                    self.source_crs, self.target_crs, always_xy=True
                )
            world_x, world_y = self._crs_transformer.transform(world_x, world_y)
        inverse = self.target_inverse_transform
        target_columns = inverse.a * world_x + inverse.b * world_y + inverse.c
        target_rows = inverse.d * world_x + inverse.e * world_y + inverse.f
        return np.column_stack([target_columns - 0.5, target_rows - 0.5])


class ScaledCoordinateMap:
    """Adapt a full-resolution pixel map to source and target previews."""

    def __init__(
        self,
        mapping: Callable[[np.ndarray], np.ndarray],
        source_full_shape: tuple[int, int],
        source_preview_shape: tuple[int, int],
        target_full_shape: tuple[int, int],
        target_preview_shape: tuple[int, int],
    ) -> None:
        self.mapping = mapping
        self.source_scale_xy = np.asarray(
            [
                source_full_shape[1] / source_preview_shape[1],
                source_full_shape[0] / source_preview_shape[0],
            ],
            dtype=np.float64,
        )
        self.target_scale_xy = np.asarray(
            [
                target_full_shape[1] / target_preview_shape[1],
                target_full_shape[0] / target_preview_shape[0],
            ],
            dtype=np.float64,
        )

    def __call__(self, coordinates_xy: np.ndarray) -> np.ndarray:
        full_coordinates = (
            np.asarray(coordinates_xy, dtype=np.float64) * self.source_scale_xy
        )
        return self.mapping(full_coordinates) / self.target_scale_xy


def preview_georeference_map(
    source: RasterioSource,
    target: RasterioSource,
    source_preview_shape: tuple[int, int],
    target_preview_shape: tuple[int, int],
) -> ScaledCoordinateMap:
    return ScaledCoordinateMap(
        PixelGeoreferenceMap(source, target),
        source.info.shape_cyx[1:],
        source_preview_shape,
        target.info.shape_cyx[1:],
        target_preview_shape,
    )


def transform_name(pair_count: int) -> str:
    if pair_count >= TPS_MIN_PAIRS:
        return "georeference + similarity + affine + residual TPS"
    if pair_count >= AFFINE_MIN_PAIRS:
        return "georeference + similarity + residual affine"
    if pair_count >= SIMILARITY_MIN_PAIRS:
        return "georeference + residual similarity"
    if pair_count >= 1:
        return "georeference + residual translation"
    return "georeference"


def estimate_inverse_map(
    fixed_xy: np.ndarray,
    moving_xy: np.ndarray,
    base_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[Any, str]:
    """Estimate a georeferenced mapping with successively finer residuals."""
    from skimage.transform import (
        AffineTransform,
        SimilarityTransform,
        ThinPlateSplineTransform,
    )

    fixed_xy = np.asarray(fixed_xy, dtype=np.float64)
    moving_xy = np.asarray(moving_xy, dtype=np.float64)
    if fixed_xy.shape != moving_xy.shape or fixed_xy.ndim != 2 or fixed_xy.shape[1] != 2:
        raise ValueError("FIXED and MOVING coordinates must have matching N x 2 shapes")
    count = len(fixed_xy)
    base = base_map or SimilarityTransform.identity(dimensionality=2)
    if count == 0:
        return base, "georeference" if base_map is not None else "identity"

    stages: list[Callable[[np.ndarray], np.ndarray]] = [base]
    predicted = base(fixed_xy)
    if count < SIMILARITY_MIN_PAIRS:
        correction = SimilarityTransform(
            translation=np.median(moving_xy - predicted, axis=0)
        )
        stages.append(correction)
        name = "residual translation"
    else:
        similarity = SimilarityTransform.from_estimate(predicted, moving_xy)
        if not similarity:
            raise ValueError("Degenerate landmarks: residual similarity failed")
        stages.append(similarity)
        predicted = similarity(predicted)
        name = "residual similarity"

        if count >= AFFINE_MIN_PAIRS:
            affine = AffineTransform.from_estimate(predicted, moving_xy)
            if not affine:
                raise ValueError("Degenerate landmarks: residual affine failed")
            stages.append(affine)
            predicted = affine(predicted)
            name = "similarity + residual affine"

        if count >= TPS_MIN_PAIRS:
            tps = ThinPlateSplineTransform.from_estimate(predicted, moving_xy)
            if not tps:
                raise ValueError("Degenerate landmarks: residual TPS failed")
            stages.append(tps)
            name = "similarity + affine + residual TPS"

    prefix = "georeference + " if base_map is not None else ""
    return ComposedInverseMap(stages), prefix + name


def compute_preview(
    generation: int,
    moving_preview: np.ndarray,
    fixed_preview_shape: tuple[int, int],
    fixed_points_yx: np.ndarray,
    moving_points_yx: np.ndarray,
    inverse_georeference_map: Callable[[np.ndarray], np.ndarray],
    forward_georeference_map: Callable[[np.ndarray], np.ndarray],
) -> tuple[int, np.ndarray, str, Any, Any]:
    """Compute a preview off the GUI thread without Qt or napari calls."""
    from skimage.transform import warp

    inverse_map, model_name = estimate_inverse_map(
        yx_to_xy(fixed_points_yx),
        yx_to_xy(moving_points_yx),
        inverse_georeference_map,
    )
    forward_map, _ = estimate_inverse_map(
        yx_to_xy(moving_points_yx),
        yx_to_xy(fixed_points_yx),
        forward_georeference_map,
    )
    warped = warp(
        moving_preview,
        inverse_map=inverse_map,
        output_shape=fixed_preview_shape,
        order=1,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    return (
        generation,
        warped.astype(np.float32, copy=False),
        model_name,
        inverse_map,
        forward_map,
    )


def export_full_resolution_tps(
    fixed_path: Path,
    moving_path: Path,
    output_path: Path,
    fixed_points_yx: np.ndarray,
    moving_points_yx: np.ndarray,
) -> str:
    """Warp each moving band independently onto the fixed raster grid."""
    import rasterio
    from skimage.transform import warp

    if len(fixed_points_yx) < TPS_MIN_PAIRS:
        raise ValueError(f"Full-resolution export requires {TPS_MIN_PAIRS} enabled pairs")
    fixed_source = RasterioSource(fixed_path)
    moving_source = RasterioSource(moving_path)
    inverse_map, _ = estimate_inverse_map(
        yx_to_xy(fixed_points_yx),
        yx_to_xy(moving_points_yx),
        PixelGeoreferenceMap(fixed_source, moving_source),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(fixed_path) as fixed, rasterio.open(moving_path) as moving:
        profile = fixed.profile.copy()
        profile.update(
            count=moving.count,
            dtype=moving.dtypes[0],
            nodata=moving.nodata,
            compress="deflate",
            tiled=True,
        )
        dtype = np.dtype(moving.dtypes[0])
        with rasterio.open(output_path, "w", **profile) as destination:
            for band_index in range(1, moving.count + 1):
                raw = moving.read(band_index)
                warped = warp(
                    raw,
                    inverse_map=inverse_map,
                    output_shape=(fixed.height, fixed.width),
                    order=1,
                    mode="constant",
                    cval=0 if moving.nodata is None else moving.nodata,
                    preserve_range=True,
                )
                if np.issubdtype(dtype, np.integer):
                    limits = np.iinfo(dtype)
                    warped = np.rint(np.clip(warped, limits.min, limits.max))
                destination.write(warped.astype(dtype, copy=False), band_index)
                description = moving.descriptions[band_index - 1]
                if description:
                    destination.set_band_description(band_index, description)
            destination.update_tags(
                REGISTRATION="manual_landmarks_tps",
                LANDMARK_PAIRS=str(len(fixed_points_yx)),
                FIXED_IMAGE=str(Path(fixed_path).resolve()),
                MOVING_IMAGE=str(Path(moving_path).resolve()),
            )
    return str(output_path)


class ComposedInverseMap:
    """Compose output-to-input pixel mappings along a registration path."""

    def __init__(self, mappings: Sequence[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.mappings = tuple(mappings)

    def __call__(self, coordinates_xy: np.ndarray) -> np.ndarray:
        result = np.asarray(coordinates_xy, dtype=np.float64)
        for mapping in self.mappings:
            result = mapping(result)
        return result


def _session_staged_inverse_map(
    session: dict[str, Any], fixed_path: Path, moving_path: Path
) -> Any:
    """Build one edge's georeferenced staged residual mapping."""
    pairs = [
        item
        for item in session.get("landmarks_full_resolution_yx", [])
        if bool(item.get("enabled", True))
        and item.get("fixed_yx") is not None
        and item.get("moving_yx") is not None
    ]
    if len(pairs) < TPS_MIN_PAIRS:
        raise ValueError(
            f"A series edge has only {len(pairs)} complete enabled pair(s); "
            f"{TPS_MIN_PAIRS} are required"
        )
    fixed_yx = np.asarray([item["fixed_yx"] for item in pairs], dtype=np.float64)
    moving_yx = np.asarray([item["moving_yx"] for item in pairs], dtype=np.float64)
    model, _ = estimate_inverse_map(
        yx_to_xy(fixed_yx),
        yx_to_xy(moving_yx),
        PixelGeoreferenceMap(
            RasterioSource(fixed_path), RasterioSource(moving_path)
        ),
    )
    return model


def _write_registered_band_files(
    reference_path: Path,
    source_path: Path,
    bands_directory: Path,
    inverse_map: Optional[Callable[[np.ndarray], np.ndarray]],
) -> list[str]:
    """Write each source band as one 2-D raster on the reference grid."""
    import rasterio
    from skimage.transform import warp

    bands_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    with rasterio.open(reference_path) as reference, rasterio.open(source_path) as source:
        dtype = np.dtype(source.dtypes[0])
        for band_index in range(1, source.count + 1):
            output_path = (
                bands_directory
                / f"{source_path.stem}_B{band_index:02d}_registered.tif"
            )
            profile = reference.profile.copy()
            profile.update(
                count=1,
                dtype=source.dtypes[0],
                nodata=source.nodata,
                compress="deflate",
            )
            with rasterio.open(output_path, "w", **profile) as destination:
                raw = None if inverse_map is None else source.read(band_index)
                if inverse_map is None and (
                    source.height != reference.height
                    or source.width != reference.width
                ):
                    raise ValueError(
                        "The reference image dimensions changed during series export"
                    )
                for _, window in destination.block_windows(1):
                    if inverse_map is None:
                        registered = source.read(band_index, window=window)
                    else:
                        column_offset = float(window.col_off)
                        row_offset = float(window.row_off)

                        def block_inverse_map(
                            coordinates_xy: np.ndarray,
                            column_offset: float = column_offset,
                            row_offset: float = row_offset,
                        ) -> np.ndarray:
                            coordinates_xy = np.asarray(
                                coordinates_xy, dtype=np.float64
                            ).copy()
                            coordinates_xy[:, 0] += column_offset
                            coordinates_xy[:, 1] += row_offset
                            return inverse_map(coordinates_xy)

                        registered = warp(
                            raw,
                            inverse_map=block_inverse_map,
                            output_shape=(int(window.height), int(window.width)),
                            order=1,
                            mode="constant",
                            cval=0 if source.nodata is None else source.nodata,
                            preserve_range=True,
                        )
                        if np.issubdtype(dtype, np.integer):
                            limits = np.iinfo(dtype)
                            registered = np.rint(
                                np.clip(registered, limits.min, limits.max)
                            )
                        registered = registered.astype(dtype, copy=False)
                    destination.write(registered, 1, window=window)
                description = source.descriptions[band_index - 1]
                destination.set_band_description(
                    1, description or f"{source_path.name} band {band_index}"
                )
                destination.update_tags(
                    REGISTRATION=(
                        "reference_copy" if inverse_map is None else "manual_series_tps"
                    ),
                    REFERENCE_IMAGE=str(reference_path.resolve()),
                    SOURCE_IMAGE=str(source_path.resolve()),
                    SOURCE_BAND=str(band_index),
                )
            outputs.append(str(output_path))
    return outputs


def export_registration_series(
    project: dict[str, Any], reference_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Export every image reachable from the reference through directed edges."""
    reference_path = reference_path.resolve()
    output_directory = output_directory.resolve()
    bands_directory = output_directory / "BANDS"
    image_paths = [Path(item["path"]).resolve() for item in project.get("images", [])]
    if reference_path not in image_paths:
        image_paths.insert(0, reference_path)

    adjacency: dict[Path, list[tuple[Path, Any]]] = {}
    skipped_edges: list[str] = []
    for item in project.get("registrations", []):
        fixed_path = Path(item["fixed_path"]).resolve()
        moving_path = Path(item["moving_path"]).resolve()
        try:
            mapping = _session_staged_inverse_map(
                item.get("session", {}), fixed_path, moving_path
            )
        except ValueError as error:
            skipped_edges.append(f"{fixed_path.name} -> {moving_path.name}: {error}")
            continue
        adjacency.setdefault(fixed_path, []).append((moving_path, mapping))

    mappings: dict[Path, Optional[Callable[[np.ndarray], np.ndarray]]] = {
        reference_path: None
    }
    paths: dict[Path, tuple[Any, ...]] = {reference_path: ()}
    queue = [reference_path]
    while queue:
        fixed_path = queue.pop(0)
        for moving_path, edge_mapping in adjacency.get(fixed_path, []):
            if moving_path in mappings:
                continue
            path = paths[fixed_path] + (edge_mapping,)
            paths[moving_path] = path
            mappings[moving_path] = ComposedInverseMap(path)
            queue.append(moving_path)

    unreachable = [str(path) for path in image_paths if path not in mappings]
    if unreachable:
        names = ", ".join(Path(path).name for path in unreachable[:5])
        if len(unreachable) > 5:
            names += f", and {len(unreachable) - 5} more"
        edge_details = "; ".join(skipped_edges[:3])
        suffix = f" Invalid edges: {edge_details}" if edge_details else ""
        raise ValueError(
            "Every series image must be reachable through directed, export-ready "
            f"edges from {reference_path.name}. Unreachable: {names}.{suffix}"
        )

    output_by_image: dict[str, list[str]] = {}
    for image_path in image_paths:
        if image_path not in mappings:
            continue
        output_by_image[str(image_path)] = _write_registered_band_files(
            reference_path,
            image_path,
            bands_directory,
            mappings[image_path],
        )

    if len(output_by_image) < 2:
        details = "; ".join(skipped_edges[:3])
        suffix = f" ({details})" if details else ""
        raise ValueError(
            "No moving series image is reachable from the selected reference"
            f"{suffix}"
        )
    return {
        "output_directory": str(output_directory),
        "reference_path": str(reference_path),
        "outputs": output_by_image,
        "unreachable": [],
        "skipped_edges": skipped_edges,
    }


def export_registration_pair_bands(
    fixed_path: Path,
    moving_path: Path,
    output_directory: Path,
    fixed_points_yx: np.ndarray,
    moving_points_yx: np.ndarray,
) -> dict[str, Any]:
    """Export a standalone FIXED/MOVING pair as separate 2-D band files."""
    if len(fixed_points_yx) < TPS_MIN_PAIRS:
        raise ValueError(f"Full-resolution export requires {TPS_MIN_PAIRS} enabled pairs")
    inverse_map, _ = estimate_inverse_map(
        yx_to_xy(fixed_points_yx),
        yx_to_xy(moving_points_yx),
        PixelGeoreferenceMap(
            RasterioSource(fixed_path), RasterioSource(moving_path)
        ),
    )
    output_directory = output_directory.resolve()
    bands_directory = output_directory / "BANDS"
    outputs = {
        str(fixed_path.resolve()): _write_registered_band_files(
            fixed_path, fixed_path, bands_directory, None
        ),
        str(moving_path.resolve()): _write_registered_band_files(
            fixed_path, moving_path, bands_directory, inverse_map
        ),
    }
    return {
        "output_directory": str(output_directory),
        "reference_path": str(fixed_path.resolve()),
        "outputs": outputs,
        "unreachable": [],
        "skipped_edges": [],
    }


def create_sample_result(
    output_path: Path,
    first_band_paths: Sequence[Path],
    center_yx: Sequence[float],
    size: int = 512,
) -> tuple[str, np.ndarray, list[str]]:
    """Create a boundless square crop stack from registered first-band files."""
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    if not first_band_paths:
        raise ValueError("No exported first-band images are available")
    row = int(round(float(center_yx[0])))
    column = int(round(float(center_yx[1])))
    row_offset = row - size // 2
    column_offset = column - size // 2
    window = Window(column_offset, row_offset, size, size)

    arrays: list[np.ndarray] = []
    names: list[str] = []
    profiles: list[dict[str, Any]] = []
    for path in first_band_paths:
        with rasterio.open(path) as source:
            fill_value = 0 if source.nodata is None else source.nodata
            arrays.append(
                source.read(1, window=window, boundless=True, fill_value=fill_value)
            )
            names.append(Path(path).stem)
            profiles.append(source.profile.copy())

    common_dtype = np.result_type(*(array.dtype for array in arrays))
    stack = np.stack(
        [array.astype(common_dtype, copy=False) for array in arrays], axis=0
    )
    profile = profiles[0]
    profile.update(
        count=len(arrays),
        height=size,
        width=size,
        dtype=np.dtype(common_dtype).name,
        transform=profile["transform"] * Affine.translation(column_offset, row_offset),
        compress="deflate",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(stack)
        for band_index, name in enumerate(names, start=1):
            destination.set_band_description(band_index, name)
        destination.update_tags(
            SAMPLE_CENTER_Y=str(row),
            SAMPLE_CENTER_X=str(column),
            SAMPLE_SIZE=str(size),
        )
    return str(output_path), stack, names


class RegistrationController:
    """Qt controller; all layer and UI writes remain on the GUI thread."""

    def __init__(
        self,
        viewer: Any,
        fixed_source: RasterioSource,
        moving_source: RasterioSource,
        preview_factor: int = 4,
        percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
    ) -> None:
        from qtpy.QtCore import QEvent, QObject, QPoint, Qt, QTimer
        from qtpy.QtGui import QCursor
        from qtpy.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        self.viewer = viewer
        self.fixed_source = fixed_source
        self.moving_source = moving_source
        self.preview_factor = preview_factor
        self.percentiles = percentiles
        self.fixed_channel = 0
        self.moving_channel = 0
        self.pairs: dict[int, LandmarkPair] = {}
        self._next_pair_id = 1
        self._pending_pair_id: Optional[int] = None
        self._layer_ids: dict[str, list[int]] = {"fixed": [], "moving": []}
        self._text_generations: dict[int, int] = {}
        self._suspend_events = False
        self._space_held = False
        self._space_generation = 0
        self._generation = 0
        self._scheduled_preview_signature: tuple[Any, ...] = ()
        self._preview_pending = False
        self._fast_phase = "idle"
        self._view_mode = "fixed"
        self._display_inverse_map: Optional[Any] = None
        self._display_forward_map: Optional[Any] = None
        self._active_workers: set[Any] = set()
        self._export_result: Optional[dict[str, Any]] = None
        self._sample_selecting = False
        self._sample_layer: Optional[Any] = None
        self._sample_reference_layer: Optional[Any] = None
        self._sample_reference_full_shape: Optional[tuple[int, int]] = None
        self._sample_reference_preview_shape: Optional[tuple[int, int]] = None
        self.session_changed_callbacks: list[Any] = []

        self.fixed_preview = fixed_source.read_preview(0, preview_factor)
        self.moving_preview = moving_source.read_preview(0, preview_factor)
        initial_registered = moving_source.read_preview_on_grid(
            0, preview_factor, fixed_source
        )
        fixed_limits = fixed_source.sample_contrast(0, percentiles)
        moving_limits = moving_source.sample_contrast(0, percentiles)

        self.fixed_image = viewer.add_image(
            self.fixed_preview,
            name="FIXED",
            colormap="red",
            blending="additive",
            contrast_limits=fixed_limits,
        )
        self.moving_image = viewer.add_image(
            self.moving_preview,
            name="Raw MOVING",
            colormap="green",
            blending="additive",
            contrast_limits=moving_limits,
            visible=False,
        )
        self.registered_image = viewer.add_image(
            initial_registered,
            name="Registered MOVING preview",
            colormap="green",
            blending="additive",
            contrast_limits=moving_limits,
            visible=False,
        )
        empty = np.empty((0, 2), dtype=np.float64)
        point_style = dict(
            size=3,
            opacity=0.48,
            text={"string": "{pair_id}", "size": 9},
        )
        self.fixed_points = viewer.add_points(
            empty,
            name="FIXED landmarks",
            features={"pair_id": np.array([], dtype=int)},
            face_color=[1.0, 0.72, 0.72, 1.0],
            border_color=[1.0, 0.9, 0.9, 0.8],
            **point_style,
        )
        self.moving_points = viewer.add_points(
            empty,
            name="MOVING landmarks",
            features={"pair_id": np.array([], dtype=int)},
            face_color=[0.72, 1.0, 0.72, 1.0],
            border_color=[0.9, 1.0, 0.9, 0.8],
            visible=False,
            **point_style,
        )
        self.fixed_centers = viewer.add_points(
            empty,
            name="FIXED landmark centers",
            size=0.75,
            face_color=[1.0, 0.0, 0.0, 1.0],
            border_width=0,
        )
        self.fixed_centers.editable = False
        self.moving_centers = viewer.add_points(
            empty,
            name="MOVING landmark centers",
            size=0.75,
            face_color=[0.0, 1.0, 0.0, 1.0],
            border_width=0,
            visible=False,
        )
        self.moving_centers.editable = False
        self.fixed_points.events.data.connect(lambda event: self._on_points_changed("fixed"))
        self.moving_points.events.data.connect(lambda event: self._on_points_changed("moving"))
        self._text_generations[id(self.fixed_points)] = 0
        self._text_generations[id(self.moving_points)] = 0

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        legend = QLabel(
            "<b><span style='color:#ff5555'>FIXED — red — left click</span>"
            " &nbsp;→&nbsp; "
            "<span style='color:#55cc66'>MOVING — green — right click</span></b>"
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        mode_box = QGroupBox("Display / input")
        mode_layout = QGridLayout(mode_box)
        buttons = (
            ("FIXED", self.show_fixed),
            ("MOVING", self.show_moving),
            ("Red/green overlay", self.show_overlay),
            ("New pair (N)", self.start_next_pair),
            ("Delete latest pair", self.delete_last_pair),
        )
        for index, (label, callback) in enumerate(buttons):
            button = QPushButton(label)
            if label == "FIXED":
                button.setStyleSheet("color: #ff5555; font-weight: bold;")
            elif label == "MOVING":
                button.setStyleSheet("color: #55cc66; font-weight: bold;")
            button.clicked.connect(callback)
            mode_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(mode_box)

        channel_row = QHBoxLayout()
        self.fixed_combo = QComboBox()
        self.moving_combo = QComboBox()
        self.fixed_combo.addItems(
            [f"FIXED B{i + 1}" for i in range(fixed_source.info.shape_cyx[0])]
        )
        self.moving_combo.addItems(
            [f"MOVING B{i + 1}" for i in range(moving_source.info.shape_cyx[0])]
        )
        self.fixed_combo.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.moving_combo.setStyleSheet("color: #55cc66; font-weight: bold;")
        self.fixed_combo.currentIndexChanged.connect(self._change_fixed_channel)
        self.moving_combo.currentIndexChanged.connect(self._change_moving_channel)
        channel_row.addWidget(self.fixed_combo)
        channel_row.addWidget(self.moving_combo)
        layout.addLayout(channel_row)

        marker_row = QHBoxLayout()
        marker_row.addWidget(QLabel("Landmark size"))
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(0.5, 20.0)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setDecimals(1)
        self.marker_size.setValue(3.0)
        self.marker_size.setToolTip("Circle diameter in preview units")
        self.marker_size.valueChanged.connect(self._change_marker_size)
        marker_row.addWidget(self.marker_size)
        marker_row.addStretch(1)
        layout.addLayout(marker_row)

        percentile_row = QHBoxLayout()
        self.low_percentile = QComboBox()
        self.high_percentile = QComboBox()
        self.low_percentile.addItems(["0.1", "0.5", "1.0", "2.0"])
        self.high_percentile.addItems(["98.0", "99.0", "99.5", "99.9"])
        self.low_percentile.setCurrentText(str(percentiles[0]))
        self.high_percentile.setCurrentText(str(percentiles[1]))
        contrast_button = QPushButton("Recompute contrast")
        contrast_button.clicked.connect(self._update_contrast)
        percentile_row.addWidget(QLabel("Contrast percentiles"))
        percentile_row.addWidget(self.low_percentile)
        percentile_row.addWidget(self.high_percentile)
        percentile_row.addWidget(contrast_button)
        layout.addLayout(percentile_row)

        self.status = QLabel(
            "Recommended order: red FIXED (left), then green MOVING (right). "
            "4=similarity, 7=residual affine, 10=residual TPS."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.updating_status = QLabel("Updating view…")
        self.updating_status.setStyleSheet(
            "color: #ffcc55; font-weight: bold; padding: 3px;"
        )
        self.updating_status.setVisible(False)
        layout.addWidget(self.updating_status)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "FIXED Y", "FIXED X", "MOVING Y", "MOVING X", "Active"])
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.setMaximumHeight(220)
        layout.addWidget(self.table)

        persistence_row = QHBoxLayout()
        save_button = QPushButton("Save state")
        save_button.clicked.connect(self.save_state)
        persistence_row.addWidget(save_button)
        export_button = QPushButton("Apply / Export TPS")
        export_button.clicked.connect(self.export_dialog)
        persistence_row.addWidget(export_button)
        self.sample_button = QPushButton("Create 512×512 sample")
        self.sample_button.setEnabled(False)
        self.sample_button.setToolTip(
            "Enabled after export; click a center point on the registered series"
        )
        self.sample_button.clicked.connect(self._begin_sample_selection)
        persistence_row.addWidget(self.sample_button)
        layout.addLayout(persistence_row)

        self._QTableWidgetItem = QTableWidgetItem
        self._QCheckBox = QCheckBox
        self._QFileDialog = QFileDialog
        self._QMessageBox = QMessageBox
        self._QTimer = QTimer
        self._QCursor = QCursor
        self._QPoint = QPoint

        canvas_widget = self._canvas_widget()
        self.canvas_message = QLabel(canvas_widget)
        self.canvas_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_message.setWordWrap(True)
        self.canvas_message.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.canvas_message.hide()

        self._hover_pair_id: Optional[int] = None
        self.hover_delete_label = QLabel()
        self.hover_delete_label.setWindowFlags(Qt.WindowType.ToolTip)
        self.hover_delete_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.hover_delete_label.setStyleSheet(
            "background: #242424; color: white; border: 1px solid #aaaaaa; "
            "padding: 5px;"
        )
        self.hover_delete_label.hide()

        controller = self

        class SpaceKeyFilter(QObject):
            """Observe Space without replacing napari's temporary pan binding."""

            def eventFilter(self, watched: Any, event: Any) -> bool:
                event_type = event.type()
                if event_type in (QEvent.Type.ApplicationDeactivate, QEvent.Type.WindowDeactivate):
                    controller._space_held = False
                elif (
                    event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
                    and event.key() == Qt.Key.Key_Space
                    and not event.isAutoRepeat()
                ):
                    pressed = event_type == QEvent.Type.KeyPress
                    if pressed and not controller._space_held:
                        controller._space_generation += 1
                    controller._space_held = pressed
                return False

        self._space_key_filter = SpaceKeyFilter()
        QApplication.instance().installEventFilter(self._space_key_filter)
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._launch_preview)
        from series_registration import SeriesRegistrationWidget

        self.series_widget = SeriesRegistrationWidget(self)
        layout.addWidget(self.series_widget, stretch=1)
        viewer.mouse_drag_callbacks.insert(0, self._on_canvas_click)
        viewer.mouse_move_callbacks.append(self._on_canvas_move)
        viewer.bind_key("n", self.start_next_pair, overwrite=True)
        viewer.bind_key("f", self.show_fixed, overwrite=True)
        viewer.bind_key("m", self.show_moving, overwrite=True)
        viewer.bind_key("o", self.show_overlay, overwrite=True)
        viewer.bind_key("Delete", self.delete_last_pair, overwrite=True)
        viewer.bind_key("Shift-Delete", self.delete_last_pair, overwrite=True)
        self.fixed_points.bind_key("Delete", self.delete_last_pair, overwrite=True)
        self.moving_points.bind_key("Delete", self.delete_last_pair, overwrite=True)
        self.fixed_centers.bind_key("Delete", self.delete_last_pair, overwrite=True)
        self.moving_centers.bind_key("Delete", self.delete_last_pair, overwrite=True)
        self.show_overlay()

    @staticmethod
    def _mouse_button(event: Any) -> int:
        """Normalize Vispy/Qt mouse-button representations."""
        button = getattr(event, "button", 0)
        name = str(getattr(button, "name", button)).lower()
        if "left" in name:
            return 1
        if "right" in name:
            return 2
        if "middle" in name or "mid" in name:
            return 3
        value = getattr(button, "value", button)
        try:
            numeric = int(value)
            return 3 if numeric == 4 else numeric
        except (TypeError, ValueError):
            return 0

    def _on_canvas_click(self, viewer: Any, event: Any):
        """Left=FIXED and right=MOVING, regardless of the active layer.

        The generator distinguishes clicks from drags. Right-button events are
        consumed here so napari never interprets them as zoom operations.
        """
        button = self._mouse_button(event)
        if button not in (1, 2, 3):
            return
        if button == 2:
            event.handled = True
        space_generation = self._space_generation
        space_was_held = self._space_held
        start_position = np.asarray(event.position[-2:], dtype=np.float64)
        hovered_pair_id = (
            self._pair_id_at_visible_landmark(start_position)
            if button == 1 and not self._sample_selecting
            else None
        )
        if button == 1 and (hovered_pair_id is not None or self._sample_selecting):
            event.handled = True
        moved = False
        yield
        while event.type == "mouse_move":
            if button == 2:
                event.handled = True
            moved = True
            yield
        if (
            moved
            or space_was_held
            or self._space_held
            or self._space_generation != space_generation
        ):
            return
        if button == 3:
            self._delete_pair_at_display_position(start_position)
            return
        if button == 1 and self._sample_selecting:
            self._create_sample_at_display_position(start_position)
            return
        if hovered_pair_id is not None:
            self._remove_pairs_and_advance({hovered_pair_id})
            self._hover_pair_id = None
            self.hover_delete_label.hide()
            return
        if self._preview_pending:
            self.status.setText(
                "Update in progress: landmark placement and movement are temporarily locked."
            )
            return
        side = "fixed" if button == 1 else "moving"
        target_yx = self._display_to_source_preview_yx(side, start_position)
        self._record_canvas_point(side, target_yx)

    def _on_canvas_move(self, viewer: Any, event: Any) -> None:
        """Show a deletion action when the pointer hovers over a landmark."""
        if self._sample_selecting:
            self._hover_pair_id = None
            self.hover_delete_label.hide()
            return
        pair_id = self._pair_id_at_visible_landmark(
            np.asarray(event.position[-2:], dtype=np.float64)
        )
        self._hover_pair_id = pair_id
        if pair_id is None:
            self.hover_delete_label.hide()
            return
        self.hover_delete_label.setText(
            f"Click point {pair_id} to delete the pair"
        )
        self.hover_delete_label.adjustSize()
        self.hover_delete_label.move(self._QCursor.pos() + self._QPoint(12, 12))
        self.hover_delete_label.show()

    def _pair_id_at_visible_landmark(
        self, display_yx: np.ndarray
    ) -> Optional[int]:
        """Return the stable ID of the nearest visible landmark."""
        if self._view_mode in ("moving_raw", "moving_registered"):
            candidates = (("moving", self.moving_points),)
        elif self._view_mode == "overlay":
            candidates = (
                ("fixed", self.fixed_points),
                ("moving", self.moving_points),
            )
        else:
            candidates = (("fixed", self.fixed_points),)
        zoom = max(float(self.viewer.camera.zoom), 1e-9)
        tolerance = 8.0 / zoom
        nearest: Optional[tuple[float, int]] = None
        for side, layer in candidates:
            if not layer.visible or len(layer.data) == 0:
                continue
            distances = np.linalg.norm(
                np.asarray(layer.data, dtype=np.float64) - display_yx, axis=1
            )
            index = int(np.argmin(distances))
            if distances[index] > tolerance or index >= len(self._layer_ids[side]):
                continue
            candidate = (float(distances[index]), self._layer_ids[side][index])
            if nearest is None or candidate[0] < nearest[0]:
                nearest = candidate
        return None if nearest is None else nearest[1]

    def _delete_pair_at_display_position(self, display_yx: np.ndarray) -> None:
        pair_id = self._pair_id_at_visible_landmark(display_yx)
        if pair_id is None:
            self.status.setText("Middle click missed all landmarks; no pair was deleted.")
            return
        self._remove_pairs_and_advance({pair_id})

    def _display_to_source_preview_yx(
        self, side: str, display_yx: np.ndarray
    ) -> tuple[float, float]:
        """Convert a click to FIXED or MOVING preview coordinates."""
        display_yx = np.asarray(display_yx, dtype=np.float64)
        display_is_fixed = self._view_mode in (
            "fixed",
            "overlay",
            "moving_registered",
        )
        if (display_is_fixed and side == "fixed") or (
            not display_is_fixed and side == "moving"
        ):
            return float(display_yx[0]), float(display_yx[1])

        complete = complete_enabled_pairs(list(self.pairs.values()))
        if (
            not complete
            or self._display_inverse_map is None
            or self._display_forward_map is None
        ):
            source = self.fixed_source if display_is_fixed else self.moving_source
            target = self.moving_source if display_is_fixed else self.fixed_source
            source_shape = self.fixed_preview.shape if display_is_fixed else self.moving_preview.shape
            target_shape = self.moving_preview.shape if display_is_fixed else self.fixed_preview.shape
            return source.map_preview_point_to(
                target, display_yx, source_shape, target_shape
            )

        # The overlay uses the FIXED grid. A right click therefore follows the
        # exact FIXED -> raw MOVING mapping that produced the visible green pixels.
        if display_is_fixed:
            mapping = self._display_inverse_map
        else:
            mapping = self._display_forward_map
        mapped_xy = mapping(yx_to_xy(display_yx.reshape(1, 2)))[0]
        return float(mapped_xy[1]), float(mapped_xy[0])

    def _record_canvas_point(self, side: str, preview_yx: tuple[float, float]) -> None:
        """Store a click under an explicit ID without relying on layer context."""
        attr = "fixed_yx" if side == "fixed" else "moving_yx"
        if self._pending_pair_id is not None:
            pair_id = self._pending_pair_id
            pair = self.pairs.setdefault(pair_id, LandmarkPair(pair_id))
            if getattr(pair, attr) is not None:
                pair_id = -1
        else:
            pair_id = -1
        if pair_id < 0 and side == "moving":
            candidates = [
                pair.pair_id
                for pair in self.pairs.values()
                if pair.fixed_yx is not None and pair.moving_yx is None
            ]
            if not candidates:
                candidates = [
                    pair.pair_id
                    for pair in self.pairs.values()
                    if pair.moving_yx is None
                ]
        elif pair_id < 0:
            candidates = [
                pair.pair_id
                for pair in self.pairs.values()
                if pair.fixed_yx is None
            ]
        else:
            candidates = []
        if pair_id < 0:
            if not candidates:
                pair_id = self._next_id()
                self.pairs[pair_id] = LandmarkPair(pair_id)
            else:
                pair_id = min(candidates)
        pair = self.pairs[pair_id]
        full_yx = preview_yx_to_full_yx(
            preview_yx, self._full_shape(side), self._preview_shape(side)
        )
        setattr(pair, attr, full_yx)
        self._fast_phase = "idle" if pair.complete else side
        self._resync_layers_from_pairs()
        other = "right-click MOVING" if side == "fixed" else "left-click FIXED"
        state = "complete" if pair.complete else f"incomplete — waiting for {other}"
        self.status.setText(f"Pair {pair_id} is {state}.")
        if pair.complete:
            if self._pending_pair_id == pair_id:
                self._pending_pair_id = None
            self._schedule_preview()
        else:
            self._notify_session_changed()
            self._show_canvas_message(
                f"Pair {pair_id}\nWaiting for {other}",
                "#55dd66" if side == "fixed" else "#ff5555",
            )

    def _next_id(self) -> int:
        pair_id = max(self._next_pair_id, max(self.pairs, default=0) + 1)
        self._next_pair_id = pair_id + 1
        return pair_id

    def _full_shape(self, side: str) -> tuple[int, int]:
        source = self.fixed_source if side == "fixed" else self.moving_source
        return source.info.shape_cyx[1:]

    def _preview_shape(self, side: str) -> tuple[int, int]:
        return self.fixed_preview.shape if side == "fixed" else self.moving_preview.shape

    def _on_points_changed(self, side: str) -> None:
        if self._suspend_events:
            return
        layer = self.fixed_points if side == "fixed" else self.moving_points
        data = np.asarray(layer.data, dtype=np.float64)
        old_ids = self._layer_ids[side]
        try:
            feature_ids = [int(value) for value in layer.features["pair_id"].tolist()]
        except Exception:
            feature_ids = []
        ids: list[int] = []
        seen: set[int] = set()
        for index in range(len(data)):
            candidate = feature_ids[index] if index < len(feature_ids) else 0
            if candidate <= 0 or candidate in seen or candidate not in self.pairs:
                candidate = self._id_for_new_point(side)
            ids.append(candidate)
            seen.add(candidate)

        for deleted_id in set(old_ids) - set(ids):
            pair = self.pairs.get(deleted_id)
            if pair is not None:
                if side == "fixed":
                    pair.fixed_yx = None
                else:
                    pair.moving_yx = None
        for pair_id, preview_yx in zip(ids, data):
            pair = self.pairs.setdefault(pair_id, LandmarkPair(pair_id))
            source_preview_yx = preview_yx
            if side == "moving" and self._view_mode in (
                "moving_registered",
                "overlay",
            ):
                source_preview_yx = self._display_to_source_preview_yx(
                    "moving", preview_yx
                )
            full_yx = preview_yx_to_full_yx(
                source_preview_yx,
                self._full_shape(side),
                self._preview_shape(side),
            )
            if side == "fixed":
                pair.fixed_yx = full_yx
            else:
                pair.moving_yx = full_yx

        self._layer_ids[side] = ids
        self._suspend_events = True
        try:
            layer.text.visible = False
            generation = self._text_generations.get(id(layer), 0) + 1
            self._text_generations[id(layer)] = generation
            layer.features = {"pair_id": np.asarray(ids, dtype=int)}
            self._defer_point_text_restore(layer, generation)
        finally:
            self._suspend_events = False
        self._refresh_table()
        self._advance_fast_mode(side, ids)
        self._schedule_preview()

    def _id_for_new_point(self, side: str) -> int:
        if self._pending_pair_id is not None:
            pair_id = self._pending_pair_id
            self.pairs.setdefault(pair_id, LandmarkPair(pair_id))
            return pair_id
        if side == "moving" and self._fast_phase == "moving":
            incomplete = [p.pair_id for p in self.pairs.values() if p.fixed_yx is not None and p.moving_yx is None]
            if incomplete:
                return max(incomplete)
        attr = "fixed_yx" if side == "fixed" else "moving_yx"
        missing = [p.pair_id for p in self.pairs.values() if getattr(p, attr) is None]
        if missing:
            return min(missing)
        pair_id = self._next_id()
        self.pairs[pair_id] = LandmarkPair(pair_id)
        return pair_id

    def _advance_fast_mode(self, side: str, ids: list[int]) -> None:
        if self._fast_phase == "fixed" and side == "fixed" and ids:
            self._fast_phase = "moving"
            self.status.setText(
                f"Pair {ids[-1]}: waiting for right-click MOVING (view unchanged)."
            )
        elif self._fast_phase == "moving" and side == "moving" and ids:
            pair = self.pairs.get(ids[-1])
            if pair and pair.complete:
                self._fast_phase = "idle"

    def _preview_signature(self) -> tuple[Any, ...]:
        """Describe only the pairs used by registration."""
        return tuple(
            (pair.pair_id, pair.fixed_yx, pair.moving_yx)
            for pair in complete_enabled_pairs(list(self.pairs.values()))
        )

    def _notify_session_changed(self) -> None:
        for callback in tuple(self.session_changed_callbacks):
            callback()

    def _schedule_preview(self, force: bool = False) -> None:
        fixed_count = len(self.fixed_points.data)
        moving_count = len(self.moving_points.data)
        complete = complete_enabled_pairs(list(self.pairs.values()))
        signature = self._preview_signature()
        self._notify_session_changed()
        if not force and signature == self._scheduled_preview_signature:
            return
        self._scheduled_preview_signature = signature
        warning = ""
        if fixed_count != moving_count:
            warning = f" WARNING: {fixed_count} FIXED / {moving_count} MOVING; incomplete IDs retained."
        self.status.setText(
            f"{len(complete)} active pair(s): {transform_name(len(complete))}.{warning}"
        )
        self._generation += 1
        self._set_preview_pending(bool(complete))
        self._preview_timer.start(180)

    def _set_preview_pending(self, pending: bool) -> None:
        """Display computation status and lock only landmark editing."""
        self._preview_pending = pending
        self.fixed_points.editable = not pending
        self.moving_points.editable = not pending
        self.updating_status.setVisible(pending)
        if pending:
            self._show_canvas_message(
                "UPDATING VIEW…\nNavigation remains available",
                "#ffcc55",
            )
        elif self.canvas_message.text().startswith("UPDATING VIEW"):
            self.canvas_message.hide()

    def _canvas_widget(self) -> Any:
        """Find the native canvas widget across supported napari versions."""
        qt_viewer = getattr(self.viewer.window, "qt_viewer", None)
        if qt_viewer is None:
            qt_viewer = getattr(self.viewer.window, "_qt_viewer", None)
        canvas = getattr(qt_viewer, "canvas", None)
        return getattr(canvas, "native", self.widget)

    def _show_canvas_message(self, text: str, border_color: str) -> None:
        """Display a non-blocking message at the center of the canvas."""
        self.canvas_message.setText(text)
        self.canvas_message.setStyleSheet(
            "background: rgba(20, 20, 20, 210); color: white; "
            "font-size: 24px; font-weight: bold; "
            f"border: 3px solid {border_color}; border-radius: 10px; padding: 18px;"
        )
        parent = self.canvas_message.parentWidget()
        width = min(620, max(320, parent.width() - 40))
        self.canvas_message.setFixedWidth(width)
        self.canvas_message.adjustSize()
        self.canvas_message.move(
            max(0, (parent.width() - self.canvas_message.width()) // 2),
            max(0, (parent.height() - self.canvas_message.height()) // 2),
        )
        self.canvas_message.raise_()
        self.canvas_message.show()

    def _launch_preview(self) -> None:
        from napari.qt.threading import create_worker

        complete = complete_enabled_pairs(list(self.pairs.values()))
        if not complete:
            self.registered_image.data = self.moving_source.read_preview_on_grid(
                self.moving_channel, self.preview_factor, self.fixed_source
            )
            self._display_inverse_map = None
            self._display_forward_map = None
            self._set_preview_pending(False)
            if self._view_mode == "moving_registered":
                self.show_moving()
            return
        fixed_yx = np.asarray(
            [
                full_yx_to_preview_yx(pair.fixed_yx, self._full_shape("fixed"), self.fixed_preview.shape)
                for pair in complete
            ]
        )
        moving_yx = np.asarray(
            [
                full_yx_to_preview_yx(pair.moving_yx, self._full_shape("moving"), self.moving_preview.shape)
                for pair in complete
            ]
        )
        generation = self._generation
        worker = create_worker(
            compute_preview,
            generation,
            self.moving_preview.copy(),
            self.fixed_preview.shape,
            fixed_yx,
            moving_yx,
            preview_georeference_map(
                self.fixed_source,
                self.moving_source,
                self.fixed_preview.shape,
                self.moving_preview.shape,
            ),
            preview_georeference_map(
                self.moving_source,
                self.fixed_source,
                self.moving_preview.shape,
                self.fixed_preview.shape,
            ),
        )
        self._active_workers.add(worker)
        worker.returned.connect(self._accept_preview)
        worker.errored.connect(
            lambda error, failed_generation=generation: self._preview_failed(
                error, failed_generation
            )
        )
        worker.finished.connect(lambda: self._active_workers.discard(worker))
        worker.start()

    def _accept_preview(
        self, result: tuple[int, np.ndarray, str, Any, Any]
    ) -> None:
        generation, warped, model_name, inverse_map, forward_map = result
        if generation != self._generation:
            return
        self.registered_image.data = warped
        self._display_inverse_map = inverse_map
        self._display_forward_map = forward_map
        if self._view_mode in ("moving_raw", "moving_registered"):
            self.show_moving()
        elif self._view_mode == "overlay":
            self._resync_layers_from_pairs()
        self._set_preview_pending(False)
        self.status.setText(f"Preview updated: {model_name}, generation {generation}.")

    def _preview_failed(self, error: Exception, generation: int) -> None:
        if generation != self._generation:
            return
        self._set_preview_pending(False)
        self.status.setText(f"Preview update failed: {error}")

    def _change_marker_size(self, value: float) -> None:
        """Apply one precise display size to both landmark families."""
        self.fixed_points.size = value
        self.moving_points.size = value
        center_size = max(0.5, value / 4.0)
        self.fixed_centers.size = center_size
        self.moving_centers.size = center_size
        for callback in tuple(self.session_changed_callbacks):
            callback()

    def start_next_pair(self, *_: Any) -> None:
        pair_id = self._next_id()
        self._pending_pair_id = pair_id
        self._fast_phase = "fixed"
        self.fixed_points.mode = "select"
        self.moving_points.mode = "select"
        self.status.setText(
            f"Pair {pair_id}: left click=FIXED, right click=MOVING, in any view."
        )
        self._show_canvas_message(
            f"Pair {pair_id}\nWaiting for left-click FIXED", "#ff5555"
        )

    def delete_last_pair(self, *_: Any) -> None:
        """Delete the latest pair regardless of view, layer, or selection."""
        if self.pairs:
            self._remove_pairs_and_advance({max(self.pairs)})
            return
        if self._pending_pair_id is not None:
            pending = self._pending_pair_id
            self._pending_pair_id = None
            self._fast_phase = "idle"
            self.status.setText(
                f"Empty pair {pending} canceled. The table is empty."
            )
            return
        self.status.setText("No pair to delete.")

    def _remove_pairs_and_advance(self, pair_ids: set[int]) -> None:
        """Delete pair records without leaving empty rows."""
        for pair_id in pair_ids:
            self.pairs.pop(pair_id, None)
            if self._pending_pair_id == pair_id:
                self._pending_pair_id = None
        self._resync_layers_from_pairs()
        self._schedule_preview()
        deleted = ", ".join(str(pair_id) for pair_id in sorted(pair_ids))
        self.status.setText(
            f"Pair(s) {deleted} deleted. Continue with red left click, then green right click."
        )

    def show_fixed(self, *_: Any) -> None:
        self._view_mode = "fixed"
        if self._sample_layer is not None:
            self._sample_layer.visible = False
        if self._sample_reference_layer is not None:
            self._sample_reference_layer.visible = False
        self.fixed_image.visible = True
        self.moving_image.visible = False
        self.registered_image.visible = False
        self.fixed_points.visible = True
        self.moving_points.visible = False
        self.fixed_centers.visible = True
        self.moving_centers.visible = False
        self.viewer.layers.selection.active = self.fixed_points

    def show_moving(self, *_: Any) -> None:
        # The individual green view uses exactly the same FIXED-grid raster as
        # the overlay, including georeferencing before the first landmark.
        self._view_mode = "moving_registered"
        if self._sample_layer is not None:
            self._sample_layer.visible = False
        if self._sample_reference_layer is not None:
            self._sample_reference_layer.visible = False
        self.fixed_image.visible = False
        self.moving_image.visible = False
        self.registered_image.visible = True
        self.fixed_points.visible = False
        self.moving_points.visible = True
        self.fixed_centers.visible = False
        self.moving_centers.visible = True
        self._resync_layers_from_pairs()
        self.viewer.layers.selection.active = self.moving_points

    def show_overlay(self, *_: Any) -> None:
        self._view_mode = "overlay"
        if self._sample_layer is not None:
            self._sample_layer.visible = False
        if self._sample_reference_layer is not None:
            self._sample_reference_layer.visible = False
        self.fixed_image.visible = True
        self.moving_image.visible = False
        self.registered_image.visible = True
        self.fixed_points.visible = True
        self.moving_points.visible = True
        self.fixed_centers.visible = True
        self.moving_centers.visible = True
        self._resync_layers_from_pairs()
        self.viewer.layers.selection.active = self.fixed_points

    def _change_fixed_channel(self, index: int) -> None:
        self.fixed_channel = index
        self.fixed_preview = self.fixed_source.read_preview(index, self.preview_factor)
        self.fixed_image.data = self.fixed_preview
        self._update_contrast()
        self._resync_layers_from_pairs()
        self._schedule_preview(force=True)

    def _change_moving_channel(self, index: int) -> None:
        self.moving_channel = index
        self.moving_preview = self.moving_source.read_preview(index, self.preview_factor)
        self.moving_image.data = self.moving_preview
        self._update_contrast()
        self._resync_layers_from_pairs()
        self._schedule_preview(force=True)

    def _update_contrast(self, *_: Any) -> None:
        percentiles = (float(self.low_percentile.currentText()), float(self.high_percentile.currentText()))
        self.percentiles = percentiles
        fixed_limits = self.fixed_source.sample_contrast(self.fixed_channel, percentiles)
        moving_limits = self.moving_source.sample_contrast(self.moving_channel, percentiles)
        self.fixed_image.contrast_limits = fixed_limits
        self.moving_image.contrast_limits = moving_limits
        self.registered_image.contrast_limits = moving_limits

    def _refresh_table(self) -> None:
        from qtpy.QtCore import Qt

        self.table.blockSignals(True)
        try:
            ordered = [self.pairs[key] for key in sorted(self.pairs)]
            self.table.setRowCount(len(ordered))
            for row, pair in enumerate(ordered):
                values: list[Any] = [pair.pair_id]
                values.extend(pair.fixed_yx or (None, None))
                values.extend(pair.moving_yx or (None, None))
                for column, value in enumerate(values):
                    text = "—" if value is None else (str(value) if column == 0 else f"{value:.2f}")
                    item = self._QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, column, item)
                enabled = self._QTableWidgetItem()
                enabled.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                enabled.setCheckState(Qt.Checked if pair.enabled else Qt.Unchecked)
                enabled.setData(Qt.UserRole, pair.pair_id)
                self.table.setItem(row, 5, enabled)
        finally:
            self.table.blockSignals(False)
        self.table.scrollToBottom()

    def _on_table_item_changed(self, item: Any) -> None:
        if item.column() != 5:
            return
        from qtpy.QtCore import Qt

        pair_id = int(item.data(Qt.UserRole))
        self.pairs[pair_id].enabled = item.checkState() == Qt.Checked
        self._schedule_preview()

    def _resync_layers_from_pairs(self) -> None:
        self._suspend_events = True
        try:
            for side, layer in (("fixed", self.fixed_points), ("moving", self.moving_points)):
                attr = "fixed_yx" if side == "fixed" else "moving_yx"
                entries = [(pid, getattr(pair, attr)) for pid, pair in sorted(self.pairs.items()) if getattr(pair, attr) is not None]
                ids = [entry[0] for entry in entries]
                preview_points = [
                    full_yx_to_preview_yx(
                        point, self._full_shape(side), self._preview_shape(side)
                    )
                    for _, point in entries
                ]
                if side == "moving" and self._view_mode in (
                    "moving_registered",
                    "overlay",
                ) and preview_points:
                    if self._display_forward_map is not None:
                        raw_xy = yx_to_xy(np.asarray(preview_points, dtype=np.float64))
                        displayed_xy = self._display_forward_map(raw_xy)
                        preview_points = displayed_xy[:, ::-1].tolist()
                    else:
                        preview_points = [
                            self.moving_source.map_preview_point_to(
                                self.fixed_source,
                                point,
                                self.moving_preview.shape,
                                self.fixed_preview.shape,
                            )
                            for point in preview_points
                        ]
                data = np.asarray(preview_points, dtype=np.float64).reshape((-1, 2))
                layer.text.visible = False
                generation = self._text_generations.get(id(layer), 0) + 1
                self._text_generations[id(layer)] = generation
                layer.data = data
                layer.features = {"pair_id": np.asarray(ids, dtype=int)}
                layer.refresh()
                self._defer_point_text_restore(layer, generation)
                self._layer_ids[side] = ids
                centers = self.fixed_centers if side == "fixed" else self.moving_centers
                centers.data = data.copy()
        finally:
            self._suspend_events = False
        self._refresh_table()

    def _defer_point_text_restore(
        self, layer: Any, generation: int, attempt: int = 0
    ) -> None:
        """Restore labels after napari's asynchronous slice becomes coherent."""
        self._QTimer.singleShot(
            20,
            lambda: self._restore_point_text(layer, generation, attempt),
        )

    def _restore_point_text(
        self, layer: Any, generation: int, attempt: int
    ) -> None:
        if self._text_generations.get(id(layer)) != generation:
            return
        point_count = len(layer.data)
        if point_count == 0:
            layer.text.visible = False
            return
        feature_count = len(layer.features)
        view_indices = np.asarray(layer._view_indices, dtype=int)
        coherent = (
            feature_count == point_count
            and (view_indices.size == 0 or int(view_indices.max()) < point_count)
        )
        if not coherent:
            if attempt < 20:
                self._defer_point_text_restore(layer, generation, attempt + 1)
            return
        layer.text.refresh(layer.features)
        layer.text.visible = True
        layer.refresh()

    def session_dict(self) -> dict[str, Any]:
        return {
            "format": "napari-tps-register",
            "version": 1,
            "fixed": asdict(self.fixed_source.info),
            "moving": asdict(self.moving_source.info),
            "preview_factor": self.preview_factor,
            "preview_shapes_yx": {
                "fixed": list(self.fixed_preview.shape),
                "moving": list(self.moving_preview.shape),
            },
            "display_channels_zero_based": {
                "fixed": self.fixed_channel,
                "moving": self.moving_channel,
            },
            "contrast_percentiles": list(self.percentiles),
            "marker_size": float(self.marker_size.value()),
            "landmarks_full_resolution_yx": [asdict(self.pairs[key]) for key in sorted(self.pairs)],
        }

    def load_image_pair(
        self,
        fixed_path: Path,
        moving_path: Path,
        session: Optional[dict[str, Any]] = None,
    ) -> None:
        """Switch the active registration edge without recreating the viewer."""
        fixed_source = RasterioSource(fixed_path)
        moving_source = RasterioSource(moving_path)
        self._generation += 1
        self.fixed_source = fixed_source
        self.moving_source = moving_source
        self.fixed_channel = 0
        self.moving_channel = 0
        self._display_inverse_map = None
        self._display_forward_map = None
        self._scheduled_preview_signature = ()
        self._pending_pair_id = None
        self._fast_phase = "idle"
        self._export_result = None
        self._sample_selecting = False
        self.sample_button.setEnabled(False)
        if self._sample_layer is not None:
            self._sample_layer.visible = False
        if self._sample_reference_layer is not None:
            self._sample_reference_layer.visible = False

        self.fixed_combo.blockSignals(True)
        self.moving_combo.blockSignals(True)
        try:
            self.fixed_combo.clear()
            self.fixed_combo.addItems(
                [
                    f"FIXED B{index + 1}"
                    for index in range(fixed_source.info.shape_cyx[0])
                ]
            )
            self.moving_combo.clear()
            self.moving_combo.addItems(
                [
                    f"MOVING B{index + 1}"
                    for index in range(moving_source.info.shape_cyx[0])
                ]
            )
            if session:
                channels = session.get("display_channels_zero_based", {})
                self.fixed_channel = min(
                    int(channels.get("fixed", 0)),
                    fixed_source.info.shape_cyx[0] - 1,
                )
                self.moving_channel = min(
                    int(channels.get("moving", 0)),
                    moving_source.info.shape_cyx[0] - 1,
                )
            self.fixed_combo.setCurrentIndex(self.fixed_channel)
            self.moving_combo.setCurrentIndex(self.moving_channel)
        finally:
            self.fixed_combo.blockSignals(False)
            self.moving_combo.blockSignals(False)

        self.fixed_preview = fixed_source.read_preview(
            self.fixed_channel, self.preview_factor
        )
        self.moving_preview = moving_source.read_preview(
            self.moving_channel, self.preview_factor
        )
        self.fixed_image.data = self.fixed_preview
        self.moving_image.data = self.moving_preview
        self.registered_image.data = moving_source.read_preview_on_grid(
            self.moving_channel, self.preview_factor, fixed_source
        )
        self.fixed_image.name = f"FIXED — {fixed_path.name}"
        self.moving_image.name = f"Raw MOVING — {moving_path.name}"
        self.registered_image.name = f"Registered MOVING — {moving_path.name}"
        self._update_contrast()

        self.pairs = {}
        self._next_pair_id = 1
        if session:
            self.load_session_payload(session, verify_paths=True)
        else:
            self._resync_layers_from_pairs()
            self._schedule_preview()
        self.show_overlay()
        self.viewer.reset_view()
        self.status.setText(
            f"Active edge: FIXED {fixed_path.name} → MOVING {moving_path.name}"
        )

    def load_session_payload(
        self, payload: dict[str, Any], verify_paths: bool = True
    ) -> None:
        """Restore full-resolution landmarks and display settings for one edge."""
        if verify_paths and (
            Path(payload["fixed"]["path"]).resolve() != self.fixed_source.path
            or Path(payload["moving"]["path"]).resolve()
            != self.moving_source.path
        ):
            raise ValueError("The JSON file references different images")
        pairs: dict[int, LandmarkPair] = {}
        for raw in payload.get("landmarks_full_resolution_yx", []):
            fixed = (
                tuple(raw["fixed_yx"]) if raw.get("fixed_yx") is not None else None
            )
            moving = (
                tuple(raw["moving_yx"])
                if raw.get("moving_yx") is not None
                else None
            )
            pair = LandmarkPair(
                int(raw["pair_id"]),
                fixed,
                moving,
                bool(raw.get("enabled", True)),
            )
            pairs[pair.pair_id] = pair
        self.pairs = pairs
        self.marker_size.setValue(float(payload.get("marker_size", 3.0)))
        self._next_pair_id = max(pairs, default=0) + 1
        self._pending_pair_id = None
        self._resync_layers_from_pairs()
        self._schedule_preview()

    def save_state(self, *_: Any) -> None:
        """Save all series edges and landmarks to the automatic OUTPUT file."""
        self.series_widget.save_state()

    def load_json_dialog(self, *_: Any) -> None:
        filename, _ = self._QFileDialog.getOpenFileName(self.widget, "Load landmarks", "", "JSON (*.json)")
        if not filename:
            return
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        try:
            self.load_session_payload(payload, verify_paths=True)
        except ValueError:
            self._QMessageBox.warning(self.widget, "Different images", "The JSON file references different images; loading was refused.")
            return
        self.status.setText(f"Landmarks loaded: {filename}")

    def export_dialog(self, *_: Any) -> None:
        from napari.qt.threading import create_worker

        complete = complete_enabled_pairs(list(self.pairs.values()))
        series_mode = (
            self.series_widget.directory is not None
            and bool(self.series_widget.images)
        )
        if not series_mode and len(complete) < TPS_MIN_PAIRS:
            self._QMessageBox.warning(self.widget, "TPS unavailable", f"At least {TPS_MIN_PAIRS} complete enabled pairs are required.")
            return
        if series_mode:
            reference_path = self._choose_series_reference()
            if reference_path is None:
                return
        else:
            reference_path = self.fixed_source.path
        suggested_directory = (
            self.series_widget.state_path.parent
            if self.series_widget.state_path is not None
            else self.moving_source.path.parent / "OUTPUT"
        )
        directory = self._QFileDialog.getExistingDirectory(
            self.widget,
            "Select registration output directory",
            str(suggested_directory),
        )
        if not directory:
            return
        output_directory = Path(directory)
        if series_mode:
            project = self.series_widget.project_dict()
            worker = create_worker(
                export_registration_series,
                project,
                reference_path,
                output_directory,
            )
        else:
            fixed_yx = np.asarray(
                [pair.fixed_yx for pair in complete], dtype=np.float64
            )
            moving_yx = np.asarray(
                [pair.moving_yx for pair in complete], dtype=np.float64
            )
            worker = create_worker(
                export_registration_pair_bands,
                self.fixed_source.path,
                self.moving_source.path,
                output_directory,
                fixed_yx,
                moving_yx,
            )
        self._active_workers.add(worker)
        self._export_result = None
        self.sample_button.setEnabled(False)
        self.status.setText("Full-resolution TPS export in progress, band by band…")
        worker.returned.connect(self._export_finished)
        worker.errored.connect(lambda error: self._QMessageBox.critical(self.widget, "Export failed", str(error)))
        worker.finished.connect(lambda: self._active_workers.discard(worker))
        worker.start()

    def _choose_series_reference(self) -> Optional[Path]:
        """Ask which series image defines the output geometry."""
        from qtpy.QtWidgets import (
            QButtonGroup,
            QDialog,
            QDialogButtonBox,
            QLabel,
            QRadioButton,
            QVBoxLayout,
        )

        dialog = QDialog(self.widget)
        dialog.setWindowTitle("Choose the series reference image")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        prompt = QLabel(
            "Select the image whose geometry and pixel grid will be used for "
            "the complete exported series:"
        )
        prompt.setWordWrap(True)
        layout.addWidget(prompt)
        group = QButtonGroup(dialog)
        selected_button = None
        for index, image in enumerate(self.series_widget.images):
            button = QRadioButton(
                f"{image.scene} — {image.modality} — {image.path.name}"
            )
            button.setProperty("image_path", str(image.path))
            group.addButton(button)
            layout.addWidget(button)
            if image.path == self.fixed_source.path or (
                selected_button is None and index == 0
            ):
                selected_button = button
        if selected_button is not None:
            selected_button.setChecked(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        checked = group.checkedButton()
        return None if checked is None else Path(checked.property("image_path"))

    def _export_finished(self, result: dict[str, Any]) -> None:
        self._export_result = result
        self.sample_button.setEnabled(True)
        output_count = sum(len(paths) for paths in result["outputs"].values())
        warning = ""
        if result.get("unreachable"):
            warning = f" {len(result['unreachable'])} image(s) were unreachable."
        self.status.setText(
            f"Export complete: {output_count} band file(s) in "
            f"{Path(result['output_directory']) / 'BANDS'}.{warning}"
        )

    def _begin_sample_selection(self, *_: Any) -> None:
        if self._export_result is None:
            return
        reference_path = Path(self._export_result["reference_path"])
        reference_outputs = self._export_result["outputs"].get(
            str(reference_path.resolve()), []
        )
        if not reference_outputs:
            self._QMessageBox.critical(
                self.widget,
                "Sample creation failed",
                "The exported reference image has no first-band file.",
            )
            return
        reference_source = RasterioSource(Path(reference_outputs[0]))
        reference_preview = reference_source.read_preview(0, self.preview_factor)
        self._sample_reference_full_shape = reference_source.info.shape_cyx[1:]
        self._sample_reference_preview_shape = reference_preview.shape
        self._sample_selecting = True
        for layer in (
            self.fixed_image,
            self.moving_image,
            self.registered_image,
            self.fixed_points,
            self.moving_points,
            self.fixed_centers,
            self.moving_centers,
        ):
            layer.visible = False
        if self._sample_layer is not None:
            self._sample_layer.visible = False
        if self._sample_reference_layer is not None:
            self.viewer.layers.remove(self._sample_reference_layer)
        self._sample_reference_layer = self.viewer.add_image(
            reference_preview,
            name=f"SAMPLE REFERENCE — {reference_path.name}",
            colormap="gray",
        )
        self.viewer.reset_view()
        self.status.setText(
            "Sample selection enabled: left-click the center of the 512×512 crop."
        )
        self._show_canvas_message(
            "SAMPLE SELECTION\nLeft-click the crop center", "#66ccff"
        )

    def _create_sample_at_display_position(self, display_yx: np.ndarray) -> None:
        from napari.qt.threading import create_worker

        result = self._export_result
        if result is None:
            self._sample_selecting = False
            return
        if (
            self._sample_reference_full_shape is None
            or self._sample_reference_preview_shape is None
        ):
            self._sample_selecting = False
            self._QMessageBox.critical(
                self.widget,
                "Sample creation failed",
                "The selected reference grid is unavailable.",
            )
            return
        center_yx = preview_yx_to_full_yx(
            display_yx,
            self._sample_reference_full_shape,
            self._sample_reference_preview_shape,
        )
        first_band_paths = [
            Path(paths[0])
            for paths in result["outputs"].values()
            if paths
        ]
        output_path = Path(result["output_directory"]) / "SAMPLE_RESULT.tif"
        self._sample_selecting = False
        self.canvas_message.hide()
        self.sample_button.setEnabled(False)
        self.status.setText("Creating 512×512 sample stack…")
        worker = create_worker(
            create_sample_result,
            output_path,
            first_band_paths,
            center_yx,
            512,
        )
        self._active_workers.add(worker)
        worker.returned.connect(self._sample_finished)
        worker.errored.connect(self._sample_failed)
        worker.finished.connect(lambda: self._active_workers.discard(worker))
        worker.start()

    def _sample_finished(
        self, result: tuple[str, np.ndarray, list[str]]
    ) -> None:
        path, stack, names = result
        if "SAMPLE_RESULT" in self.viewer.layers:
            self.viewer.layers.remove("SAMPLE_RESULT")
        for layer in (
            self.fixed_image,
            self.moving_image,
            self.registered_image,
            self.fixed_points,
            self.moving_points,
            self.fixed_centers,
            self.moving_centers,
        ):
            layer.visible = False
        if self._sample_reference_layer is not None:
            self._sample_reference_layer.visible = False
        self._sample_layer = self.viewer.add_image(
            stack,
            name="SAMPLE_RESULT",
            metadata={"source_images": names, "path": path},
        )
        self.viewer.reset_view()
        self.sample_button.setEnabled(True)
        self.status.setText(
            f"Sample stack saved and displayed: {path} ({len(names)} images)."
        )

    def _sample_failed(self, error: Exception) -> None:
        self.sample_button.setEnabled(True)
        self._QMessageBox.critical(self.widget, "Sample creation failed", str(error))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, help="FIXED GeoTIFF")
    parser.add_argument("--moving", type=Path, help="MOVING GeoTIFF")
    parser.add_argument(
        "--dir",
        type=Path,
        dest="series_dir",
        help="Directory containing a series named <scene>_<MS|IR>.tif",
    )
    parser.add_argument("--preview-factor", type=int, default=4)
    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preview_factor < 1:
        raise SystemExit("--preview-factor must be greater than or equal to 1")
    if args.series_dir is not None and (args.fixed is not None or args.moving is not None):
        raise SystemExit("Use either --dir or --fixed with --moving, not both")
    if args.series_dir is None and (args.fixed is None or args.moving is None):
        raise SystemExit("Provide --dir, or provide both --fixed and --moving")

    if args.series_dir is not None:
        from series_registration import discover_series

        if not args.series_dir.is_dir():
            raise SystemExit(f"Directory not found: {args.series_dir}")
        images, warnings = discover_series(args.series_dir)
        if len(images) < 2:
            details = f" ({'; '.join(warnings)})" if warnings else ""
            raise SystemExit(
                f"At least two recognized MS/IR GeoTIFF files are required in --dir{details}"
            )
        preferred = sorted(
            (image for image in images if image.modality == "MS"),
            key=lambda image: image.scene,
        )
        initial = preferred if len(preferred) >= 2 else sorted(
            images, key=lambda image: (image.scene, image.modality)
        )
        fixed_path, moving_path = initial[0].path, initial[1].path
    else:
        fixed_path, moving_path = args.fixed, args.moving

    import napari

    viewer = napari.Viewer(title="Manual multispectral registration")
    controller = RegistrationController(
        viewer,
        RasterioSource(fixed_path),
        RasterioSource(moving_path),
        args.preview_factor,
        (args.low_percentile, args.high_percentile),
    )
    if args.series_dir is not None:
        controller.series_widget.load_directory(args.series_dir)
    viewer.window.add_dock_widget(controller.widget, name="Landmark registration", area="right")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
