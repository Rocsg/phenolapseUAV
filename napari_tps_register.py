#!/usr/bin/env python3
"""Manual landmark registration for large GeoTIFF files in napari.

The interactive preview is downsampled. Full-resolution data are read only
during export and are warped separately, one spectral band at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

SIMILARITY_MIN_PAIRS = 4
AFFINE_MIN_PAIRS = 7
TPS_MIN_PAIRS = 10
DEFAULT_PERCENTILES = (0.5, 99.5)
EXPORT_MANIFEST_FILENAME = "phenolapseuav_export.json"
EXPORT_POINTER_FILENAME = ".phenolapseuav_last_export.json"
REGISTERED_IMAGES_DIRECTORY = "REGISTERED_IMAGES"
EXPORT_SUBSAMPLE_FACTORS = (1, 2, 4)
DEFAULT_EXPORT_WORKERS = 4
DEFAULT_EXPORT_SUBSAMPLE = 4
CROP_PADDING_PIXELS = 5


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


class ExportCancelled(RuntimeError):
    """Raised when a full-resolution export is cancelled cooperatively."""


@dataclass(frozen=True)
class BandExportTask:
    reference_path: Path
    source_path: Path
    band_index: int
    band_count: int
    bands_directory: Path
    inverse_map: Optional[Callable[[np.ndarray], np.ndarray]]
    display_range: Optional[tuple[float, float]] = None
    subsample_factor: int = 1
    output_bbox: Optional[tuple[int, int, int, int]] = None
    source_bbox: Optional[tuple[int, int, int, int]] = None


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
        if factor < 1:
            raise ValueError("Preview factor must be greater than or equal to 1")
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


def useful_data_bbox(
    path: Path,
    padding: int = CROP_PADDING_PIXELS,
    cancel_event: Optional[threading.Event] = None,
) -> tuple[int, int, int, int]:
    """Return padded (row start, row stop, column start, column stop) data bounds."""
    import rasterio

    if padding < 0:
        raise ValueError("Crop padding must be greater than or equal to zero")
    row_min: Optional[int] = None
    row_max = 0
    column_min: Optional[int] = None
    column_max = 0
    with rasterio.open(path) as source:
        for _, window in source.block_windows(1):
            if cancel_event is not None and cancel_event.is_set():
                raise ExportCancelled("Export cancelled")
            valid = np.zeros((int(window.height), int(window.width)), dtype=bool)
            masks = source.read_masks(window=window)
            for band_index, dtype_name in enumerate(source.dtypes):
                values = source.read(band_index + 1, window=window)
                band_valid = masks[band_index] > 0
                if np.issubdtype(values.dtype, np.floating):
                    band_valid &= np.isfinite(values)
                dtype = np.dtype(dtype_name)
                if np.issubdtype(dtype, np.integer):
                    limits = np.iinfo(dtype)
                    band_valid &= values != limits.min
                    band_valid &= values != limits.max
                else:
                    limits = np.finfo(dtype)
                    band_valid &= values != limits.min
                    band_valid &= values != limits.max
                nodata = source.nodatavals[band_index]
                if nodata is not None:
                    if np.isnan(nodata):
                        band_valid &= ~np.isnan(values)
                    else:
                        band_valid &= values != nodata
                valid |= band_valid
            rows, columns = np.nonzero(valid)
            if rows.size:
                first_row = int(window.row_off) + int(rows.min())
                last_row = int(window.row_off) + int(rows.max()) + 1
                first_column = int(window.col_off) + int(columns.min())
                last_column = int(window.col_off) + int(columns.max()) + 1
                row_min = first_row if row_min is None else min(row_min, first_row)
                row_max = max(row_max, last_row)
                column_min = (
                    first_column
                    if column_min is None
                    else min(column_min, first_column)
                )
                column_max = max(column_max, last_column)
        if row_min is None or column_min is None:
            raise ValueError(f"No valid raster data found in {Path(path).name}")
        return (
            max(0, row_min - padding),
            min(source.height, row_max + padding),
            max(0, column_min - padding),
            min(source.width, column_max + padding),
        )


def transformed_source_bbox(
    output_bbox: tuple[int, int, int, int],
    inverse_map: Optional[Callable[[np.ndarray], np.ndarray]],
    source_data_bbox: tuple[int, int, int, int],
    margin: int = CROP_PADDING_PIXELS,
    samples_per_axis: int = 65,
) -> tuple[int, int, int, int]:
    """Intersect a dense transformed FIXED ROI with one source's data crop."""
    if inverse_map is None:
        return output_bbox
    row_start, row_stop, column_start, column_stop = output_bbox
    ys = np.linspace(row_start, max(row_start, row_stop - 1), samples_per_axis)
    xs = np.linspace(
        column_start, max(column_start, column_stop - 1), samples_per_axis
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    mapped = np.asarray(
        inverse_map(np.column_stack([grid_x.ravel(), grid_y.ravel()])),
        dtype=np.float64,
    )
    mapped = mapped[np.isfinite(mapped).all(axis=1)]
    if not mapped.size:
        return (0, 0, 0, 0)
    mapped_column_start = int(np.floor(mapped[:, 0].min())) - margin
    mapped_column_stop = int(np.ceil(mapped[:, 0].max())) + 1 + margin
    mapped_row_start = int(np.floor(mapped[:, 1].min())) - margin
    mapped_row_stop = int(np.ceil(mapped[:, 1].max())) + 1 + margin
    data_row_start, data_row_stop, data_column_start, data_column_stop = (
        source_data_bbox
    )
    result = (
        max(data_row_start, mapped_row_start),
        min(data_row_stop, mapped_row_stop),
        max(data_column_start, mapped_column_start),
        min(data_column_stop, mapped_column_stop),
    )
    if result[0] >= result[1] or result[2] >= result[3]:
        return (0, 0, 0, 0)
    return result


def _resolution_directory_name(subsample_factor: int) -> str:
    if subsample_factor not in EXPORT_SUBSAMPLE_FACTORS:
        raise ValueError("Export subsampling factor must be 1, 2, or 4")
    return "FULL" if subsample_factor == 1 else f"SUBSAMPLE_{subsample_factor}"


def _imagej_spatial_calibration(
    path: Path,
) -> Optional[tuple[float, float, str, int]]:
    """Return pixels/unit X/Y, ImageJ unit, and TIFF resolution-unit code."""
    import rasterio
    from pyproj import CRS

    with rasterio.open(path) as source:
        if source.crs is None:
            return None
        transform = source.transform
        pixel_width = float(np.hypot(transform.a, transform.d))
        pixel_height = float(np.hypot(transform.b, transform.e))
        crs = CRS.from_user_input(source.crs)
    if pixel_width <= 0 or pixel_height <= 0:
        return None
    if crs.is_geographic:
        # Geographic affine coordinates are angular; TIFF has no angular
        # ResolutionUnit code, but ImageJ honors unit=degree in its description.
        return 1.0 / pixel_width, 1.0 / pixel_height, "degree", 1
    axis_info = crs.axis_info
    if not axis_info or axis_info[0].unit_conversion_factor is None:
        return None
    metres_per_crs_unit = float(axis_info[0].unit_conversion_factor)
    pixel_width_cm = pixel_width * metres_per_crs_unit * 100.0
    pixel_height_cm = pixel_height * metres_per_crs_unit * 100.0
    if pixel_width_cm <= 0 or pixel_height_cm <= 0:
        return None
    return 1.0 / pixel_width_cm, 1.0 / pixel_height_cm, "cm", 3


def _write_imagej_display_ranges(
    path: Path, ranges: Sequence[float]
) -> None:
    """Add Fiji display and spatial calibration without changing pixels."""
    from fractions import Fraction

    from tifffile import imagej_metadata_tag

    values = tuple(float(value) for value in ranges)
    if not values or len(values) % 2:
        raise ValueError("ImageJ display ranges require min/max pairs")
    calibration = _imagej_spatial_calibration(path)
    with path.open("r+b") as stream:
        header = stream.read(16)
        if header[:2] == b"II":
            byteorder = "<"
        elif header[:2] == b"MM":
            byteorder = ">"
        else:
            raise ValueError(f"Not a TIFF file: {path}")
        version = struct.unpack(f"{byteorder}H", header[2:4])[0]
        if version == 42:
            count_format, offset_format = "H", "I"
            count_size, entry_size, offset_size = 2, 12, 4
            first_ifd_offset = struct.unpack(f"{byteorder}I", header[4:8])[0]
            header_pointer_offset = 4
        elif version == 43:
            offset_bytes, reserved = struct.unpack(
                f"{byteorder}HH", header[4:8]
            )
            if offset_bytes != 8 or reserved != 0:
                raise ValueError(f"Unsupported BigTIFF header: {path}")
            count_format, offset_format = "Q", "Q"
            count_size, entry_size, offset_size = 8, 20, 8
            first_ifd_offset = struct.unpack(f"{byteorder}Q", header[8:16])[0]
            header_pointer_offset = 8
        else:
            raise ValueError(f"Unsupported TIFF version {version}: {path}")

        stream.seek(first_ifd_offset)
        entry_count = struct.unpack(
            f"{byteorder}{count_format}", stream.read(count_size)
        )[0]
        entries = [stream.read(entry_size) for _ in range(entry_count)]
        next_ifd = stream.read(offset_size)
        entries = [
            entry
            for entry in entries
            if struct.unpack(f"{byteorder}H", entry[:2])[0]
            not in (
                (270, 282, 283, 296, 50838, 50839)
                if calibration is not None
                else (270, 50838, 50839)
            )
        ]

        stream.seek(0, os.SEEK_END)
        new_entries: list[bytes] = []
        description_lines = [
            "ImageJ=1.54p",
            "images=1",
            "hyperstack=false",
        ]
        if calibration is not None:
            description_lines.append(f"unit={calibration[2]}")
        description_lines.extend(
            (f"min={values[0]:.17g}", f"max={values[1]:.17g}")
        )
        description = ("\n".join(description_lines) + "\n").encode(
            "ascii"
        ) + b"\0"
        padding = (-stream.tell()) % 8
        if padding:
            stream.write(b"\0" * padding)
        description_offset = stream.tell()
        stream.write(description)
        if version == 42:
            if description_offset > 0xFFFFFFFF:
                raise OverflowError("ImageJ description exceeds classic TIFF offsets")
            new_entries.append(
                struct.pack(
                    f"{byteorder}HHII",
                    270,
                    2,
                    len(description),
                    description_offset,
                )
            )
        else:
            new_entries.append(
                struct.pack(
                    f"{byteorder}HHQQ",
                    270,
                    2,
                    len(description),
                    description_offset,
                )
            )
        for code, dtype, count, data, _ in imagej_metadata_tag(
            {"Ranges": values}, byteorder
        ):
            padding = (-stream.tell()) % 8
            if padding:
                stream.write(b"\0" * padding)
            data_offset = stream.tell()
            stream.write(data)
            if version == 42:
                if data_offset > 0xFFFFFFFF:
                    raise OverflowError("ImageJ metadata exceeds classic TIFF offsets")
                new_entries.append(
                    struct.pack(
                        f"{byteorder}HHII", code, dtype, count, data_offset
                    )
                )
            else:
                new_entries.append(
                    struct.pack(
                        f"{byteorder}HHQQ", code, dtype, count, data_offset
                    )
                )

        if calibration is not None:
            x_resolution, y_resolution, _, resolution_unit = calibration
            for code, resolution in (
                (282, x_resolution),
                (283, y_resolution),
            ):
                fraction = Fraction(resolution).limit_denominator(1_000_000_000)
                numerator = fraction.numerator
                denominator = fraction.denominator
                if not (0 < numerator <= 0xFFFFFFFF and denominator <= 0xFFFFFFFF):
                    raise OverflowError("ImageJ pixel resolution exceeds TIFF RATIONAL")
                rational = struct.pack(
                    f"{byteorder}II", numerator, denominator
                )
                if version == 42:
                    padding = (-stream.tell()) % 8
                    if padding:
                        stream.write(b"\0" * padding)
                    rational_offset = stream.tell()
                    stream.write(rational)
                    if rational_offset > 0xFFFFFFFF:
                        raise OverflowError(
                            "ImageJ resolution exceeds classic TIFF offsets"
                        )
                    new_entries.append(
                        struct.pack(
                            f"{byteorder}HHII",
                            code,
                            5,
                            1,
                            rational_offset,
                        )
                    )
                else:
                    new_entries.append(
                        struct.pack(f"{byteorder}HHQ", code, 5, 1) + rational
                    )
            if version == 42:
                unit_value = struct.pack(
                    f"{byteorder}H", resolution_unit
                ) + b"\0\0"
                new_entries.append(
                    struct.pack(f"{byteorder}HHI", 296, 3, 1) + unit_value
                )
            else:
                unit_value = struct.pack(
                    f"{byteorder}H", resolution_unit
                ) + b"\0" * 6
                new_entries.append(
                    struct.pack(f"{byteorder}HHQ", 296, 3, 1) + unit_value
                )

        padding = (-stream.tell()) % 8
        if padding:
            stream.write(b"\0" * padding)
        new_ifd_offset = stream.tell()
        all_entries = sorted(
            entries + new_entries,
            key=lambda entry: struct.unpack(f"{byteorder}H", entry[:2])[0],
        )
        stream.write(struct.pack(f"{byteorder}{count_format}", len(all_entries)))
        for entry in all_entries:
            stream.write(entry)
        stream.write(next_ifd)
        stream.flush()
        os.fsync(stream.fileno())

        # Activate the complete replacement IFD last. Pixel offsets and bytes
        # referenced by its copied entries are unchanged.
        stream.seek(header_pointer_offset)
        stream.write(struct.pack(f"{byteorder}{offset_format}", new_ifd_offset))
        stream.flush()
        os.fsync(stream.fileno())


def _read_imagej_display_ranges(path: Path) -> Optional[tuple[float, ...]]:
    """Read Fiji display hints, if present, without loading raster pixels."""
    from tifffile import TiffFile

    with TiffFile(path) as tif:
        metadata = tif.pages.first.tags.valueof(50839) or {}
        ranges = metadata.get("Ranges")
    if ranges is None:
        return None
    return tuple(float(value) for value in ranges)


def _registration_signature(payload: dict[str, Any]) -> str:
    """Return a deterministic fingerprint of export-relevant state."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pair_registration_signature(
    fixed_path: Path,
    moving_path: Path,
    pairs: Sequence[LandmarkPair],
) -> str:
    return _registration_signature(
        {
            "kind": "pair",
            "reference_path": str(fixed_path.resolve()),
            "moving_path": str(moving_path.resolve()),
            "landmarks": [
                {
                    "pair_id": pair.pair_id,
                    "fixed_yx": pair.fixed_yx,
                    "moving_yx": pair.moving_yx,
                }
                for pair in complete_enabled_pairs(pairs)
            ],
        }
    )


def _series_registration_signature(
    project: dict[str, Any], reference_path: Path
) -> str:
    registrations = []
    for item in project.get("registrations", []):
        session = item.get("session", {})
        landmarks = [
            {
                "pair_id": int(pair["pair_id"]),
                "fixed_yx": pair["fixed_yx"],
                "moving_yx": pair["moving_yx"],
            }
            for pair in session.get("landmarks_full_resolution_yx", [])
            if bool(pair.get("enabled", True))
            and pair.get("fixed_yx") is not None
            and pair.get("moving_yx") is not None
        ]
        registrations.append(
            {
                "fixed_path": str(Path(item["fixed_path"]).resolve()),
                "moving_path": str(Path(item["moving_path"]).resolve()),
                "landmarks": landmarks,
            }
        )
    registrations.sort(
        key=lambda item: (item["fixed_path"], item["moving_path"])
    )
    return _registration_signature(
        {
            "kind": "series",
            "reference_path": str(reference_path.resolve()),
            "images": sorted(
                str(Path(item["path"]).resolve())
                for item in project.get("images", [])
            ),
            "registrations": registrations,
        }
    )


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
        if (
            self.source_crs is not None
            and self.target_crs is not None
            and self.source_crs != self.target_crs
        ):
            from pyproj import Transformer

            self._crs_transformer = Transformer.from_crs(
                self.source_crs, self.target_crs, always_xy=True
            )

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
            assert self._crs_transformer is not None
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
    contrast_percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
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
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
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
    ranges = [
        value
        for channel in range(moving_source.info.shape_cyx[0])
        for value in moving_source.sample_contrast(channel, contrast_percentiles)
    ]
    _write_imagej_display_ranges(output_path, ranges)
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


def _band_output_path(task: BandExportTask) -> Path:
    return (
        task.bands_directory
        / f"{task.source_path.stem}_B{task.band_index:02d}_registered.tif"
    )


def _source_display_ranges(
    source_path: Path,
    percentiles: tuple[float, float],
) -> list[tuple[float, float]]:
    source = RasterioSource(source_path)
    return [
        source.sample_contrast(channel, percentiles)
        for channel in range(source.info.shape_cyx[0])
    ]


def _write_registered_band_task(
    task: BandExportTask,
    cancel_event: threading.Event,
    block_completed: Callable[[BandExportTask], None],
) -> tuple[Path, Path]:
    """Compute one true 2-D band into a temporary GeoTIFF."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from skimage.transform import warp

    if cancel_event.is_set():
        raise ExportCancelled("Export cancelled")
    final_path = _band_output_path(task)
    temporary_path = final_path.with_suffix(final_path.suffix + ".partial")
    task.bands_directory.mkdir(parents=True, exist_ok=True)
    with rasterio.open(task.reference_path) as reference, rasterio.open(
        task.source_path
    ) as source:
        if task.inverse_map is None and task.output_bbox is None and (
            source.height != reference.height or source.width != reference.width
        ):
            raise ValueError(
                "The reference image dimensions changed during series export"
            )
        factor = int(task.subsample_factor)
        if factor not in EXPORT_SUBSAMPLE_FACTORS:
            raise ValueError("Export subsampling factor must be 1, 2, or 4")
        output_bbox = task.output_bbox or (0, reference.height, 0, reference.width)
        row_start, row_stop, column_start, column_stop = output_bbox
        output_height = max(1, int(np.ceil((row_stop - row_start) / factor)))
        output_width = max(1, int(np.ceil((column_stop - column_start) / factor)))
        source_bbox = task.source_bbox or (0, source.height, 0, source.width)
        source_row_start, source_row_stop, source_column_start, source_column_stop = (
            source_bbox
        )
        source_crop_empty = (
            source_row_start >= source_row_stop
            or source_column_start >= source_column_stop
        )
        source_crop_height = source_row_stop - source_row_start
        source_crop_width = source_column_stop - source_column_start
        downsampled_source_height = max(
            1, int(np.ceil(source_crop_height / factor))
        )
        downsampled_source_width = max(
            1, int(np.ceil(source_crop_width / factor))
        )
        source_scale_y = source_crop_height / downsampled_source_height
        source_scale_x = source_crop_width / downsampled_source_width
        dtype = np.dtype(source.dtypes[task.band_index - 1])
        fill_value = 0 if source.nodata is None else source.nodata
        profile = reference.profile.copy()
        profile.update(
            count=1,
            dtype=dtype.name,
            nodata=source.nodata,
            compress="deflate",
            tiled=False,
            height=output_height,
            width=output_width,
            transform=(
                reference.transform
                * Affine.translation(column_start, row_start)
                * Affine.scale(factor, factor)
            ),
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        raw = None
        if task.inverse_map is not None and not source_crop_empty:
            raw = source.read(
                task.band_index,
                window=Window(
                    source_column_start,
                    source_row_start,
                    source_column_stop - source_column_start,
                    source_row_stop - source_row_start,
                ),
                out_shape=(downsampled_source_height, downsampled_source_width),
                resampling=Resampling.average,
            )
        try:
            with rasterio.open(temporary_path, "w", **profile) as destination:
                for _, window in destination.block_windows(1):
                    if cancel_event.is_set():
                        raise ExportCancelled("Export cancelled")
                    if task.inverse_map is None:
                        registered = source.read(
                            task.band_index,
                            window=Window(
                                column_start + float(window.col_off) * factor,
                                row_start + float(window.row_off) * factor,
                                float(window.width) * factor,
                                float(window.height) * factor,
                            ),
                            out_shape=(int(window.height), int(window.width)),
                            resampling=Resampling.average,
                            boundless=True,
                            fill_value=fill_value,
                        )
                    elif raw is None:
                        registered = np.full(
                            (int(window.height), int(window.width)),
                            fill_value,
                            dtype=dtype,
                        )
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
                            coordinates_xy[:, 0] = (
                                column_start
                                + (
                                    coordinates_xy[:, 0]
                                    + column_offset
                                    + 0.5
                                )
                                * factor
                                - 0.5
                            )
                            coordinates_xy[:, 1] = (
                                row_start
                                + (
                                    coordinates_xy[:, 1]
                                    + row_offset
                                    + 0.5
                                )
                                * factor
                                - 0.5
                            )
                            mapped = np.asarray(
                                task.inverse_map(coordinates_xy), dtype=np.float64
                            ).copy()
                            mapped[:, 0] = (
                                mapped[:, 0] - source_column_start + 0.5
                            ) / source_scale_x - 0.5
                            mapped[:, 1] = (
                                mapped[:, 1] - source_row_start + 0.5
                            ) / source_scale_y - 0.5
                            return mapped

                        registered = warp(
                            raw,
                            inverse_map=block_inverse_map,
                            output_shape=(int(window.height), int(window.width)),
                            order=1,
                            mode="constant",
                            cval=fill_value,
                            preserve_range=True,
                        )
                        if np.issubdtype(dtype, np.integer):
                            limits = np.iinfo(dtype)
                            registered = np.rint(
                                np.clip(registered, limits.min, limits.max)
                            )
                        registered = registered.astype(dtype, copy=False)
                    destination.write(registered, 1, window=window)
                    block_completed(task)
                description = source.descriptions[task.band_index - 1]
                destination.set_band_description(
                    1,
                    description
                    or f"{task.source_path.name} band {task.band_index}",
                )
                destination.update_tags(
                    REGISTRATION=(
                        "reference_copy"
                        if task.inverse_map is None
                        else "manual_series_tps"
                    ),
                    REFERENCE_IMAGE=str(task.reference_path.resolve()),
                    SOURCE_IMAGE=str(task.source_path.resolve()),
                    SOURCE_BAND=str(task.band_index),
                    SUBSAMPLE_FACTOR=str(factor),
                    FIXED_CROP_BBOX=",".join(str(value) for value in output_bbox),
                    SOURCE_CROP_BBOX=",".join(str(value) for value in source_bbox),
                )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    if task.display_range is not None:
        _write_imagej_display_ranges(temporary_path, task.display_range)
    return temporary_path, final_path


def _assemble_multiband(
    source_path: Path,
    band_paths: Sequence[Path],
    multiband_directory: Path,
    cancel_event: threading.Event,
    display_ranges: Optional[Sequence[float]] = None,
) -> tuple[Path, Path]:
    """Assemble completed 2-D files into one all-bands GeoTIFF."""
    import rasterio

    if cancel_event.is_set():
        raise ExportCancelled("Export cancelled")
    multiband_directory.mkdir(parents=True, exist_ok=True)
    final_path = multiband_directory / f"{source_path.stem}_registered_all_bands.tif"
    temporary_path = final_path.with_suffix(final_path.suffix + ".partial")
    try:
        with rasterio.open(band_paths[0]) as first:
            profile = first.profile.copy()
            profile.update(count=len(band_paths), compress="deflate")
            if not profile.get("tiled", False):
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
        with rasterio.open(temporary_path, "w", **profile) as destination:
            for band_index, band_path in enumerate(band_paths, start=1):
                if cancel_event.is_set():
                    raise ExportCancelled("Export cancelled")
                with rasterio.open(band_path) as source:
                    for _, window in source.block_windows(1):
                        if cancel_event.is_set():
                            raise ExportCancelled("Export cancelled")
                        destination.write(
                            source.read(1, window=window), band_index, window=window
                        )
                    destination.set_band_description(
                        band_index,
                        source.descriptions[0]
                        or f"{source_path.name} band {band_index}",
                    )
                    destination.update_tags(
                        band_index,
                        SOURCE_IMAGE=str(source_path.resolve()),
                        SOURCE_BAND=str(band_index),
                    )
            destination.update_tags(SOURCE_IMAGE=str(source_path.resolve()))
        if display_ranges is not None:
            _write_imagej_display_ranges(temporary_path, display_ranges)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path, final_path


def _run_band_tasks(
    tasks: Sequence[BandExportTask],
    worker_count: int,
    cancel_event: threading.Event,
    progress_callback: Optional[Callable[[dict[str, Any]], None]],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Run independent bands concurrently, then assemble per-image stacks."""
    import rasterio

    if not tasks:
        return {}, {}
    with rasterio.open(tasks[0].reference_path) as reference:
        blocks_per_band = sum(1 for _ in reference.block_windows(1))
        reference_area = reference.height * reference.width
        output_bbox = tasks[0].output_bbox or (
            0,
            reference.height,
            0,
            reference.width,
        )
    factor = tasks[0].subsample_factor
    output_area = max(
        1, int(np.ceil((output_bbox[1] - output_bbox[0]) / factor))
    ) * max(1, int(np.ceil((output_bbox[3] - output_bbox[2]) / factor)))
    estimated_blocks_per_band = max(
        1, int(np.ceil(blocks_per_band * output_area / reference_area))
    )
    total_blocks = estimated_blocks_per_band * len(tasks)
    completed_blocks = 0
    completed_bands = 0
    active_bands = 0
    warped_completed = 0
    copied_completed = 0
    warped_total = sum(task.inverse_map is not None for task in tasks)
    copied_total = len(tasks) - warped_total
    recent_warp_durations: list[float] = []
    progress_lock = threading.Lock()

    def report_progress(task: BandExportTask, event: str) -> None:
        with progress_lock:
            snapshot = {
                "phase": "warp",
                "event": event,
                "completed": completed_bands,
                "total": len(tasks),
                "active": active_bands,
                "warped_completed": warped_completed,
                "warped_total": warped_total,
                "copied_completed": copied_completed,
                "copied_total": copied_total,
                "blocks_completed": completed_blocks,
                "blocks_total": total_blocks,
                "image": task.source_path.name,
                "band": task.band_index,
                "band_count": task.band_count,
                "task_kind": "warp" if task.inverse_map is not None else "copy",
                "subsample_factor": task.subsample_factor,
            }
            if recent_warp_durations and warped_completed < warped_total:
                average_duration = sum(recent_warp_durations) / len(
                    recent_warp_durations
                )
                remaining_waves = (warped_total - warped_completed) / worker_count
                snapshot["eta_seconds"] = average_duration * remaining_waves
        if progress_callback is not None:
            progress_callback(snapshot)

    def block_completed(task: BandExportTask) -> None:
        nonlocal completed_blocks
        with progress_lock:
            completed_blocks += 1
            current = completed_blocks
        report_interval = max(1, total_blocks // 200)
        if current % report_interval == 0:
            report_progress(task, "activity")

    def run_task(task: BandExportTask) -> tuple[Path, Path, float]:
        nonlocal active_bands
        with progress_lock:
            active_bands += 1
        report_progress(task, "started")
        started_at = monotonic()
        try:
            temporary_path, final_path = _write_registered_band_task(
                task,
                cancel_event,
                block_completed,
            )
            return temporary_path, final_path, monotonic() - started_at
        finally:
            with progress_lock:
                active_bands -= 1

    effective_worker_count = max(
        1, min(int(worker_count), len(tasks), os.cpu_count() or 1)
    )
    worker_count = effective_worker_count
    completed_files: list[tuple[Path, Path]] = []
    try:
        with ThreadPoolExecutor(max_workers=effective_worker_count) as executor:
            futures = {
                executor.submit(run_task, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    temporary_path, final_path, duration = future.result()
                    completed_files.append((temporary_path, final_path))
                    with progress_lock:
                        completed_bands += 1
                        if task.inverse_map is None:
                            copied_completed += 1
                        else:
                            warped_completed += 1
                            recent_warp_durations.append(duration)
                            del recent_warp_durations[:-10]
                    report_progress(task, "completed")
                except Exception:
                    cancel_event.set()
                    for pending in futures:
                        pending.cancel()
                    raise
        if cancel_event.is_set():
            raise ExportCancelled("Export cancelled")

        outputs: dict[str, list[str]] = {}
        multiband_outputs: dict[str, str] = {}
        completed_by_final = {
            final_path: temporary_path
            for temporary_path, final_path in completed_files
        }
        completed_multiband: list[tuple[Path, Path]] = []
        sources = list(dict.fromkeys(task.source_path.resolve() for task in tasks))
        for index, source_path in enumerate(sources, start=1):
            source_tasks = sorted(
                (task for task in tasks if task.source_path.resolve() == source_path),
                key=lambda task: task.band_index,
            )
            final_band_paths = [_band_output_path(task) for task in source_tasks]
            temporary_band_paths = [
                completed_by_final[path] for path in final_band_paths
            ]
            outputs[str(source_path)] = [str(path) for path in final_band_paths]
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "assemble",
                        "image": source_path.name,
                        "completed": index,
                        "total": len(sources),
                        "subsample_factor": tasks[0].subsample_factor,
                    }
                )
            multiband_temporary, multiband_final = _assemble_multiband(
                source_path,
                temporary_band_paths,
                tasks[0].bands_directory.parent / "MULTIBAND",
                cancel_event,
                [
                    value
                    for task in source_tasks
                    if task.display_range is not None
                    for value in task.display_range
                ]
                if all(task.display_range is not None for task in source_tasks)
                else None,
            )
            completed_multiband.append((multiband_temporary, multiband_final))
            multiband_outputs[str(source_path)] = str(multiband_final)

        if cancel_event.is_set():
            raise ExportCancelled("Export cancelled")
        # Keep the previous completed export untouched until all warped bands
        # and assembled products are ready.
        for temporary_path, final_path in completed_files + completed_multiband:
            temporary_path.replace(final_path)
        return outputs, multiband_outputs
    finally:
        for temporary_path in tasks[0].bands_directory.glob("*.partial"):
            temporary_path.unlink(missing_ok=True)
        multiband_directory = tasks[0].bands_directory.parent / "MULTIBAND"
        if multiband_directory.exists():
            for temporary_path in multiband_directory.glob("*.partial"):
                temporary_path.unlink(missing_ok=True)


def _write_registered_band_files(
    reference_path: Path,
    source_path: Path,
    bands_directory: Path,
    inverse_map: Optional[Callable[[np.ndarray], np.ndarray]],
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    progress_offset: int = 0,
    progress_total: int = 0,
    contrast_percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> list[str]:
    """Compatibility wrapper that exports one image sequentially."""
    import rasterio

    del progress_offset, progress_total
    display_ranges = _source_display_ranges(source_path, contrast_percentiles)
    with rasterio.open(source_path) as source:
        tasks = [
            BandExportTask(
                reference_path,
                source_path,
                band_index,
                source.count,
                bands_directory,
                inverse_map,
                display_ranges[band_index - 1],
            )
            for band_index in range(1, source.count + 1)
        ]
    outputs, _ = _run_band_tasks(
        tasks, 1, threading.Event(), progress_callback
    )
    return outputs[str(source_path.resolve())]


def _save_export_manifest(result: dict[str, Any]) -> Path:
    """Persist a completed export atomically for discovery in later sessions."""
    output_directory = Path(result["output_directory"])
    state_directory = output_directory / "STATE"
    state_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = state_directory / EXPORT_MANIFEST_FILENAME
    payload = {
        "format": "phenolapseuav-export",
        "version": 3,
        "result": result,
    }
    resolution = result.get(
        "resolution",
        _resolution_directory_name(int(result.get("subsample_factor", 1))),
    )
    manifest_text = json.dumps(payload, indent=2, sort_keys=True)
    for path in (
        manifest_path,
        state_directory / "EXPORTS" / f"{resolution}.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(manifest_text, encoding="utf-8")
        temporary_path.replace(path)
    return manifest_path


def _load_export_manifest_path(manifest_path: Path) -> Optional[dict[str, Any]]:
    """Load one manifest only when every recorded output still exists."""
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != "phenolapseuav-export":
            return None
        result = payload["result"]
        paths = [
            Path(path)
            for image_outputs in result["outputs"].values()
            for path in image_outputs
        ]
        paths.extend(Path(path) for path in result.get("multiband_outputs", {}).values())
        if not paths or not all(path.is_file() for path in paths):
            return None
        return result
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _load_export_manifest(output_directory: Path) -> Optional[dict[str, Any]]:
    """Return the most recent completed export for backward compatibility."""
    output_directory = Path(output_directory)
    manifest_path = output_directory / "STATE" / EXPORT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        manifest_path = output_directory / EXPORT_MANIFEST_FILENAME
    return _load_export_manifest_path(manifest_path)


def _load_resolution_export_manifests(
    output_directory: Path,
    registration_signature: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Return all complete resolution exports, optionally filtering stale ones."""
    output_directory = Path(output_directory)
    results: dict[str, dict[str, Any]] = {}
    exports_directory = output_directory / "STATE" / "EXPORTS"
    for manifest_path in sorted(exports_directory.glob("*.json")):
        result = _load_export_manifest_path(manifest_path)
        if result is None:
            continue
        if (
            registration_signature is not None
            and result.get("registration_signature") != registration_signature
        ):
            continue
        resolution = result.get("resolution")
        if resolution in {"FULL", "SUBSAMPLE_2", "SUBSAMPLE_4"}:
            results[resolution] = result
    latest = _load_export_manifest(output_directory)
    if latest is not None:
        resolution = latest.get("resolution")
        if (
            resolution in {"FULL", "SUBSAMPLE_2", "SUBSAMPLE_4"}
            and (
                registration_signature is None
                or latest.get("registration_signature") == registration_signature
            )
        ):
            results.setdefault(resolution, latest)
    return results


def _prepare_export_geometry(
    reference_path: Path,
    image_paths: Sequence[Path],
    mappings: dict[Path, Optional[Callable[[np.ndarray], np.ndarray]]],
    crop_to_data: bool,
    cancel_event: Optional[threading.Event] = None,
) -> tuple[
    tuple[int, int, int, int],
    dict[Path, tuple[int, int, int, int]],
]:
    """Compute the shared FIXED output crop and bounded per-source crops."""
    import rasterio

    reference_path = reference_path.resolve()
    with rasterio.open(reference_path) as reference:
        full_reference_bbox = (0, reference.height, 0, reference.width)
    if not crop_to_data:
        source_bboxes: dict[Path, tuple[int, int, int, int]] = {}
        for image_path in image_paths:
            resolved = image_path.resolve()
            with rasterio.open(resolved) as source:
                source_bboxes[resolved] = (0, source.height, 0, source.width)
        return full_reference_bbox, source_bboxes

    data_bboxes = {
        image_path.resolve(): useful_data_bbox(
            image_path, cancel_event=cancel_event
        )
        for image_path in image_paths
    }
    output_bbox = data_bboxes[reference_path]
    source_bboxes = {
        image_path.resolve(): transformed_source_bbox(
            output_bbox,
            mappings[image_path.resolve()],
            data_bboxes[image_path.resolve()],
        )
        for image_path in image_paths
    }
    return output_bbox, source_bboxes


def export_registration_series(
    project: dict[str, Any],
    reference_path: Path,
    output_directory: Path,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    worker_count: int = 1,
    cancel_event: Optional[threading.Event] = None,
    contrast_percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
    registration_signature: Optional[str] = None,
    subsample_factor: int = 1,
    crop_to_data: bool = True,
) -> dict[str, Any]:
    """Export every image reachable from the reference through directed edges."""
    reference_path = reference_path.resolve()
    output_directory = Path(output_directory)
    resolution_name = _resolution_directory_name(subsample_factor)
    bands_directory = (
        output_directory
        / REGISTERED_IMAGES_DIRECTORY
        / resolution_name
        / "BANDS"
    )
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

    import rasterio

    band_counts: dict[Path, int] = {}
    display_ranges: dict[Path, list[tuple[float, float]]] = {}
    for image_path in image_paths:
        with rasterio.open(image_path) as source:
            band_counts[image_path] = source.count
        display_ranges[image_path] = _source_display_ranges(
            image_path, contrast_percentiles
        )
    active_cancel_event = cancel_event or threading.Event()
    output_bbox, source_bboxes = _prepare_export_geometry(
        reference_path,
        image_paths,
        mappings,
        crop_to_data,
        active_cancel_event,
    )
    tasks = [
        BandExportTask(
            reference_path,
            image_path,
            band_index,
            band_counts[image_path],
            bands_directory,
            mappings[image_path],
            display_ranges[image_path][band_index - 1],
            subsample_factor,
            output_bbox,
            source_bboxes[image_path],
        )
        for image_path in image_paths
        for band_index in range(1, band_counts[image_path] + 1)
    ]
    outputs, multiband_outputs = _run_band_tasks(
        tasks,
        worker_count,
        active_cancel_event,
        progress_callback,
    )
    result = {
        "output_directory": str(output_directory),
        "reference_path": str(reference_path),
        "outputs": outputs,
        "multiband_outputs": multiband_outputs,
        "unreachable": [],
        "skipped_edges": skipped_edges,
        "registration_signature": registration_signature,
        "contrast_percentiles": list(contrast_percentiles),
        "subsample_factor": subsample_factor,
        "resolution": resolution_name,
        "crop_to_data": crop_to_data,
        "output_bbox": list(output_bbox),
        "source_bboxes": {
            str(path): list(bbox) for path, bbox in source_bboxes.items()
        },
    }
    _save_export_manifest(result)
    return result


def export_registration_pair_bands(
    fixed_path: Path,
    moving_path: Path,
    output_directory: Path,
    fixed_points_yx: np.ndarray,
    moving_points_yx: np.ndarray,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    worker_count: int = 1,
    cancel_event: Optional[threading.Event] = None,
    contrast_percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
    registration_signature: Optional[str] = None,
    subsample_factor: int = 1,
    crop_to_data: bool = True,
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
    output_directory = Path(output_directory)
    resolution_name = _resolution_directory_name(subsample_factor)
    bands_directory = (
        output_directory
        / REGISTERED_IMAGES_DIRECTORY
        / resolution_name
        / "BANDS"
    )
    import rasterio

    with rasterio.open(fixed_path) as reference:
        fixed_band_count = reference.count
    with rasterio.open(moving_path) as moving:
        moving_band_count = moving.count
    display_ranges = {
        fixed_path.resolve(): _source_display_ranges(
            fixed_path, contrast_percentiles
        ),
        moving_path.resolve(): _source_display_ranges(
            moving_path, contrast_percentiles
        ),
    }
    resolved_fixed = fixed_path.resolve()
    resolved_moving = moving_path.resolve()
    mappings = {resolved_fixed: None, resolved_moving: inverse_map}
    active_cancel_event = cancel_event or threading.Event()
    output_bbox, source_bboxes = _prepare_export_geometry(
        resolved_fixed,
        [resolved_fixed, resolved_moving],
        mappings,
        crop_to_data,
        active_cancel_event,
    )
    tasks = [
        BandExportTask(
            fixed_path,
            path,
            band_index,
            band_count,
            bands_directory,
            mapping,
            display_ranges[path.resolve()][band_index - 1],
            subsample_factor,
            output_bbox,
            source_bboxes[path.resolve()],
        )
        for path, band_count, mapping in (
            (fixed_path, fixed_band_count, None),
            (moving_path, moving_band_count, inverse_map),
        )
        for band_index in range(1, band_count + 1)
    ]
    outputs, multiband_outputs = _run_band_tasks(
        tasks,
        worker_count,
        active_cancel_event,
        progress_callback,
    )
    result = {
        "output_directory": str(output_directory),
        "reference_path": str(fixed_path.resolve()),
        "outputs": outputs,
        "multiband_outputs": multiband_outputs,
        "unreachable": [],
        "skipped_edges": [],
        "registration_signature": registration_signature,
        "contrast_percentiles": list(contrast_percentiles),
        "subsample_factor": subsample_factor,
        "resolution": resolution_name,
        "crop_to_data": crop_to_data,
        "output_bbox": list(output_bbox),
        "source_bboxes": {
            str(path): list(bbox) for path, bbox in source_bboxes.items()
        },
    }
    _save_export_manifest(result)
    return result


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
    display_ranges: list[float] = []
    for path in first_band_paths:
        with rasterio.open(path) as source:
            fill_value = 0 if source.nodata is None else source.nodata
            arrays.append(
                source.read(1, window=window, boundless=True, fill_value=fill_value)
            )
            names.append(Path(path).stem)
            profiles.append(source.profile.copy())
        imagej_ranges = _read_imagej_display_ranges(Path(path))
        if imagej_ranges is not None and len(imagej_ranges) >= 2:
            display_ranges.extend(imagej_ranges[:2])
        else:
            display_ranges.extend(robust_contrast_limits(arrays[-1]))

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
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
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
    _write_imagej_display_ranges(output_path, display_ranges)
    return str(output_path), stack, names


class RegistrationController:
    """Qt controller; all layer and UI writes remain on the GUI thread."""

    def __init__(
        self,
        viewer: Any,
        fixed_source: RasterioSource,
        moving_source: RasterioSource,
        preview_factor: int = 8,
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
            QSpinBox,
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
        self._export_started_at: Optional[float] = None
        self._export_progress_relay: Optional[Any] = None
        self._export_cancel_event: Optional[threading.Event] = None
        self._sample_selecting = False
        self._sample_layer: Optional[Any] = None
        self._sample_reference_layer: Optional[Any] = None
        self._sample_reference_full_shape: Optional[tuple[int, int]] = None
        self._sample_reference_preview_shape: Optional[tuple[int, int]] = None
        self._roi_exports: dict[str, dict[str, Any]] = {}
        self._roi_shapefile_path: Optional[Path] = None
        self._roi_regions: list[Any] = []
        self._roi_reference_layer: Optional[Any] = None
        self._roi_shapes_layer: Optional[Any] = None
        self._roi_labels_layer: Optional[Any] = None
        self._roi_viewing = False
        self._roi_started_at: Optional[float] = None
        self._roi_progress_relay: Optional[Any] = None
        self.session_changed_callbacks: list[Any] = []
        self.standalone_state_path: Optional[Path] = None

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

        self.update_complete_pairs_only = QCheckBox(
            "Update only after both points are placed"
        )
        self.update_complete_pairs_only.setChecked(True)
        self.update_complete_pairs_only.setToolTip(
            "When enabled, wait for both the FIXED and MOVING point before updating. "
            "Deleting a point or pair always updates the view."
        )
        layout.addWidget(self.update_complete_pairs_only)

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
        self.export_button = QPushButton("Apply / Export TPS")
        self.export_button.clicked.connect(self.export_dialog)
        persistence_row.addWidget(self.export_button)
        persistence_row.addWidget(QLabel("Workers"))
        self.export_workers = QSpinBox()
        self.export_workers.setRange(1, max(1, os.cpu_count() or 1))
        self.export_workers.setValue(
            min(DEFAULT_EXPORT_WORKERS, max(1, os.cpu_count() or 1))
        )
        self.export_workers.setToolTip(
            "Parallel band workers. Each active warped band holds its required "
            "source crop at the selected export resolution in memory."
        )
        persistence_row.addWidget(self.export_workers)
        self.cancel_export_button = QPushButton("Stop export")
        self.cancel_export_button.setEnabled(False)
        self.cancel_export_button.clicked.connect(self._cancel_export)
        persistence_row.addWidget(self.cancel_export_button)
        self.sample_button = QPushButton("Create 512×512 sample")
        self.sample_button.setEnabled(False)
        self.sample_button.setToolTip(
            "Enabled after export; click a center point on the registered series"
        )
        self.sample_button.clicked.connect(self._begin_sample_selection)
        persistence_row.addWidget(self.sample_button)
        layout.addLayout(persistence_row)

        export_options_row = QHBoxLayout()
        export_options_row.addWidget(QLabel("Export resolution"))
        self.export_resolution = QComboBox()
        self.export_resolution.addItem("Full resolution (×1)", 1)
        self.export_resolution.addItem("Subsample ×2", 2)
        self.export_resolution.addItem("Subsample ×4", 4)
        self.export_resolution.setCurrentIndex(
            self.export_resolution.findData(DEFAULT_EXPORT_SUBSAMPLE)
        )
        self.export_resolution.setToolTip(
            "Subsample source imagery before applying the registration. "
            "Each resolution is stored in its own output directory."
        )
        export_options_row.addWidget(self.export_resolution)
        self.crop_export_to_data = QCheckBox("Crop export to useful data")
        self.crop_export_to_data.setChecked(True)
        self.crop_export_to_data.setToolTip(
            "Crop the FIXED grid and every source to valid non-background data, "
            "with a 5-pixel safety margin."
        )
        export_options_row.addWidget(self.crop_export_to_data)
        export_options_row.addStretch(1)
        layout.addLayout(export_options_row)

        self.roi_group = QGroupBox("Shapefile ROI analysis")
        roi_layout = QGridLayout(self.roi_group)
        roi_layout.addWidget(QLabel("Registered resolution"), 0, 0)
        self.roi_resolution = QComboBox()
        self.roi_resolution.currentIndexChanged.connect(
            self._roi_resolution_changed
        )
        roi_layout.addWidget(self.roi_resolution, 0, 1)
        self.roi_load_button = QPushButton("Load ROI shapefile…")
        self.roi_load_button.clicked.connect(self._choose_roi_shapefile)
        roi_layout.addWidget(self.roi_load_button, 0, 2)
        roi_layout.addWidget(QLabel("Unique region ID"), 1, 0)
        self.roi_id_field = QComboBox()
        self.roi_id_field.setToolTip(
            "Only attributes containing one distinct, non-empty value per "
            "polygon are listed. Each value names one patch folder."
        )
        roi_layout.addWidget(self.roi_id_field, 1, 1)
        roi_layout.addWidget(QLabel("Region name"), 1, 2)
        self.roi_name_field = QComboBox()
        roi_layout.addWidget(self.roi_name_field, 1, 3)
        self.roi_display_button = QPushButton("Display polygons")
        self.roi_display_button.clicked.connect(self._display_roi_shapefile)
        roi_layout.addWidget(self.roi_display_button, 1, 4)
        roi_layout.addWidget(QLabel("Patch margin (pixels)"), 2, 0)
        self.roi_margin = QSpinBox()
        self.roi_margin.setRange(0, 10000)
        self.roi_margin.setValue(5)
        roi_layout.addWidget(self.roi_margin, 2, 1)
        self.roi_statistics_button = QPushButton("Export ROI statistics")
        self.roi_statistics_button.clicked.connect(
            lambda: self._start_roi_analysis(True, False)
        )
        roi_layout.addWidget(self.roi_statistics_button, 2, 2)
        self.roi_patches_button = QPushButton("Export ROI patches + masks")
        self.roi_patches_button.clicked.connect(
            lambda: self._start_roi_analysis(False, True)
        )
        roi_layout.addWidget(self.roi_patches_button, 2, 3, 1, 2)
        self.roi_group.setEnabled(False)
        layout.addWidget(self.roi_group)

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
        self._restore_existing_export()

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
        if self._roi_viewing:
            # ROI layers are display-only. Leave mouse handling to napari so
            # Space + drag and the standard pan/zoom interactions keep working.
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
        pending = self.pairs.get(self._pending_pair_id)
        if side == "moving":
            candidates = [
                pair.pair_id
                for pair in self.pairs.values()
                if pair.fixed_yx is not None and pair.moving_yx is None
            ]
            if pending is not None and pending.pair_id in candidates:
                pair_id = pending.pair_id
            elif candidates:
                pair_id = max(candidates)
            else:
                self.status.setText(
                    "Right click ignored: place a red FIXED point with left click first."
                )
                self._show_canvas_message(
                    "RIGHT CLICK IGNORED\nPlace FIXED first",
                    "#ff5555",
                )
                return
        else:
            fixed_only = [
                pair.pair_id
                for pair in self.pairs.values()
                if pair.fixed_yx is not None and pair.moving_yx is None
            ]
            missing_fixed = [
                pair.pair_id
                for pair in self.pairs.values()
                if pair.fixed_yx is None
            ]
            if pending is not None and not pending.complete:
                pair_id = pending.pair_id
            elif fixed_only:
                # A second left click moves the unpaired FIXED point instead of
                # creating another incomplete pair.
                pair_id = max(fixed_only)
            elif missing_fixed:
                pair_id = min(missing_fixed)
            else:
                pair_id = self._next_id()
                self.pairs[pair_id] = LandmarkPair(pair_id)
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
            if self.update_complete_pairs_only.isChecked():
                self._notify_session_changed()
            else:
                self._schedule_preview(force=True)
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

        deleted_ids = set(old_ids) - set(ids)
        for deleted_id in deleted_ids:
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
        self._schedule_preview(
            force=bool(deleted_ids) or not self.update_complete_pairs_only.isChecked()
        )

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

    def _export_matches_current_registration(
        self, result: dict[str, Any]
    ) -> bool:
        saved_signature = result.get("registration_signature")
        if not saved_signature:
            return False
        reference_path = Path(result["reference_path"])
        if self.series_widget.directory is not None and self.series_widget.images:
            current_signature = _series_registration_signature(
                self.series_widget.project_dict(), reference_path
            )
        else:
            current_signature = _pair_registration_signature(
                self.fixed_source.path,
                self.moving_source.path,
                list(self.pairs.values()),
            )
        return saved_signature == current_signature

    def _current_registration_signature(self, reference_path: Path) -> str:
        if self.series_widget.directory is not None and self.series_widget.images:
            return _series_registration_signature(
                self.series_widget.project_dict(), reference_path
            )
        return _pair_registration_signature(
            self.fixed_source.path,
            self.moving_source.path,
            list(self.pairs.values()),
        )

    def _schedule_preview(self, force: bool = False) -> None:
        fixed_count = len(self.fixed_points.data)
        moving_count = len(self.moving_points.data)
        complete = complete_enabled_pairs(list(self.pairs.values()))
        signature = self._preview_signature()
        self._notify_session_changed()
        if self._export_result is not None and not self._export_matches_current_registration(
            self._export_result
        ):
            self._export_result = None
            self.sample_button.setEnabled(False)
            self._roi_exports.clear()
            self.roi_resolution.clear()
            self.roi_group.setEnabled(False)
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
        """Display a non-blocking message centered at the top of the canvas."""
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
            20,
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
            if self.canvas_message.text().startswith("RELOADING PREVIEWS"):
                self.canvas_message.hide()
            if self._view_mode == "moving_registered":
                self.show_moving()
            self.status.setText(
                f"Preview updated: georeference, factor {self.preview_factor}."
            )
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
        self.status.setText(
            f"Preview updated: {model_name}, factor {self.preview_factor}, "
            f"generation {generation}."
        )

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
            self._schedule_preview(force=True)
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
        self._schedule_preview(force=True)
        deleted = ", ".join(str(pair_id) for pair_id in sorted(pair_ids))
        self.status.setText(
            f"Pair(s) {deleted} deleted. Continue with red left click, then green right click."
        )

    def show_fixed(self, *_: Any) -> None:
        self._view_mode = "fixed"
        self._roi_viewing = False
        for layer in (
            self._roi_reference_layer,
            self._roi_shapes_layer,
            self._roi_labels_layer,
        ):
            if layer is not None:
                layer.visible = False
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
        self._roi_viewing = False
        for layer in (
            self._roi_reference_layer,
            self._roi_shapes_layer,
            self._roi_labels_layer,
        ):
            if layer is not None:
                layer.visible = False
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
        self._roi_viewing = False
        for layer in (
            self._roi_reference_layer,
            self._roi_shapes_layer,
            self._roi_labels_layer,
        ):
            if layer is not None:
                layer.visible = False
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

    def set_preview_factor(self, factor: int) -> None:
        """Reload previews at a new resolution while preserving full-resolution landmarks."""
        factor = int(factor)
        if factor < 1:
            raise ValueError("Preview factor must be greater than or equal to 1")
        if factor == self.preview_factor:
            return

        old_shape = self.fixed_preview.shape
        old_center = tuple(self.viewer.camera.center)
        old_zoom = float(self.viewer.camera.zoom)
        self.status.setText(f"Reloading previews with factor {factor}…")
        self._show_canvas_message(
            f"RELOADING PREVIEWS…\nSubsampling factor {factor}",
            "#66bbff",
        )

        # Read into local variables first so a failed read leaves the current
        # previews and factor untouched.
        try:
            fixed_preview = self.fixed_source.read_preview(self.fixed_channel, factor)
            moving_preview = self.moving_source.read_preview(self.moving_channel, factor)
        except Exception:
            if self.canvas_message.text().startswith("RELOADING PREVIEWS"):
                self.canvas_message.hide()
            self.status.setText(
                f"Preview reload failed; factor {self.preview_factor} remains active."
            )
            raise

        self._generation += 1
        self._preview_timer.stop()
        self.preview_factor = factor
        self.fixed_preview = fixed_preview
        self.moving_preview = moving_preview
        self.fixed_image.data = fixed_preview
        self.moving_image.data = moving_preview
        self.registered_image.data = np.zeros(fixed_preview.shape, dtype=np.float32)
        self._display_inverse_map = None
        self._display_forward_map = None
        self._scheduled_preview_signature = ()
        self._resync_layers_from_pairs()

        # Keep approximately the same geographic field of view. Landmark
        # coordinates are rebuilt from their full-resolution values above.
        if len(old_center) >= 2 and old_shape[0] > 0 and old_shape[1] > 0:
            center = list(old_center)
            center[-2] *= fixed_preview.shape[0] / old_shape[0]
            center[-1] *= fixed_preview.shape[1] / old_shape[1]
            self.viewer.camera.center = tuple(center)
            scale_y = old_shape[0] / fixed_preview.shape[0]
            scale_x = old_shape[1] / fixed_preview.shape[1]
            self.viewer.camera.zoom = old_zoom * float(np.sqrt(scale_y * scale_x))

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
        self._restore_existing_export()

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

    def configure_standalone_state(self, state_path: Path) -> bool:
        """Restore and continuously save one standalone FIXED/MOVING pair."""
        self.standalone_state_path = state_path.resolve()
        restored = False
        state_error: Optional[Exception] = None
        if self.standalone_state_path.is_file():
            try:
                payload = json.loads(
                    self.standalone_state_path.read_text(encoding="utf-8")
                )
                self.load_session_payload(payload, verify_paths=True)
                restored = True
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                state_error = error
        if self._save_standalone_state not in self.session_changed_callbacks:
            self.session_changed_callbacks.append(self._save_standalone_state)
        if state_error is None:
            self._save_standalone_state()
            if restored:
                self.status.setText(
                    f"Standalone pair restored: {self.standalone_state_path}"
                )
            else:
                self.status.setText(
                    f"Standalone pair state initialized: {self.standalone_state_path}"
                )
        else:
            self.status.setText(
                f"Standalone state not restored: {state_error}. "
                "The existing file is preserved until the next landmark edit."
            )
        return restored

    def _save_standalone_state(self) -> bool:
        if self.standalone_state_path is None:
            return False
        try:
            self.standalone_state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.standalone_state_path.with_suffix(
                self.standalone_state_path.suffix + ".tmp"
            )
            temporary.write_text(
                json.dumps(self.session_dict(), indent=2), encoding="utf-8"
            )
            temporary.replace(self.standalone_state_path)
            return True
        except OSError as error:
            self.status.setText(f"Failed to save standalone state: {error}")
            return False

    def save_state(self, *_: Any) -> None:
        """Save either the active series project or standalone pair."""
        if self.series_widget.directory is not None:
            self.series_widget.save_state()
            return
        if self._save_standalone_state():
            self.status.setText(f"State saved: {self.standalone_state_path}")
        else:
            self.status.setText("No automatic state path is configured.")

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
        from qtpy.QtCore import QObject, Signal

        class ExportProgressRelay(QObject):
            progressed = Signal(object)

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
        output_directory = self._automatic_output_directory()
        output_directory.mkdir(parents=True, exist_ok=True)
        progress_relay = ExportProgressRelay(self.widget)
        progress_relay.progressed.connect(self._export_progress)
        self._export_progress_relay = progress_relay
        self._export_cancel_event = threading.Event()
        worker_count = self.export_workers.value()
        subsample_factor = int(self.export_resolution.currentData())
        crop_to_data = self.crop_export_to_data.isChecked()
        if series_mode:
            project = self.series_widget.project_dict()
            registration_signature = _series_registration_signature(
                project, reference_path
            )
            worker = create_worker(
                export_registration_series,
                project,
                reference_path,
                output_directory,
                progress_relay.progressed.emit,
                worker_count,
                self._export_cancel_event,
                self.percentiles,
                registration_signature,
                subsample_factor,
                crop_to_data,
            )
        else:
            fixed_yx = np.asarray(
                [pair.fixed_yx for pair in complete], dtype=np.float64
            )
            moving_yx = np.asarray(
                [pair.moving_yx for pair in complete], dtype=np.float64
            )
            registration_signature = _pair_registration_signature(
                self.fixed_source.path,
                self.moving_source.path,
                complete,
            )
            worker = create_worker(
                export_registration_pair_bands,
                self.fixed_source.path,
                self.moving_source.path,
                output_directory,
                fixed_yx,
                moving_yx,
                progress_relay.progressed.emit,
                worker_count,
                self._export_cancel_event,
                self.percentiles,
                registration_signature,
                subsample_factor,
                crop_to_data,
            )
        self._active_workers.add(worker)
        self._export_result = None
        self._export_started_at = monotonic()
        self.export_button.setEnabled(False)
        self.export_workers.setEnabled(False)
        self.export_resolution.setEnabled(False)
        self.crop_export_to_data.setEnabled(False)
        self.cancel_export_button.setEnabled(True)
        self.sample_button.setEnabled(False)
        self.status.setText(
            f"Raster export ×{subsample_factor} starting with "
            f"{worker_count} worker(s)…"
        )
        self._show_canvas_message(
            f"RASTER EXPORT ×{subsample_factor}…\n"
            "Detecting useful data and preparing files",
            "#66bbff",
        )
        worker.returned.connect(self._export_finished)
        worker.errored.connect(self._export_failed)
        worker.finished.connect(lambda: self._active_workers.discard(worker))
        worker.start()

    def _cancel_export(self, *_: Any) -> None:
        """Request cooperative cancellation after each active raster block."""
        if self._export_cancel_event is None:
            return
        self._export_cancel_event.set()
        self.cancel_export_button.setEnabled(False)
        self.status.setText(
            "Stopping export… Active workers will stop after their current block."
        )
        self._show_canvas_message(
            "STOPPING EXPORT…\nWaiting for active blocks to finish",
            "#ff9955",
        )

    def _automatic_output_directory(self) -> Path:
        """Return the deterministic sibling output directory for current inputs."""
        from series_registration import registered_output_directory

        input_directory = (
            self.series_widget.directory
            if self.series_widget.directory is not None
            else self.moving_source.path.parent
        )
        return registered_output_directory(input_directory)

    def _export_pointer_path(self) -> Path:
        return (
            self._automatic_output_directory()
            / "STATE"
            / EXPORT_POINTER_FILENAME
        )

    def _remember_export(self, result: dict[str, Any]) -> None:
        """Remember an arbitrary export directory beside the current project."""
        pointer_path = self._export_pointer_path()
        try:
            pointer_path.parent.mkdir(parents=True, exist_ok=True)
            pointer_path.write_text(
                json.dumps(
                    {"output_directory": result["output_directory"]}, indent=2
                ),
                encoding="utf-8",
            )
        except OSError:
            # The export itself remains valid and can still be discovered in
            # the default output directory.
            pass

    def _refresh_roi_exports(self, output_directory: Path) -> None:
        """Expose all complete, non-stale resolutions to the ROI selector."""
        latest = _load_export_manifest(output_directory)
        if latest is None:
            self._roi_exports = {}
        else:
            signature = self._current_registration_signature(
                Path(latest["reference_path"])
            )
            self._roi_exports = _load_resolution_export_manifests(
                output_directory, signature
            )
        previous = self.roi_resolution.currentData()
        self.roi_resolution.blockSignals(True)
        self.roi_resolution.clear()
        labels = {
            "FULL": "Full resolution (×1)",
            "SUBSAMPLE_2": "Subsample ×2",
            "SUBSAMPLE_4": "Subsample ×4",
        }
        for resolution in ("FULL", "SUBSAMPLE_2", "SUBSAMPLE_4"):
            if resolution in self._roi_exports:
                self.roi_resolution.addItem(labels[resolution], resolution)
        if previous is not None:
            index = self.roi_resolution.findData(previous)
            if index >= 0:
                self.roi_resolution.setCurrentIndex(index)
        self.roi_resolution.blockSignals(False)
        self.roi_group.setEnabled(bool(self._roi_exports))
        has_shapefile = self._roi_shapefile_path is not None
        self.roi_display_button.setEnabled(has_shapefile)
        self.roi_statistics_button.setEnabled(has_shapefile)
        self.roi_patches_button.setEnabled(has_shapefile)

    def _selected_roi_export(self) -> Optional[dict[str, Any]]:
        resolution = self.roi_resolution.currentData()
        return self._roi_exports.get(str(resolution))

    def _roi_resolution_changed(self, *_: Any) -> None:
        if self._roi_shapefile_path is not None:
            self._display_roi_shapefile()

    def _choose_roi_shapefile(self, *_: Any) -> None:
        from roi_analysis import shapefile_fields, unique_shapefile_fields

        filename, _ = self._QFileDialog.getOpenFileName(
            self.widget, "Load ROI shapefile", "", "Shapefile (*.shp)"
        )
        if not filename:
            return
        try:
            fields = shapefile_fields(Path(filename))
            if not fields:
                raise ValueError("The shapefile has no attribute fields")
            unique_fields = unique_shapefile_fields(Path(filename))
            if not unique_fields:
                raise ValueError(
                    "The shapefile has no non-empty attribute with one unique "
                    "value per polygon. Add a unique region ID field before "
                    "exporting patches."
                )
        except Exception as error:
            self._QMessageBox.critical(
                self.widget, "ROI shapefile failed", str(error)
            )
            return
        self._roi_shapefile_path = Path(filename).resolve()
        self.roi_id_field.clear()
        self.roi_id_field.addItems(unique_fields)
        self.roi_name_field.clear()
        self.roi_name_field.addItems(fields)
        preferred_id = next(
            (
                field
                for field in unique_fields
                if field.casefold() in {"id", "region_id", "plot_id"}
            ),
            unique_fields[0],
        )
        preferred_name = next(
            (
                field
                for field in fields
                if field.casefold() in {"name", "nom", "region_name"}
            ),
            fields[min(1, len(fields) - 1)],
        )
        self.roi_id_field.setCurrentText(preferred_id)
        self.roi_name_field.setCurrentText(preferred_name)
        self.roi_display_button.setEnabled(True)
        self.roi_statistics_button.setEnabled(True)
        self.roi_patches_button.setEnabled(True)
        self._display_roi_shapefile()

    def _remove_roi_layers(self) -> None:
        for attribute in (
            "_roi_reference_layer",
            "_roi_shapes_layer",
            "_roi_labels_layer",
        ):
            layer = getattr(self, attribute)
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
            setattr(self, attribute, None)

    def _display_roi_shapefile(self, *_: Any) -> None:
        import rasterio

        from roi_analysis import geometry_display_paths, load_shapefile_regions

        result = self._selected_roi_export()
        if result is None or self._roi_shapefile_path is None:
            return
        reference_key = str(Path(result["reference_path"]).resolve())
        reference_outputs = result["outputs"].get(reference_key, [])
        if not reference_outputs:
            return
        reference_path = Path(reference_outputs[0])
        try:
            with rasterio.open(reference_path) as reference:
                regions = load_shapefile_regions(
                    self._roi_shapefile_path,
                    reference.crs,
                    self.roi_id_field.currentText(),
                    self.roi_name_field.currentText(),
                )
                full_shape = (reference.height, reference.width)
                transform = reference.transform
            reference_source = RasterioSource(reference_path)
            preview = reference_source.read_preview(0, self.preview_factor)
            stored_ranges = _read_imagej_display_ranges(reference_path)
            if stored_ranges is not None and len(stored_ranges) >= 2:
                reference_limits = (
                    float(stored_ranges[0]),
                    float(stored_ranges[1]),
                )
            else:
                reference_limits = reference_source.sample_contrast(
                    0, self.percentiles
                )
            scale = (
                full_shape[0] / preview.shape[0],
                full_shape[1] / preview.shape[1],
            )
            paths, label_points, labels = geometry_display_paths(
                regions, transform, scale
            )
        except Exception as error:
            self._QMessageBox.critical(
                self.widget, "ROI display failed", str(error)
            )
            return
        self._roi_regions = regions
        self._remove_roi_layers()
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
        self._roi_reference_layer = self.viewer.add_image(
            preview,
            name=f"ROI REFERENCE — {result['resolution']}",
            colormap="gray",
            contrast_limits=reference_limits,
        )
        self._roi_shapes_layer = self.viewer.add_shapes(
            paths,
            shape_type="path",
            name="ROI outlines",
            edge_color="yellow",
            edge_width=1,
        )
        self._roi_shapes_layer.editable = False
        self._roi_labels_layer = self.viewer.add_points(
            np.asarray(label_points),
            name="ROI labels",
            size=2,
            opacity=1,
            face_color=[0.0, 0.0, 0.0, 0.0],
            border_color=[0.0, 0.0, 0.0, 0.0],
            features={"label": labels},
            text={
                "string": "{label}",
                "size": 9,
                "color": "yellow",
            },
        )
        self._roi_labels_layer.editable = False
        self.viewer.layers.selection.active = self._roi_reference_layer
        self._roi_viewing = True
        self.viewer.reset_view()
        self.status.setText(
            f"Displayed {len(regions)} ROI polygon(s) on {result['resolution']}."
        )

    def _set_roi_analysis_enabled(self, enabled: bool) -> None:
        self.roi_load_button.setEnabled(enabled)
        self.roi_display_button.setEnabled(
            enabled and self._roi_shapefile_path is not None
        )
        self.roi_statistics_button.setEnabled(
            enabled and self._roi_shapefile_path is not None
        )
        self.roi_patches_button.setEnabled(
            enabled and self._roi_shapefile_path is not None
        )

    def _start_roi_analysis(
        self, write_statistics: bool, write_patches: bool
    ) -> None:
        from napari.qt.threading import create_worker
        from qtpy.QtCore import QObject, Signal

        from roi_analysis import analyze_exported_rois, load_shapefile_regions

        class ROIProgressRelay(QObject):
            progressed = Signal(object)

        result = self._selected_roi_export()
        if result is None or self._roi_shapefile_path is None:
            return
        reference_key = str(Path(result["reference_path"]).resolve())
        reference_outputs = result["outputs"].get(reference_key, [])
        if not reference_outputs:
            return
        try:
            import rasterio

            with rasterio.open(reference_outputs[0]) as reference:
                regions = load_shapefile_regions(
                    self._roi_shapefile_path,
                    reference.crs,
                    self.roi_id_field.currentText(),
                    self.roi_name_field.currentText(),
                )
        except Exception as error:
            self._QMessageBox.critical(
                self.widget, "ROI analysis failed", str(error)
            )
            return
        progress_relay = ROIProgressRelay(self.widget)
        progress_relay.progressed.connect(self._roi_analysis_progress)
        self._roi_progress_relay = progress_relay
        worker = create_worker(
            analyze_exported_rois,
            result,
            regions,
            self._roi_shapefile_path,
            self.roi_margin.value(),
            write_statistics,
            write_patches,
            progress_relay.progressed.emit,
        )
        worker.returned.connect(self._roi_analysis_finished)
        worker.errored.connect(self._roi_analysis_failed)
        worker.finished.connect(lambda: self._active_workers.discard(worker))
        self._active_workers.add(worker)
        self._roi_started_at = monotonic()
        self._set_roi_analysis_enabled(False)
        operation = "statistics" if write_statistics else "patches and masks"
        self.status.setText(f"ROI analysis starting: {operation}…")
        self._show_canvas_message(
            f"ROI ANALYSIS — {operation.upper()}…\nPreparing regions and files",
            "#66bbff",
        )
        worker.start()

    def _roi_analysis_progress(self, progress: dict[str, Any]) -> None:
        """Render thread-safe ROI progress in the panel and on the canvas."""
        phase = str(progress.get("phase", "preparing"))
        completed = int(progress.get("completed", 0))
        total = max(1, int(progress.get("total", 1)))
        percent = min(100, round(100 * completed / total))
        elapsed = (
            0.0
            if self._roi_started_at is None
            else monotonic() - self._roi_started_at
        )
        region_index = int(progress.get("region_index", 0))
        region_count = int(progress.get("region_count", 0))
        region_id = str(progress.get("region_id", ""))
        source_image = str(progress.get("source_image", ""))
        if phase == "preparing":
            detail = "Preparing regions and output files"
        elif phase == "mask":
            detail = f"Mask complete — region {region_id}"
        elif phase == "rgb":
            detail = f"Pseudo-RGB complete — {source_image}"
        elif phase == "tsv":
            detail = "Writing roi_statistics.tsv"
        elif phase == "complete":
            detail = "Finalizing ROI analysis"
        else:
            band_number = int(progress.get("band_number", 0))
            band_count = int(progress.get("band_count", 0))
            detail = (
                f"{source_image} — band {band_number}/{band_count} complete"
            )
        region_text = (
            f"region {region_index}/{region_count} ({region_id})"
            if region_index and region_count
            else f"{region_count} region(s)"
        )
        counts = f"{completed}/{total} steps · {percent}% · {region_text}"
        timing = f"elapsed {self._format_duration(elapsed)}"
        self.status.setText(
            f"ROI analysis: {counts} — {detail} — {timing}"
        )
        self._show_canvas_message(
            f"ROI ANALYSIS — {percent}%\n{counts}\n{detail}\n{timing}",
            "#66bbff",
        )

    def _roi_analysis_finished(self, result: Any) -> None:
        elapsed = (
            0.0
            if self._roi_started_at is None
            else monotonic() - self._roi_started_at
        )
        self._roi_started_at = None
        self._roi_progress_relay = None
        self._set_roi_analysis_enabled(True)
        if self.canvas_message.text().startswith("ROI ANALYSIS"):
            self.canvas_message.hide()
        self.status.setText(
            f"ROI analysis complete: {result.region_count} region(s), "
            f"{result.patch_count} band patch(es), {result.mask_count} mask(s), "
            f"{result.rgb_count} pseudo-RGB file(s) after "
            f"{self._format_duration(elapsed)} in {result.output_directory}."
        )

    def _roi_analysis_failed(self, error: Exception) -> None:
        self._roi_started_at = None
        self._roi_progress_relay = None
        self._set_roi_analysis_enabled(True)
        if self.canvas_message.text().startswith("ROI ANALYSIS"):
            self.canvas_message.hide()
        self.status.setText(f"ROI analysis failed: {error}")
        self._QMessageBox.critical(
            self.widget, "ROI analysis failed", str(error)
        )

    def _restore_existing_export(
        self, output_directory: Optional[Path] = None
    ) -> bool:
        """Enable sample creation when a complete prior export is available."""
        candidates: list[Path] = []
        if output_directory is not None:
            candidates.append(output_directory)
        pointer_paths = [
            self._export_pointer_path(),
            self.moving_source.path.parent / EXPORT_POINTER_FILENAME,
        ]
        for pointer_path in pointer_paths:
            if not pointer_path.is_file():
                continue
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                candidates.append(Path(pointer["output_directory"]))
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass
        if self.series_widget.state_path is not None:
            candidates.append(self.series_widget.state_path.parent.parent)
        candidates.append(self._automatic_output_directory())
        # Backward compatibility with exports created before the deterministic
        # sibling <input>_REGISTERED layout.
        candidates.append(self.moving_source.path.parent / "OUTPUT")

        expected_paths = {str(self.fixed_source.path), str(self.moving_source.path)}
        if self.series_widget.images:
            expected_paths = {str(image.path.resolve()) for image in self.series_widget.images}
        stale_export_found = False
        for candidate in candidates:
            result = _load_export_manifest(candidate)
            if result is None:
                continue
            exported_paths = {str(Path(path).resolve()) for path in result["outputs"]}
            if not expected_paths.issubset(exported_paths):
                continue
            if not self._export_matches_current_registration(result):
                stale_export_found = True
                continue
            self._export_result = result
            self.sample_button.setEnabled(True)
            self._refresh_roi_exports(Path(result["output_directory"]))
            self.status.setText(
                "Existing completed export detected; 512×512 sample creation is available."
            )
            return True
        if stale_export_found:
            self._export_result = None
            self.sample_button.setEnabled(False)
            self.status.setText(
                "Existing export is obsolete because registration points changed; "
                "run a new full-resolution export."
            )
        return False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m"
        if minutes:
            return f"{minutes:d}m {seconds:02d}s"
        return f"{seconds:d}s"

    def _export_progress(self, progress: dict[str, Any]) -> None:
        """Render thread-safe export progress forwarded onto the GUI thread."""
        if progress.get("phase") == "assemble":
            factor = int(progress.get("subsample_factor", 1))
            detail = (
                f"{progress['image']} — {progress['completed']}/{progress['total']}"
            )
            self.status.setText(f"Assembling all-bands GeoTIFF: {detail}")
            self._show_canvas_message(
                f"ASSEMBLING ×{factor} MULTIBAND GEOTIFFS…\n{detail}",
                "#66bbff",
            )
            return
        completed = int(progress["completed"])
        total = max(1, int(progress["total"]))
        elapsed = (
            0.0
            if self._export_started_at is None
            else monotonic() - self._export_started_at
        )
        active = int(progress.get("active", 0))
        warped_completed = int(progress.get("warped_completed", 0))
        warped_total = int(progress.get("warped_total", 0))
        copied_completed = int(progress.get("copied_completed", 0))
        copied_total = int(progress.get("copied_total", 0))
        action = "completed" if progress.get("event") == "completed" else "working"
        detail = (
            f"{progress['image']} — band {progress['band']}/{progress['band_count']} "
            f"({action})"
        )
        counts = (
            f"{completed}/{total} bands complete · {active} active · "
            f"warped {warped_completed}/{warped_total} · "
            f"reference copies {copied_completed}/{copied_total}"
        )
        eta = progress.get("eta_seconds")
        eta_text = (
            f"ETA {self._format_duration(float(eta))}"
            if eta is not None
            else "ETA calculating after first warped band…"
        )
        timing = f"elapsed {self._format_duration(elapsed)} · {eta_text}"
        factor = int(progress.get("subsample_factor", 1))
        self.status.setText(
            f"Raster export ×{factor}: {counts} — {detail} — {timing}"
        )
        self._show_canvas_message(
            f"RASTER EXPORT ×{factor}\n{counts}\n{detail}\n{timing}",
            "#66bbff",
        )

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
        self._remember_export(result)
        self._refresh_roi_exports(Path(result["output_directory"]))
        elapsed = (
            0.0
            if self._export_started_at is None
            else monotonic() - self._export_started_at
        )
        self._export_started_at = None
        self._export_progress_relay = None
        self._export_cancel_event = None
        self.export_button.setEnabled(True)
        self.export_workers.setEnabled(True)
        self.export_resolution.setEnabled(True)
        self.crop_export_to_data.setEnabled(True)
        self.cancel_export_button.setEnabled(False)
        self.sample_button.setEnabled(True)
        if self.canvas_message.text().startswith(
            ("RASTER EXPORT", "ASSEMBLING")
        ):
            self.canvas_message.hide()
        output_count = sum(len(paths) for paths in result["outputs"].values())
        warning = ""
        if result.get("unreachable"):
            warning = f" {len(result['unreachable'])} image(s) were unreachable."
        self.status.setText(
            f"Export complete: {output_count} band file(s) in "
            f"{Path(result['output_directory']) / REGISTERED_IMAGES_DIRECTORY} "
            f"after {self._format_duration(elapsed)}.{warning}"
        )

    def _export_failed(self, error: Exception) -> None:
        cancelled = isinstance(error, ExportCancelled) or (
            self._export_cancel_event is not None
            and self._export_cancel_event.is_set()
        )
        self._export_started_at = None
        self._export_progress_relay = None
        self._export_cancel_event = None
        self.export_button.setEnabled(True)
        self.export_workers.setEnabled(True)
        self.export_resolution.setEnabled(True)
        self.crop_export_to_data.setEnabled(True)
        self.cancel_export_button.setEnabled(False)
        if self.canvas_message.text().startswith(
            ("RASTER EXPORT", "ASSEMBLING", "STOPPING EXPORT")
        ):
            self.canvas_message.hide()
        if cancelled:
            if self._restore_existing_export():
                self.status.setText(
                    "Export stopped. The previous completed export is still valid "
                    "for the current control points."
                )
            else:
                self.status.setText(
                    "Export stopped. No completed export matches the current "
                    "control points; restart export after choosing the worker count."
                )
            return
        self.status.setText(f"Raster export failed: {error}")
        self._QMessageBox.critical(self.widget, "Export failed", str(error))

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
        output_path = (
            Path(result["output_directory"]) / "PATCHES" / "SAMPLE_RESULT.tif"
        )
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
    parser.add_argument("--preview-factor", type=int, default=8)
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
