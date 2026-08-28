"""Regression tests for shapefile ROI statistics and patch helpers."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from roi_analysis import (
    ROIRegion,
    analyze_exported_rois,
    geometry_window,
    load_shapefile_regions,
    polygon_mask,
    shapefile_fields,
    summarize_values,
    unique_shapefile_fields,
    valid_polygon_values,
)


class ROIStatisticsTests(unittest.TestCase):
    def test_population_statistics_and_raw_mad(self) -> None:
        result = summarize_values(np.array([0, 2, 4, 100], dtype=np.uint16))

        self.assertEqual(result["valid_pixel_count"], 4)
        self.assertEqual(result["minimum"], 0.0)
        self.assertEqual(result["maximum"], 100.0)
        self.assertEqual(result["median"], 3.0)
        self.assertEqual(result["median_absolute_deviation"], 2.0)
        self.assertAlmostEqual(
            result["standard_deviation"],
            float(np.std([0, 2, 4, 100], ddof=0)),
        )

    def test_validity_keeps_zero_and_excludes_mask_nodata_and_nonfinite(self) -> None:
        values = np.array([[0.0, -1.0], [np.nan, np.inf]], dtype=np.float32)
        raster_mask = np.array([[255, 255], [255, 255]], dtype=np.uint8)
        inside = np.ones((2, 2), dtype=bool)

        selected = valid_polygon_values(values, raster_mask, inside, -1.0)

        np.testing.assert_array_equal(selected, [0.0])


class ROIMaskTests(unittest.TestCase):
    def test_center_rule_preserves_polygon_hole(self) -> None:
        import rasterio
        from affine import Affine
        from shapely.geometry import Polygon

        geometry = Polygon(
            [(0, 0), (5, 0), (5, 5), (0, 5)],
            holes=[[(2, 2), (3, 2), (3, 3), (2, 3)]],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "grid.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=5,
                height=5,
                count=1,
                dtype="uint8",
                transform=Affine.translation(0, 5) * Affine.scale(1, -1),
            ) as destination:
                destination.write(np.ones((5, 5), dtype=np.uint8), 1)
            with rasterio.open(path) as source:
                window = geometry_window(source, geometry, margin=0)
                mask = polygon_mask(source, geometry, window)

        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(int(mask.sum()), 24)
        self.assertFalse(mask[2, 2])


class ROIShapefileTests(unittest.TestCase):
    def test_shapefile_requires_crs(self) -> None:
        import fiona
        from shapely.geometry import box, mapping

        with TemporaryDirectory() as directory:
            path = Path(directory) / "no_crs.shp"
            with fiona.open(
                path,
                "w",
                driver="ESRI Shapefile",
                schema={"geometry": "Polygon", "properties": {"id": "str"}},
            ) as destination:
                destination.write(
                    {"geometry": mapping(box(0, 0, 1, 1)), "properties": {"id": "A"}}
                )

            with self.assertRaisesRegex(ValueError, "no declared CRS"):
                shapefile_fields(path)

    def test_shapefile_is_reprojected_and_attributes_are_selected(self) -> None:
        import fiona
        from shapely.geometry import box, mapping

        with TemporaryDirectory() as directory:
            path = Path(directory) / "plots.shp"
            with fiona.open(
                path,
                "w",
                driver="ESRI Shapefile",
                crs="EPSG:4326",
                schema={
                    "geometry": "Polygon",
                    "properties": {"plot_id": "str", "name": "str"},
                },
            ) as destination:
                destination.write(
                    {
                        "geometry": mapping(box(0, 0, 1, 1)),
                        "properties": {"plot_id": "P1", "name": "North"},
                    }
                )

            regions = load_shapefile_regions(
                path, "EPSG:3857", "plot_id", "name"
            )

            self.assertEqual(regions[0].region_id, "P1")
            self.assertEqual(regions[0].region_name, "North")
            self.assertGreater(regions[0].geometry.bounds[2], 100_000)

    def test_only_distinct_nonempty_attributes_are_offered_as_ids(self) -> None:
        import fiona
        from shapely.geometry import box, mapping

        with TemporaryDirectory() as directory:
            path = Path(directory) / "plots.shp"
            with fiona.open(
                path,
                "w",
                driver="ESRI Shapefile",
                crs="EPSG:4326",
                schema={
                    "geometry": "Polygon",
                    "properties": {"plot_id": "str", "group": "str"},
                },
            ) as destination:
                for index in range(2):
                    destination.write(
                        {
                            "geometry": mapping(box(index, 0, index + 1, 1)),
                            "properties": {
                                "plot_id": f"P{index + 1}",
                                "group": "fgydhdgyhhgck",
                            },
                        }
                    )

            self.assertEqual(unique_shapefile_fields(path), ["plot_id"])
            with self.assertRaisesRegex(ValueError, "not unique"):
                load_shapefile_regions(path, "EPSG:4326", "group", "group")


class ROIOutputTests(unittest.TestCase):
    def test_analysis_writes_tsv_stripped_patches_mask_and_rgb(self) -> None:
        import rasterio
        from affine import Affine
        from shapely.geometry import box
        from tifffile import TiffFile

        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "scene_MS.tif"
            registered = []
            transform = Affine.translation(0, 6) * Affine.scale(1, -1)
            for band_number in range(1, 4):
                source_data = np.arange(36, dtype=np.uint16).reshape(6, 6)
                source_data += band_number * 100
                band_path = root / f"scene_MS_B{band_number:02d}.tif"
                with rasterio.open(
                    band_path,
                    "w",
                    driver="GTiff",
                    width=6,
                    height=6,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:32631",
                    transform=transform,
                    tiled=False,
                    compress="deflate",
                ) as destination:
                    destination.write(source_data, 1)
                registered.append(str(band_path))
            with rasterio.open(
                original,
                "w",
                driver="GTiff",
                width=6,
                height=6,
                count=3,
                dtype="uint16",
                crs="EPSG:32631",
                transform=transform,
            ) as destination:
                destination.descriptions = ("Blue", "Green", "Red")
                for band_number in range(1, 4):
                    destination.write(
                        np.full((6, 6), band_number, dtype=np.uint16), band_number
                    )
            result = {
                "output_directory": str(root),
                "reference_path": str(original.resolve()),
                "resolution": "FULL",
                "outputs": {str(original.resolve()): registered},
            }
            region = ROIRegion(
                "plot 1", "Plot one", box(1, 1, 5, 5), {"id": "plot 1"}
            )
            progress = []

            analysis = analyze_exported_rois(
                result,
                [region],
                root / "plots.shp",
                margin=1,
                progress_callback=progress.append,
            )

            self.assertIsNotNone(analysis.statistics_path)
            self.assertEqual(progress[0]["phase"], "preparing")
            self.assertEqual(progress[0]["completed"], 0)
            self.assertEqual(progress[-1]["phase"], "complete")
            self.assertEqual(progress[-1]["completed"], progress[-1]["total"])
            self.assertIn("mask", {item["phase"] for item in progress})
            self.assertIn("band", {item["phase"] for item in progress})
            self.assertIn("rgb", {item["phase"] for item in progress})
            self.assertIn("tsv", {item["phase"] for item in progress})
            self.assertEqual(
                analysis.output_directory.parent.name,
                "DATA_FOR_SHAPEFILE_plots",
            )
            with analysis.statistics_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["region_id"], "plot 1")
            self.assertEqual(rows[0]["sensor"], "MS")
            self.assertEqual(rows[0]["band_description"], "Blue")
            patch_directory = analysis.output_directory / "PATCHES" / "plot_1"
            with TiffFile(patch_directory / "scene_MS_B01.tif") as tif:
                self.assertEqual(len(tif.pages), 1)
                self.assertFalse(tif.pages[0].is_tiled)
            with rasterio.open(patch_directory / "roi_mask.tif") as mask_source:
                self.assertEqual(set(np.unique(mask_source.read(1))), {0, 255})
            with rasterio.open(patch_directory / "scene_MS_RGB.tif") as rgb_source:
                self.assertEqual(rgb_source.count, 3)
                self.assertEqual(rgb_source.dtypes, ("uint8", "uint8", "uint8"))
            for calibrated_path in (
                patch_directory / "scene_MS_B01.tif",
                patch_directory / "roi_mask.tif",
                patch_directory / "scene_MS_RGB.tif",
            ):
                with TiffFile(calibrated_path) as tif:
                    self.assertIn("XResolution", tif.pages.first.tags)
                    self.assertIn("YResolution", tif.pages.first.tags)
                    self.assertEqual(tif.imagej_metadata["unit"], "cm")
                    x_resolution = tif.pages.first.tags["XResolution"].value
                    y_resolution = tif.pages.first.tags["YResolution"].value
                    self.assertEqual(
                        (
                            x_resolution[0] / x_resolution[1],
                            y_resolution[0] / y_resolution[1],
                        ),
                        (0.01, 0.01),
                    )


if __name__ == "__main__":
    unittest.main()
