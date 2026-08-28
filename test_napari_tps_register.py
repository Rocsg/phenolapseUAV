"""Pure-function regression tests for the manual registration prototype."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from napari_tps_register import (
    ComposedInverseMap,
    ImageInfo,
    LandmarkPair,
    RasterioSource,
    RegistrationController,
    _write_registered_band_files,
    complete_enabled_pairs,
    create_sample_result,
    estimate_inverse_map,
    preview_yx_to_full_yx,
    robust_contrast_limits,
    transform_name,
    yx_to_xy,
)


class CoordinateTests(unittest.TestCase):
    def test_yx_to_xy(self) -> None:
        result = yx_to_xy(np.array([[10.0, 20.0], [30.0, 40.0]]))
        np.testing.assert_array_equal(result, [[20.0, 10.0], [40.0, 30.0]])

    def test_preview_to_full_uses_each_axis_scale(self) -> None:
        result = preview_yx_to_full_yx((10, 20), (800, 1200), (200, 200))
        self.assertEqual(result, (40.0, 120.0))

    def test_georeferenced_preview_mapping_round_trip_with_different_pixel_sizes(self) -> None:
        from affine import Affine

        fixed = RasterioSource.__new__(RasterioSource)
        fixed._info = ImageInfo("fixed.tif", (6, 100, 120), "uint16", 0)
        fixed.transform = Affine.translation(500, 1000) * Affine.scale(1, -1)
        fixed.crs = None
        moving = RasterioSource.__new__(RasterioSource)
        moving._info = ImageInfo("moving_ir.tif", (1, 50, 60), "uint16", 0)
        moving.transform = Affine.translation(500, 1000) * Affine.scale(2, -2)
        moving.crs = None

        fixed_point = (20.0, 30.0)
        moving_point = fixed.map_preview_point_to(
            moving, fixed_point, (50, 60), (25, 30)
        )
        round_trip = moving.map_preview_point_to(
            fixed, moving_point, (25, 30), (50, 60)
        )
        np.testing.assert_allclose(round_trip, fixed_point)

    def test_composed_inverse_map_preserves_path_order(self) -> None:
        first = lambda points: points + np.array([2.0, 3.0])
        second = lambda points: points * 4.0
        mapping = ComposedInverseMap((first, second))
        np.testing.assert_allclose(mapping([[1.0, 2.0]]), [[12.0, 20.0]])


class ContrastTests(unittest.TestCase):
    def test_integer_sentinels_are_excluded(self) -> None:
        values = np.array([0, 10, 20, 30, np.iinfo(np.uint16).max], dtype=np.uint16)
        self.assertEqual(robust_contrast_limits(values, (0, 100)), (10.0, 30.0))

    def test_normalized_float_one_is_excluded(self) -> None:
        values = np.array([0.0, 0.2, 0.8, 1.0, np.nan, np.inf], dtype=np.float32)
        low, high = robust_contrast_limits(values, (0, 100))
        self.assertAlmostEqual(low, 0.2)
        self.assertAlmostEqual(high, 0.8)


class LandmarkTests(unittest.TestCase):
    def test_incomplete_and_disabled_pairs_are_not_used(self) -> None:
        pairs = [
            LandmarkPair(1, (1, 2), (3, 4)),
            LandmarkPair(2, (5, 6), None),
            LandmarkPair(3, (7, 8), (9, 10), enabled=False),
        ]
        self.assertEqual([pair.pair_id for pair in complete_enabled_pairs(pairs)], [1])

    def test_incomplete_fixed_point_does_not_change_preview_signature(self) -> None:
        controller = RegistrationController.__new__(RegistrationController)
        controller.pairs = {}
        before = controller._preview_signature()
        controller.pairs[1] = LandmarkPair(1, fixed_yx=(10.0, 20.0))
        self.assertEqual(controller._preview_signature(), before)

        controller.pairs[1].moving_yx = (12.0, 24.0)
        self.assertNotEqual(controller._preview_signature(), before)

    def test_model_thresholds(self) -> None:
        expected = {
            0: "georeference",
            1: "georeference + residual translation",
            3: "georeference + residual translation",
            4: "georeference + residual similarity",
            6: "georeference + residual similarity",
            7: "georeference + similarity + residual affine",
            9: "georeference + similarity + residual affine",
            10: "georeference + similarity + affine + residual TPS",
        }
        for count, name in expected.items():
            with self.subTest(count=count):
                self.assertEqual(transform_name(count), name)

    def test_translation_maps_fixed_output_to_moving_input(self) -> None:
        fixed_xy = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        moving_xy = fixed_xy + np.array([7, -4])
        inverse_map, name = estimate_inverse_map(fixed_xy, moving_xy)
        self.assertEqual(name, "residual translation")
        np.testing.assert_allclose(inverse_map(fixed_xy), moving_xy)

    def test_similarity_corrects_georeferencing_residual(self) -> None:
        fixed_xy = np.array(
            [[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float
        )
        georeference = lambda points: points + np.array([20.0, -10.0])
        angle = np.deg2rad(3.0)
        matrix = 1.02 * np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        georeferenced = georeference(fixed_xy)
        moving_xy = georeferenced @ matrix.T + np.array([4.0, 7.0])

        inverse_map, name = estimate_inverse_map(
            fixed_xy, moving_xy, georeference
        )

        self.assertEqual(name, "georeference + residual similarity")
        np.testing.assert_allclose(inverse_map(fixed_xy), moving_xy, atol=1e-8)

    def test_tps_is_fitted_after_similarity_and_affine_residuals(self) -> None:
        yy, xx = np.mgrid[0:3, 0:4]
        fixed_xy = np.column_stack([xx.ravel() * 50.0, yy.ravel() * 50.0])
        moving_xy = fixed_xy + np.column_stack(
            [5.0 + 0.002 * fixed_xy[:, 1] ** 2, -3.0 + 0.001 * fixed_xy[:, 0] ** 2]
        )

        inverse_map, name = estimate_inverse_map(fixed_xy, moving_xy)

        self.assertEqual(name, "similarity + affine + residual TPS")
        np.testing.assert_allclose(inverse_map(fixed_xy), moving_xy, atol=1e-5)


class ExportTests(unittest.TestCase):
    @staticmethod
    def _write_raster(path: Path, data: np.ndarray) -> None:
        import rasterio
        from affine import Affine

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[2],
            height=data.shape[1],
            count=data.shape[0],
            dtype=data.dtype,
            transform=Affine.translation(100, 200) * Affine.scale(1, -1),
            crs="EPSG:32631",
        ) as destination:
            destination.write(data)

    def test_band_export_writes_independent_single_band_rasters(self) -> None:
        import rasterio

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            data = np.stack(
                [
                    np.full((32, 32), 11, dtype=np.uint16),
                    np.full((32, 32), 22, dtype=np.uint16),
                ]
            )
            self._write_raster(source, data)
            outputs = _write_registered_band_files(
                source, source, root / "BANDS", None
            )

            self.assertEqual(len(outputs), 2)
            with rasterio.open(outputs[0]) as exported:
                self.assertEqual(exported.count, 1)
                self.assertEqual(exported.read(1).ndim, 2)
                np.testing.assert_array_equal(exported.read(1), data[0])

    def test_sample_result_is_one_band_per_series_image(self) -> None:
        import rasterio

        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first_B01_registered.tif"
            second = root / "second_B01_registered.tif"
            self._write_raster(first, np.full((1, 20, 20), 3, dtype=np.uint16))
            self._write_raster(second, np.full((1, 20, 20), 7, dtype=np.uint16))

            output, stack, names = create_sample_result(
                root / "SAMPLE_RESULT.tif", [first, second], (2, 3), size=8
            )

            self.assertEqual(stack.shape, (2, 8, 8))
            self.assertEqual(names, [first.stem, second.stem])
            with rasterio.open(output) as sample:
                self.assertEqual((sample.count, sample.height, sample.width), (2, 8, 8))
                self.assertEqual(sample.descriptions, tuple(names))
                self.assertTrue(np.any(sample.read(1) == 3))
                self.assertTrue(np.any(sample.read(2) == 7))


if __name__ == "__main__":
    unittest.main()
