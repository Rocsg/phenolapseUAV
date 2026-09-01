"""Pure-function regression tests for the manual registration prototype."""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import napari_tps_register as registration_module

from napari_tps_register import (
    REGISTERED_IMAGES_DIRECTORY,
    BandExportTask,
    ComposedInverseMap,
    ExportCancelled,
    ImageInfo,
    LandmarkPair,
    RasterioSource,
    RegistrationController,
    _load_export_manifest,
    _load_resolution_export_manifests,
    _pair_registration_signature,
    _read_imagej_display_ranges,
    _run_band_tasks,
    _save_export_manifest,
    _write_imagej_display_ranges,
    _write_registered_band_files,
    complete_enabled_pairs,
    create_sample_result,
    estimate_inverse_map,
    export_registration_pair_bands,
    preview_yx_to_full_yx,
    robust_contrast_limits,
    transform_name,
    transformed_source_bbox,
    useful_data_bbox,
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

    def test_raster_preview_is_projected_onto_reference_grid(self) -> None:
        import rasterio
        from affine import Affine

        with TemporaryDirectory() as directory:
            root = Path(directory)
            moving_path = root / "moving.tif"
            fixed_path = root / "fixed.tif"
            for path, shape, transform in (
                (
                    moving_path,
                    (2, 2),
                    Affine.translation(500, 1000) * Affine.scale(2, -2),
                ),
                (
                    fixed_path,
                    (4, 4),
                    Affine.translation(500, 1000) * Affine.scale(1, -1),
                ),
            ):
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=shape[1],
                    height=shape[0],
                    count=1,
                    dtype="uint16",
                    transform=transform,
                ) as destination:
                    destination.write(np.full(shape, 7, dtype=np.uint16), 1)

            preview = RasterioSource(moving_path).read_preview_on_grid(
                0, 1, RasterioSource(fixed_path)
            )

            self.assertEqual(preview.shape, (4, 4))
            np.testing.assert_array_equal(preview, np.full((4, 4), 7))

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
    def _canvas_controller(self) -> RegistrationController:
        controller = RegistrationController.__new__(RegistrationController)
        controller.pairs = {}
        controller._next_pair_id = 1
        controller._pending_pair_id = None
        controller._fast_phase = "idle"
        controller._full_shape = lambda side: (100, 100)
        controller._preview_shape = lambda side: (100, 100)
        controller._resync_layers_from_pairs = Mock()
        controller._notify_session_changed = Mock()
        controller._schedule_preview = Mock()
        controller._show_canvas_message = Mock()
        controller.status = Mock()
        controller.update_complete_pairs_only = Mock()
        controller.update_complete_pairs_only.isChecked.return_value = True
        return controller

    def test_second_fixed_click_moves_unpaired_point(self) -> None:
        controller = self._canvas_controller()

        controller._record_canvas_point("fixed", (10.0, 20.0))
        controller._record_canvas_point("fixed", (30.0, 40.0))

        self.assertEqual(list(controller.pairs), [1])
        self.assertEqual(controller.pairs[1].fixed_yx, (30.0, 40.0))
        self.assertIsNone(controller.pairs[1].moving_yx)

    def test_moving_click_without_fixed_point_is_ignored(self) -> None:
        controller = self._canvas_controller()

        controller._record_canvas_point("moving", (30.0, 40.0))

        self.assertEqual(controller.pairs, {})
        controller._resync_layers_from_pairs.assert_not_called()
        controller._schedule_preview.assert_not_called()

    def test_moving_click_completes_existing_fixed_pair(self) -> None:
        controller = self._canvas_controller()

        controller._record_canvas_point("fixed", (10.0, 20.0))
        controller._record_canvas_point("moving", (30.0, 40.0))

        self.assertEqual(list(controller.pairs), [1])
        self.assertTrue(controller.pairs[1].complete)
        controller._schedule_preview.assert_called_once_with()

    def test_preview_factor_reload_preserves_landmarks_and_view(self) -> None:
        controller = RegistrationController.__new__(RegistrationController)
        controller.preview_factor = 4
        controller.fixed_channel = 0
        controller.moving_channel = 1
        controller.fixed_preview = np.zeros((50, 80))
        controller.moving_preview = np.zeros((40, 60))
        controller.fixed_source = Mock()
        controller.moving_source = Mock()
        controller.fixed_source.read_preview.return_value = np.zeros((20, 32))
        controller.moving_source.read_preview.return_value = np.zeros((16, 24))
        controller.fixed_image = SimpleNamespace(data=None)
        controller.moving_image = SimpleNamespace(data=None)
        controller.registered_image = SimpleNamespace(data=None)
        controller.viewer = SimpleNamespace(
            camera=SimpleNamespace(center=(0.0, 25.0, 40.0), zoom=2.0)
        )
        controller.status = Mock()
        controller._show_canvas_message = Mock()
        controller._preview_timer = Mock()
        controller._resync_layers_from_pairs = Mock()
        controller._schedule_preview = Mock()
        controller._display_inverse_map = object()
        controller._display_forward_map = object()
        controller._scheduled_preview_signature = ((1,),)
        controller._generation = 3

        controller.set_preview_factor(10)

        self.assertEqual(controller.preview_factor, 10)
        self.assertEqual(controller.fixed_preview.shape, (20, 32))
        self.assertEqual(controller.moving_preview.shape, (16, 24))
        controller.fixed_source.read_preview.assert_called_once_with(0, 10)
        controller.moving_source.read_preview.assert_called_once_with(1, 10)
        controller._resync_layers_from_pairs.assert_called_once_with()
        controller._schedule_preview.assert_called_once_with(force=True)
        self.assertEqual(controller.viewer.camera.center, (0.0, 10.0, 16.0))
        self.assertAlmostEqual(controller.viewer.camera.zoom, 5.0)
        self.assertIsNone(controller._display_inverse_map)
        self.assertIsNone(controller._display_forward_map)

    def test_failed_preview_factor_reload_keeps_current_factor(self) -> None:
        controller = RegistrationController.__new__(RegistrationController)
        controller.preview_factor = 4
        controller.fixed_channel = 0
        controller.moving_channel = 0
        controller.fixed_preview = np.zeros((50, 80))
        controller.moving_preview = np.zeros((40, 60))
        controller.fixed_source = Mock()
        controller.moving_source = Mock()
        controller.fixed_source.read_preview.side_effect = OSError("read failed")
        controller.viewer = SimpleNamespace(
            camera=SimpleNamespace(center=(0.0, 25.0, 40.0), zoom=2.0)
        )
        controller.status = Mock()
        controller._show_canvas_message = Mock()
        controller.canvas_message = Mock()
        controller.canvas_message.text.return_value = "RELOADING PREVIEWS…"

        with self.assertRaisesRegex(OSError, "read failed"):
            controller.set_preview_factor(10)

        self.assertEqual(controller.preview_factor, 4)
        self.assertEqual(controller.fixed_preview.shape, (50, 80))

    def test_export_progress_reports_band_counts_without_false_percentage(self) -> None:
        controller = RegistrationController.__new__(RegistrationController)
        controller._export_started_at = None
        controller.status = Mock()
        controller._show_canvas_message = Mock()

        controller._export_progress(
            {
                "phase": "warp",
                "event": "completed",
                "completed": 7,
                "total": 35,
                "active": 7,
                "warped_completed": 1,
                "warped_total": 29,
                "copied_completed": 6,
                "copied_total": 6,
                "image": "2022_09h_STR_IR.tif",
                "band": 1,
                "band_count": 1,
                "eta_seconds": 600.0,
            }
        )

        status = controller.status.setText.call_args.args[0]
        self.assertIn("7/35 bands complete", status)
        self.assertIn("warped 1/29", status)
        self.assertIn("reference copies 6/6", status)
        self.assertIn("ETA 10m 00s", status)
        self.assertNotIn("%", status)

    def test_standalone_pair_state_is_saved_and_restored_automatically(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "PAIR_REGISTERED" / "STATE" / "pair.json"
            payload = {
                "format": "napari-tps-register",
                "fixed": {"path": "/tmp/fixed.tif"},
                "moving": {"path": "/tmp/moving.tif"},
                "landmarks_full_resolution_yx": [],
            }
            state_path.parent.mkdir(parents=True)
            state_path.write_text(json.dumps(payload), encoding="utf-8")
            controller = RegistrationController.__new__(RegistrationController)
            controller.standalone_state_path = None
            controller.session_changed_callbacks = []
            controller.status = Mock()
            controller.session_dict = Mock(return_value=payload)
            controller.load_session_payload = Mock()

            restored = controller.configure_standalone_state(state_path)

            self.assertTrue(restored)
            controller.load_session_payload.assert_called_once_with(
                payload, verify_paths=True
            )
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8")), payload
            )
            self.assertIn(
                controller._save_standalone_state,
                controller.session_changed_callbacks,
            )

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
            multiband_path = root / "MULTIBAND" / "source_registered_all_bands.tif"
            with rasterio.open(multiband_path) as exported:
                self.assertEqual(exported.count, 2)
                np.testing.assert_array_equal(exported.read(), data)

    def test_useful_data_bbox_unites_bands_and_adds_padding(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.tif"
            data = np.zeros((2, 20, 20), dtype=np.uint16)
            data[0, 5:10, 6:11] = 10
            data[1, 8:14, 10:16] = 20
            self._write_raster(path, data)

            self.assertEqual(useful_data_bbox(path, padding=2), (3, 16, 4, 18))

    def test_transformed_source_bbox_never_exceeds_source_data_crop(self) -> None:
        output_bbox = (10, 50, 20, 60)
        data_bbox = (15, 45, 25, 55)

        result = transformed_source_bbox(
            output_bbox,
            lambda coordinates: coordinates,
            data_bbox,
        )

        self.assertEqual(result, data_bbox)

    def test_subsampled_crop_updates_shape_and_georeferencing(self) -> None:
        import rasterio
        from affine import Affine

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            data = np.arange(1, 65, dtype=np.uint16).reshape(1, 8, 8)
            self._write_raster(source, data)
            task = BandExportTask(
                source,
                source,
                1,
                1,
                root / "BANDS",
                None,
                subsample_factor=2,
                output_bbox=(2, 8, 2, 8),
                source_bbox=(2, 8, 2, 8),
            )

            outputs, _ = _run_band_tasks(
                [task], 1, threading.Event(), None
            )

            with rasterio.open(outputs[str(source.resolve())][0]) as exported:
                self.assertEqual(exported.shape, (3, 3))
                self.assertEqual(
                    exported.transform,
                    Affine.translation(100, 200)
                    * Affine.scale(1, -1)
                    * Affine.translation(2, 2)
                    * Affine.scale(2, 2),
                )

    def test_source_crop_writes_standard_background_outside_available_data(
        self,
    ) -> None:
        import rasterio

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            data = np.full((1, 8, 8), 9, dtype=np.uint16)
            self._write_raster(source, data)
            task = BandExportTask(
                source,
                source,
                1,
                1,
                root / "BANDS",
                lambda coordinates: coordinates,
                output_bbox=(0, 8, 0, 8),
                source_bbox=(2, 6, 2, 6),
            )

            outputs, _ = _run_band_tasks(
                [task], 1, threading.Event(), None
            )

            with rasterio.open(outputs[str(source.resolve())][0]) as exported:
                result = exported.read(1)
            expected = np.zeros((8, 8), dtype=np.uint16)
            expected[2:6, 2:6] = 9
            np.testing.assert_array_equal(result, expected)

    def test_pair_export_uses_registered_images_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixed = root / "fixed.tif"
            moving = root / "moving.tif"
            band = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) + 1
            data = np.stack([band, band + 1000])
            self._write_raster(fixed, data)
            self._write_raster(moving, data)
            yy, xx = np.mgrid[0:3, 0:4]
            points = np.column_stack(
                [yy.ravel() * 10.0 + 3.0, xx.ravel() * 8.0 + 3.0]
            )
            progress: list[dict[str, object]] = []

            result = export_registration_pair_bands(
                fixed,
                moving,
                root / "SERIES_REGISTERED",
                points,
                points,
                progress_callback=progress.append,
                worker_count=2,
            )

            expected_root = (
                root
                / "SERIES_REGISTERED"
                / REGISTERED_IMAGES_DIRECTORY
                / "FULL"
            )
            for paths in result["outputs"].values():
                self.assertTrue(
                    all(Path(path).parent == expected_root / "BANDS" for path in paths)
                )
            self.assertTrue(
                all(
                    Path(path).parent == expected_root / "MULTIBAND"
                    for path in result["multiband_outputs"].values()
                )
            )
            completed = [
                item
                for item in progress
                if item.get("phase") == "warp" and item.get("event") == "completed"
            ]
            self.assertEqual(completed[-1]["completed"], 4)
            self.assertEqual(completed[-1]["copied_completed"], 2)
            self.assertEqual(completed[-1]["warped_completed"], 2)
            self.assertTrue(
                any(
                    item["warped_completed"] == 1 and "eta_seconds" in item
                    for item in completed
                )
            )

    def test_pair_export_combines_useful_crop_and_subsampled_georeferencing(
        self,
    ) -> None:
        import rasterio
        from affine import Affine
        from tifffile import TiffFile

        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixed = root / "fixed.tif"
            moving = root / "moving.tif"
            data = np.zeros((1, 40, 40), dtype=np.uint16)
            data[:, 10:30, 10:30] = 100
            self._write_raster(fixed, data)
            self._write_raster(moving, data)
            yy, xx = np.mgrid[0:3, 0:4]
            points = np.column_stack(
                [yy.ravel() * 12.0 + 7.0, xx.ravel() * 10.0 + 5.0]
            )

            result = export_registration_pair_bands(
                fixed,
                moving,
                root / "SERIES_REGISTERED",
                points,
                points,
                worker_count=2,
                subsample_factor=2,
                crop_to_data=True,
            )

            self.assertEqual(result["resolution"], "SUBSAMPLE_2")
            self.assertEqual(result["output_bbox"], [5, 35, 5, 35])
            imagej_resolutions = []
            for paths in result["outputs"].values():
                with rasterio.open(paths[0]) as exported:
                    self.assertEqual(exported.shape, (15, 15))
                    self.assertEqual(
                        exported.transform,
                        Affine.translation(100, 200)
                        * Affine.scale(1, -1)
                        * Affine.translation(5, 5)
                        * Affine.scale(2, 2),
                    )
                with TiffFile(paths[0]) as tif:
                    x_resolution = tif.pages.first.tags["XResolution"].value
                    y_resolution = tif.pages.first.tags["YResolution"].value
                    imagej_resolutions.append(
                        (
                            x_resolution[0] / x_resolution[1],
                            y_resolution[0] / y_resolution[1],
                        )
                    )
            self.assertEqual(len(set(imagej_resolutions)), 1)
            # The 1 m FIXED pixels become 2 m (200 cm) at subsample ×2.
            self.assertEqual(imagej_resolutions[0], (0.005, 0.005))

    def test_export_adds_fiji_ranges_without_changing_pixels_or_georeference(self) -> None:
        import rasterio
        from tifffile import TiffFile

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            band1 = np.arange(1, 1025, dtype=np.uint16).reshape(32, 32)
            band2 = band1 + 1000
            data = np.stack([band1, band2])
            self._write_raster(source, data)
            with rasterio.open(source) as original:
                original_transform = original.transform
                original_crs = original.crs

            outputs = _write_registered_band_files(
                source,
                source,
                root / "BANDS",
                None,
                contrast_percentiles=(0.0, 100.0),
            )

            self.assertEqual(
                _read_imagej_display_ranges(Path(outputs[0])), (1.0, 1024.0)
            )
            self.assertEqual(
                _read_imagej_display_ranges(Path(outputs[1])),
                (1001.0, 2024.0),
            )
            with TiffFile(outputs[0]) as exported_band:
                self.assertEqual(len(exported_band.pages), 1)
                self.assertTrue(exported_band.is_imagej)
                self.assertEqual(exported_band.pages[0].shape, (32, 32))
                self.assertFalse(exported_band.pages[0].is_tiled)
                self.assertEqual(
                    tuple(exported_band.imagej_metadata["Ranges"]),
                    (1.0, 1024.0),
                )
                self.assertFalse(exported_band.imagej_metadata["hyperstack"])
                self.assertEqual(exported_band.imagej_metadata["images"], 1)
                self.assertEqual(exported_band.imagej_metadata["min"], 1.0)
                self.assertEqual(exported_band.imagej_metadata["max"], 1024.0)
            multiband = root / "MULTIBAND" / "source_registered_all_bands.tif"
            self.assertEqual(
                _read_imagej_display_ranges(multiband),
                (1.0, 1024.0, 1001.0, 2024.0),
            )
            with rasterio.open(multiband) as exported:
                np.testing.assert_array_equal(exported.read(), data)
                self.assertEqual(exported.transform, original_transform)
                self.assertEqual(exported.crs, original_crs)

    def test_imagej_calibration_uses_centimetres_and_affine_pixel_size(self) -> None:
        import rasterio
        from affine import Affine
        from tifffile import TiffFile

        with TemporaryDirectory() as directory:
            path = Path(directory) / "half_centimetre.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=8,
                height=8,
                count=1,
                dtype="uint16",
                crs="EPSG:32631",
                transform=Affine.translation(500_000, 5_000_000)
                * Affine.scale(0.005, -0.01),
            ) as destination:
                destination.write(np.ones((8, 8), dtype=np.uint16), 1)

            _write_imagej_display_ranges(path, (0.0, 1.0))

            with TiffFile(path) as tif:
                page = tif.pages.first
                x_resolution = page.tags["XResolution"].value
                y_resolution = page.tags["YResolution"].value
                self.assertAlmostEqual(
                    x_resolution[0] / x_resolution[1], 2.0
                )
                self.assertAlmostEqual(
                    y_resolution[0] / y_resolution[1], 1.0
                )
                self.assertEqual(int(page.tags["ResolutionUnit"].value), 3)
                self.assertEqual(tif.imagej_metadata["unit"], "cm")

    def test_unrepresentable_imagej_calibration_does_not_abort_export(self) -> None:
        import rasterio
        from affine import Affine
        from tifffile import TiffFile

        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny_pixels.tif"
            data = np.ones((8, 8), dtype=np.uint16)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=8,
                height=8,
                count=1,
                dtype=data.dtype,
                crs="EPSG:32631",
                transform=Affine.scale(1e-12, -1e-12),
            ) as destination:
                destination.write(data, 1)

            _write_imagej_display_ranges(path, (0.0, 1.0))

            self.assertEqual(_read_imagej_display_ranges(path), (0.0, 1.0))
            with rasterio.open(path) as exported:
                np.testing.assert_array_equal(exported.read(1), data)
                self.assertEqual(exported.transform, Affine.scale(1e-12, -1e-12))
            with TiffFile(path) as tif:
                self.assertNotIn("unit", tif.imagej_metadata)

    def test_registration_signature_changes_with_control_points(self) -> None:
        first = [LandmarkPair(1, (10.0, 20.0), (11.0, 22.0))]
        moved = [LandmarkPair(1, (10.0, 20.0), (12.0, 22.0))]

        first_signature = _pair_registration_signature(
            Path("fixed.tif"), Path("moving.tif"), first
        )
        moved_signature = _pair_registration_signature(
            Path("fixed.tif"), Path("moving.tif"), moved
        )

        self.assertNotEqual(first_signature, moved_signature)

    def test_band_export_reports_progress_through_last_band(self) -> None:
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
            progress: list[dict[str, object]] = []

            _write_registered_band_files(
                source,
                source,
                root / "BANDS",
                None,
                progress.append,
                0,
                2,
            )

            self.assertTrue(progress)
            warp_progress = [item for item in progress if item["phase"] == "warp"]
            self.assertEqual(warp_progress[-1]["completed"], 2)
            self.assertEqual(warp_progress[-1]["total"], 2)
            self.assertEqual(warp_progress[-1]["copied_completed"], 2)
            self.assertEqual(warp_progress[-1]["copied_total"], 2)
            self.assertEqual(warp_progress[-1]["warped_completed"], 0)
            self.assertEqual(warp_progress[-1]["warped_total"], 0)
            self.assertEqual(warp_progress[-1]["band_count"], 2)
            self.assertEqual(progress[-1]["phase"], "assemble")

    def test_parallel_band_tasks_keep_2d_outputs_and_assemble_afterward(self) -> None:
        import rasterio

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            data = np.stack(
                [
                    np.full((32, 32), 11, dtype=np.uint16),
                    np.full((32, 32), 22, dtype=np.uint16),
                    np.full((32, 32), 33, dtype=np.uint16),
                ]
            )
            self._write_raster(source, data)
            tasks = [
                BandExportTask(source, source, index, 3, root / "BANDS", None)
                for index in range(1, 4)
            ]

            outputs, multiband = _run_band_tasks(
                tasks, 3, threading.Event(), None
            )

            band_paths = outputs[str(source.resolve())]
            self.assertEqual(len(band_paths), 3)
            for index, path in enumerate(band_paths):
                with rasterio.open(path) as exported:
                    self.assertEqual(exported.count, 1)
                    self.assertEqual(exported.read(1).ndim, 2)
                    np.testing.assert_array_equal(exported.read(1), data[index])
            with rasterio.open(multiband[str(source.resolve())]) as exported:
                self.assertEqual(exported.count, 3)
                np.testing.assert_array_equal(exported.read(), data)

    def test_cancelled_band_tasks_leave_no_new_or_partial_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            self._write_raster(
                source, np.full((2, 32, 32), 5, dtype=np.uint16)
            )
            tasks = [
                BandExportTask(source, source, index, 2, root / "BANDS", None)
                for index in range(1, 3)
            ]
            cancel_event = threading.Event()
            cancel_event.set()

            with self.assertRaises(ExportCancelled):
                _run_band_tasks(tasks, 2, cancel_event, None)

            self.assertFalse(list(root.rglob("*.partial")))
            self.assertFalse(list((root / "BANDS").glob("*.tif")))
            self.assertFalse(
                (root / "STATE" / "phenolapseuav_export.json").exists()
            )

    def test_cancellation_during_parallel_writes_cleans_temporary_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tif"
            self._write_raster(
                source, np.full((2, 64, 64), 5, dtype=np.uint16)
            )
            tasks = [
                BandExportTask(source, source, index, 2, root / "BANDS", None)
                for index in range(1, 3)
            ]
            cancel_event = threading.Event()
            original_writer = registration_module._write_registered_band_task

            def cancelling_writer(task, event, block_completed):
                def cancel_after_block(completed_task):
                    block_completed(completed_task)
                    event.set()

                return original_writer(task, event, cancel_after_block)

            with patch.object(
                registration_module,
                "_write_registered_band_task",
                side_effect=cancelling_writer,
            ):
                with self.assertRaises(ExportCancelled):
                    _run_band_tasks(tasks, 2, cancel_event, None)

            self.assertFalse(list(root.rglob("*.partial")))
            self.assertFalse(list((root / "BANDS").glob("*.tif")))
            self.assertFalse(list((root / "MULTIBAND").glob("*.tif")))

    def test_completed_export_manifest_can_be_restored(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixed = root / "fixed.tif"
            moving = root / "moving.tif"
            fixed_band = root / "fixed_B01_registered.tif"
            moving_band = root / "moving_B01_registered.tif"
            fixed_all = root / "fixed_registered_all_bands.tif"
            moving_all = root / "moving_registered_all_bands.tif"
            for path in (fixed_band, moving_band, fixed_all, moving_all):
                path.touch()
            result = {
                "output_directory": str(root),
                "reference_path": str(fixed),
                "outputs": {
                    str(fixed): [str(fixed_band)],
                    str(moving): [str(moving_band)],
                },
                "multiband_outputs": {
                    str(fixed): str(fixed_all),
                    str(moving): str(moving_all),
                },
                "unreachable": [],
                "skipped_edges": [],
            }

            _save_export_manifest(result)
            restored = _load_export_manifest(root)

            self.assertEqual(restored, result)
            moving_all.unlink()
            self.assertIsNone(_load_export_manifest(root))

    def test_resolution_manifests_coexist_and_filter_stale_signatures(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "band.tif"
            output.touch()
            base = {
                "output_directory": str(root),
                "reference_path": str(root / "fixed.tif"),
                "outputs": {str(root / "fixed.tif"): [str(output)]},
                "multiband_outputs": {},
            }
            for resolution, signature in (
                ("FULL", "current"),
                ("SUBSAMPLE_4", "stale"),
            ):
                _save_export_manifest(
                    {
                        **base,
                        "resolution": resolution,
                        "registration_signature": signature,
                    }
                )

            restored = _load_resolution_export_manifests(root, "current")

            self.assertEqual(set(restored), {"FULL"})

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
            self.assertEqual(
                _read_imagej_display_ranges(Path(output)),
                (3.0, 4.0, 7.0, 8.0),
            )


if __name__ == "__main__":
    unittest.main()
