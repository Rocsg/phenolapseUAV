"""Tests for series filename parsing and discovery."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from series_registration import (
    RegistrationEdge,
    SeriesImage,
    SeriesRegistrationWidget,
    _session_with_relative_paths,
    _session_with_resolved_paths,
    automatic_state_path,
    best_moving_path,
    discover_series,
    parse_series_image,
    registered_output_directory,
)


class SeriesPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from qtpy.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _widget(self) -> SeriesRegistrationWidget:
        controller = Mock(session_changed_callbacks=[])
        controller.session_dict.return_value = {}

        def load_pair(fixed: Path, moving: Path, session: dict | None) -> None:
            controller.session_dict.return_value = deepcopy(session) if session else {
                "fixed": {"path": str(fixed)},
                "moving": {"path": str(moving)},
                "landmarks_full_resolution_yx": [],
            }

        controller.load_image_pair.side_effect = load_pair
        widget = SeriesRegistrationWidget(controller)
        self.addCleanup(widget.deleteLater)
        return widget

    def _directory(self, root: Path) -> Path:
        directory = root.resolve() / "Raw"
        directory.mkdir()
        for name in ("A_MS.tif", "B_MS.tif", "A_IR.tif"):
            (directory / name).touch()
        return directory

    def test_initial_pair_autosaves_and_restores_without_graph_click(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary))
            widget = self._widget()
            widget.load_directory(directory)
            self.assertEqual(
                widget.current_edge,
                RegistrationEdge(directory / "A_MS.tif", directory / "B_MS.tif"),
            )
            session = widget.controller.session_dict.return_value
            session["landmarks_full_resolution_yx"] = [
                {"pair_id": 1, "fixed_yx": [10, 20], "moving_yx": [11, 21], "enabled": True}
            ]
            widget.controller.session_changed_callbacks[0]()
            payload = json.loads(automatic_state_path(directory).read_text())
            self.assertEqual(len(payload["registrations"]), 1)
            self.assertEqual(
                payload["registrations"][0]["session"]["landmarks_full_resolution_yx"],
                session["landmarks_full_resolution_yx"],
            )

            reopened = self._widget()
            reopened.load_directory(directory)
            self.assertEqual(reopened.current_edge, widget.current_edge)
            self.assertEqual(reopened.controller.session_dict(), session)

    def test_last_active_pair_and_all_sessions_survive_project_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = self._directory(root)
            widget = self._widget()
            widget.load_directory(directory)
            widget.controller.session_dict.return_value["landmarks_full_resolution_yx"] = [
                {"pair_id": 1, "fixed_yx": [1, 2], "moving_yx": [3, 4]}
            ]
            widget._session_changed()
            widget._activate_edge(directory / "A_MS.tif", directory / "A_IR.tif")
            widget.controller.session_dict.return_value["landmarks_full_resolution_yx"] = [
                {"pair_id": 2, "fixed_yx": [5, 6], "moving_yx": [7, 8]}
            ]
            widget._session_changed()
            copied = root / "copied"
            copied.mkdir()
            shutil.copytree(directory, copied / directory.name)
            shutil.copytree(
                registered_output_directory(directory),
                copied / registered_output_directory(directory).name,
            )
            reopened = self._widget()
            reopened.load_directory(copied / directory.name)
            self.assertEqual(reopened.current_edge.moving_path.name, "A_IR.tif")
            self.assertEqual(reopened.current_edge.fixed_path.parent.parent, copied)
            self.assertEqual(len(reopened.sessions), 2)
            self.assertEqual(
                reopened.controller.session_dict()["landmarks_full_resolution_yx"][0]["pair_id"],
                2,
            )
            reopened._activate_edge(
                copied / "Raw" / "A_MS.tif", copied / "Raw" / "B_MS.tif"
            )
            self.assertEqual(
                reopened.controller.session_dict()["landmarks_full_resolution_yx"][0]["pair_id"],
                1,
            )

    def test_save_state_includes_unsaved_complete_and_incomplete_points_on_all_edges(self) -> None:
        from napari_tps_register import RegistrationController

        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary))
            widget = self._widget()
            widget.load_directory(directory)
            first_points = [
                {"pair_id": 1, "fixed_yx": [1, 2], "moving_yx": [3, 4], "enabled": True},
                {"pair_id": 2, "fixed_yx": [5, 6], "moving_yx": None, "enabled": False},
            ]
            widget.controller.session_dict.return_value[
                "landmarks_full_resolution_yx"
            ] = deepcopy(first_points)
            widget._activate_edge(directory / "A_MS.tif", directory / "A_IR.tif")
            active_points = [
                {"pair_id": 3, "fixed_yx": [7, 8], "moving_yx": [9, 10], "enabled": False},
                {"pair_id": 4, "fixed_yx": [11, 12], "moving_yx": None, "enabled": True},
            ]
            widget.controller.session_dict.return_value[
                "landmarks_full_resolution_yx"
            ] = deepcopy(active_points)

            widget.controller.series_widget = widget
            RegistrationController.save_state(widget.controller)

            payload = json.loads(automatic_state_path(directory).read_text())
            self.assertEqual(len(payload["registrations"]), 2)
            self.assertEqual(
                [
                    item["session"]["landmarks_full_resolution_yx"]
                    for item in payload["registrations"]
                ],
                [first_points, active_points],
            )
            reopened = self._widget()
            reopened.load_directory(directory)
            self.assertEqual(
                reopened.controller.session_dict()["landmarks_full_resolution_yx"],
                active_points,
            )
            reopened._activate_edge(directory / "A_MS.tif", directory / "B_MS.tif")
            self.assertEqual(
                reopened.controller.session_dict()["landmarks_full_resolution_yx"],
                first_points,
            )

    def test_older_state_without_active_pair_restores_saved_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary))
            widget = self._widget()
            widget.load_directory(directory)
            state_path = automatic_state_path(directory)
            payload = json.loads(state_path.read_text())
            payload.pop("active_registration")
            registration = payload["registrations"][0]
            registration["moving_path"] = "Raw/A_IR.tif"
            registration["session"]["moving"]["path"] = "Raw/A_IR.tif"
            registration["session"]["landmarks_full_resolution_yx"] = [
                {"pair_id": 3, "fixed_yx": [10, 20], "moving_yx": [11, 21]}
            ]
            state_path.write_text(json.dumps(payload))
            reopened = self._widget()
            reopened.load_directory(directory)
            self.assertEqual(reopened.current_edge.moving_path.name, "A_IR.tif")
            self.assertEqual(
                reopened.controller.session_dict()["landmarks_full_resolution_yx"][0]["pair_id"],
                3,
            )

    def test_invalid_state_is_reported_without_overwriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary))
            state_path = automatic_state_path(directory)
            state_path.parent.mkdir(parents=True)
            state_path.write_text('{"format": "napari-tps-register-series",')
            original = state_path.read_bytes()
            widget = self._widget()
            with self.assertRaisesRegex(ValueError, "Unable to load saved series state"):
                widget.load_directory(directory)
            self.assertEqual(state_path.read_bytes(), original)
            widget.controller.load_image_pair.assert_not_called()
            self.assertIsNone(widget.directory)

    def test_failed_pair_load_does_not_replace_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary))
            initial = self._widget()
            initial.load_directory(directory)
            original = automatic_state_path(directory).read_bytes()
            reopened = self._widget()
            reopened.controller.load_image_pair.side_effect = OSError("Unreadable TIFF")
            with self.assertRaisesRegex(OSError, "Unreadable TIFF"):
                reopened.load_directory(directory)
            self.assertEqual(automatic_state_path(directory).read_bytes(), original)
            self.assertIsNone(reopened.current_edge)

    def test_missing_source_does_not_discard_saved_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary))
            widget = self._widget()
            widget.load_directory(directory)
            original = automatic_state_path(directory).read_bytes()
            (directory / "B_MS.tif").unlink()
            reopened = self._widget()
            with self.assertRaisesRegex(ValueError, "missing source images"):
                reopened.load_directory(directory)
            self.assertEqual(automatic_state_path(directory).read_bytes(), original)
            reopened.controller.load_image_pair.assert_not_called()

    def test_initial_save_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._directory(Path(temporary))
            widget = self._widget()
            widget._write_state_file = Mock(return_value=False)
            with self.assertRaisesRegex(OSError, "Unable to save the active registration"):
                widget.load_directory(directory)


class SeriesDiscoveryTests(unittest.TestCase):
    def test_session_paths_are_portable_and_legacy_paths_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            raw = parent / "STR"
            raw.mkdir()
            fixed = raw / "fixed_MS.tif"
            moving = raw / "moving_MS.tif"
            session = {
                "fixed": {"path": str(fixed)},
                "moving": {"path": str(moving)},
            }

            portable = _session_with_relative_paths(session, parent)

            self.assertEqual(portable["fixed"]["path"], "STR/fixed_MS.tif")
            self.assertEqual(portable["moving"]["path"], "STR/moving_MS.tif")
            self.assertEqual(
                _session_with_resolved_paths(portable, parent), session
            )
            self.assertEqual(
                _session_with_resolved_paths(session, parent), session
            )

    def test_state_path_uses_registered_sibling_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            series = root / "Input" / "STR"
            series.mkdir(parents=True)
            self.assertEqual(
                automatic_state_path(series),
                root
                / "Input"
                / "STR_REGISTERED"
                / "STATE"
                / "manual_registration_state.json",
            )

    def test_output_root_is_next_to_selected_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            series = Path(temporary) / "STR"
            series.mkdir()
            self.assertEqual(
                registered_output_directory(series),
                series.parent / "STR_REGISTERED",
            )

    def test_parse_current_drone_naming(self) -> None:
        image = parse_series_image(Path("2022_13h_STR_MS.tif"))
        self.assertEqual(image.scene, "2022_13h_STR")
        self.assertEqual(image.modality, "MS")

    def test_parse_ir_case_insensitively(self) -> None:
        image = parse_series_image(Path("2023_09h_STR_ir.TIFF"))
        self.assertEqual(image.scene, "2023_09h_STR")
        self.assertEqual(image.modality, "IR")

    def test_unknown_modality_is_not_guessed(self) -> None:
        with self.assertRaises(ValueError):
            parse_series_image(Path("2022_13h_STR.tif"))

    def test_discovery_builds_unique_scene_modality_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in (
                "2022_09h_STR_MS.tif",
                "2022_09h_STR_IR.tif",
                "2022_13h_STR_MS.tif",
                "notes.txt",
            ):
                (directory / name).touch()
            images, warnings = discover_series(directory)
        self.assertEqual(len(images), 3)
        self.assertEqual(warnings, [])
        self.assertEqual(
            {(image.scene, image.modality) for image in images},
            {
                ("2022_09h_STR", "MS"),
                ("2022_09h_STR", "IR"),
                ("2022_13h_STR", "MS"),
            },
        )

    def test_best_moving_prefers_next_alphanumeric_image_in_same_modality(
        self,
    ) -> None:
        images = [
            SeriesImage(Path("A_MS.tif").resolve(), "A", "MS"),
            SeriesImage(Path("A_IR.tif").resolve(), "A", "IR"),
            SeriesImage(Path("B_MS.tif").resolve(), "B", "MS"),
        ]

        self.assertEqual(
            best_moving_path(images, Path("A_MS.tif")),
            Path("B_MS.tif").resolve(),
        )

    def test_best_moving_falls_back_to_same_scene_in_next_modality(self) -> None:
        images = [
            SeriesImage(Path("A_MS.tif").resolve(), "A", "MS"),
            SeriesImage(Path("B_MS.tif").resolve(), "B", "MS"),
            SeriesImage(Path("B_IR.tif").resolve(), "B", "IR"),
        ]

        self.assertEqual(
            best_moving_path(images, Path("B_MS.tif")),
            Path("B_IR.tif").resolve(),
        )

    def test_best_moving_returns_none_without_later_image_or_modality(self) -> None:
        images = [SeriesImage(Path("B_IR.tif").resolve(), "B", "IR")]

        self.assertIsNone(best_moving_path(images, Path("B_IR.tif")))

    def test_fixed_click_activates_automatic_moving_image(self) -> None:
        fixed = Path("A_MS.tif").resolve()
        moving = Path("B_MS.tif").resolve()
        widget = SimpleNamespace(
            images=[
                SeriesImage(fixed, "A", "MS"),
                SeriesImage(moving, "B", "MS"),
            ],
            first_selection=None,
            _activate_edge=Mock(),
            _refresh_node_roles=Mock(),
            status=Mock(),
        )

        SeriesRegistrationWidget._graph_node_clicked(widget, fixed, "fixed")

        widget._activate_edge.assert_called_once_with(fixed, moving)

    def test_moving_click_overrides_automatic_choice_for_current_fixed(self) -> None:
        fixed = Path("A_MS.tif").resolve()
        automatic = Path("B_MS.tif").resolve()
        replacement = Path("C_MS.tif").resolve()
        widget = SimpleNamespace(
            first_selection=None,
            current_edge=RegistrationEdge(fixed, automatic),
            _activate_edge=Mock(),
            status=Mock(),
        )

        SeriesRegistrationWidget._graph_node_clicked(widget, replacement, "moving")

        widget._activate_edge.assert_called_once_with(fixed, replacement)


if __name__ == "__main__":
    unittest.main()
