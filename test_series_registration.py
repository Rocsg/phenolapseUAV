"""Tests for series filename parsing and discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from series_registration import (
    RegistrationEdge,
    SeriesImage,
    SeriesRegistrationWidget,
    automatic_state_path,
    best_moving_path,
    discover_series,
    parse_series_image,
    registered_output_directory,
)


class SeriesDiscoveryTests(unittest.TestCase):
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
