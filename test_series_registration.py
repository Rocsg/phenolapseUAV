"""Tests for series filename parsing and discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from series_registration import automatic_state_path, discover_series, parse_series_image


class SeriesDiscoveryTests(unittest.TestCase):
    def test_state_path_mirrors_series_below_sibling_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            series = root / "Input" / "STR"
            output = root / "Output"
            series.mkdir(parents=True)
            output.mkdir()
            self.assertEqual(
                automatic_state_path(series),
                output / "STR" / "manual_registration_state.json",
            )

    def test_state_path_uses_output_when_input_marker_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            series = Path(temporary) / "STR"
            series.mkdir()
            self.assertEqual(
                automatic_state_path(series),
                series.parent / "OUTPUT" / "STR" / "manual_registration_state.json",
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


if __name__ == "__main__":
    unittest.main()
