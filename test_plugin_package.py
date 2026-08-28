"""Release-package smoke tests for phenolapseUAV."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class PluginPackageTests(unittest.TestCase):
    def test_package_import_is_headless_safe(self) -> None:
        import phenolapseuav

        self.assertRegex(phenolapseuav.__version__, r"^\d+\.\d+\.\d+")

    def test_manifest_declares_widget(self) -> None:
        manifest = (
            Path(__file__).parent / "src" / "phenolapseuav" / "napari.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("name: phenolapseuav", manifest)
        self.assertIn("phenolapseuav._widget:PhenolapseUAVWidget", manifest)

    def test_pair_state_path_is_stored_in_registered_sibling(self) -> None:
        from phenolapseuav._widget import PhenolapseUAVWidget

        fixed = Path("/tmp/Input/PAIR/fixed.tif")
        moving = Path("/tmp/Input/PAIR/moving.tif")

        state_path = PhenolapseUAVWidget._pair_state_path(fixed, moving)

        self.assertEqual(state_path.parent, Path("/tmp/Input/PAIR_REGISTERED/STATE"))
        self.assertTrue(state_path.name.startswith("fixed__TO__moving_"))
        self.assertEqual(state_path.suffix, ".json")

    def test_workflow_choice_reveals_only_its_context_and_hides_welcome(self) -> None:
        from phenolapseuav._widget import PhenolapseUAVWidget

        widget = SimpleNamespace(
            pair_mode_button=Mock(),
            series_mode_button=Mock(),
            pair_box=Mock(),
            series_box=Mock(),
            preview_options=Mock(),
            welcome_message=Mock(),
        )

        PhenolapseUAVWidget._select_workflow(widget, "pair")

        widget.pair_mode_button.setChecked.assert_called_once_with(True)
        widget.series_mode_button.setChecked.assert_called_once_with(False)
        widget.pair_box.setVisible.assert_called_once_with(True)
        widget.series_box.setVisible.assert_called_once_with(False)
        widget.preview_options.setVisible.assert_called_once_with(True)
        widget.welcome_message.hide.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
