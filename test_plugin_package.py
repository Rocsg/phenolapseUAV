"""Release-package smoke tests for phenolapseUAV."""

from __future__ import annotations

import unittest
from pathlib import Path


class PluginPackageTests(unittest.TestCase):
    def test_package_import_is_headless_safe(self) -> None:
        import phenolapseuav

        self.assertRegex(phenolapseuav.__version__, r"^\d+\.\d+\.\d+")

    def test_manifest_declares_widget(self) -> None:
        manifest = (
            Path(__file__).parent / "src" / "phenolapseuav" / "napari.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("name: phenolapseUAV", manifest)
        self.assertIn("phenolapseuav._widget:PhenolapseUAVWidget", manifest)


if __name__ == "__main__":
    unittest.main()