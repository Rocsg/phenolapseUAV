"""Release-package smoke tests for phenolapseUAV."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class PanelSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from qtpy.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_dock_fills_height_and_keeps_manual_width_after_resizing(self) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget, QMainWindow, QSizePolicy, QWidget

        from phenolapseuav._widget import PhenolapseUAVWidget

        window = QMainWindow()
        window.setCentralWidget(QWidget())
        panel = PhenolapseUAVWidget(
            SimpleNamespace(window=SimpleNamespace())
        )
        dock = QDockWidget("Registration", window)
        dock.setWidget(panel)
        # Reproduce the size policy imposed by napari's dock wrapper.
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        window.resize(1500, 900)
        try:
            window.show()
            self.app.processEvents()
            self.app.processEvents()
            self.assertAlmostEqual(dock.width(), window.width() / 3, delta=20)
            self.assertGreater(panel.height(), window.height() * 0.9)
            self.assertEqual(
                panel.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Expanding
            )
            window.resizeDocks([dock], [620], Qt.Orientation.Horizontal)
            self.app.processEvents()
            manual_width = dock.width()
            window.hide()
            window.show()
            self.app.processEvents()
            self.assertAlmostEqual(dock.width(), manual_width, delta=5)

            for width, height in ((1000, 600), (1800, 1000)):
                window.resize(width, height)
                self.app.processEvents()
                self.assertGreater(panel.height(), window.height() * 0.9)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_loading_controller_fills_panel_without_resetting_dock_width(self) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget, QMainWindow, QScrollArea, QVBoxLayout, QWidget

        from phenolapseuav._panel import RegistrationPanel
        from phenolapseuav._widget import PhenolapseUAVWidget

        window = QMainWindow()
        window.setCentralWidget(QWidget())
        panel = PhenolapseUAVWidget(SimpleNamespace(window=SimpleNamespace()))
        dock = QDockWidget("Registration", window)
        dock.setWidget(panel)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        window.resize(1500, 900)
        controller_panel = RegistrationPanel()
        scroll = QScrollArea()
        content = QWidget()
        content.setMinimumHeight(2000)
        scroll.setWidget(content)
        QVBoxLayout(controller_panel).addWidget(scroll)
        controller = SimpleNamespace(widget=controller_panel)
        try:
            window.show()
            self.app.processEvents()
            self.app.processEvents()
            window.resizeDocks([dock], [600], Qt.Orientation.Horizontal)
            self.app.processEvents()
            manual_width = dock.width()
            with (
                patch("napari_tps_register.RasterioSource"),
                patch("napari_tps_register.RegistrationController", return_value=controller),
            ):
                panel._install_controller(Path("fixed.tif"), Path("moving.tif"))
            self.app.processEvents()
            self.app.processEvents()
            self.assertAlmostEqual(dock.width(), manual_width, delta=5)
            self.assertGreater(controller_panel.height(), panel.height() * 0.9)
            self.assertGreater(scroll.height(), controller_panel.height() * 0.9)
            initial_height = scroll.height()
            window.resize(1100, 600)
            self.app.processEvents()
            self.assertLess(scroll.height(), initial_height)
            self.assertGreater(controller_panel.height(), panel.height() * 0.9)
            self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()


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
