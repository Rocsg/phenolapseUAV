"""npe2 widget entry point for phenolapseUAV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PhenolapseUAVWidget(QWidget):
    """Launch the landmark-registration controller inside an existing viewer."""

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.controller: Optional[Any] = None
        self.fixed_path: Optional[Path] = None
        self.moving_path: Optional[Path] = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Register one FIXED/MOVING pair or open a complete MS/IR UAV series."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        pair_box = QGroupBox("Open one image pair")
        pair_layout = QVBoxLayout(pair_box)
        buttons = QHBoxLayout()
        self.fixed_button = QPushButton("Choose FIXED…")
        self.moving_button = QPushButton("Choose MOVING…")
        buttons.addWidget(self.fixed_button)
        buttons.addWidget(self.moving_button)
        pair_layout.addLayout(buttons)
        self.pair_status = QLabel("No pair selected.")
        self.pair_status.setWordWrap(True)
        pair_layout.addWidget(self.pair_status)
        self.open_pair_button = QPushButton("Open selected pair")
        self.open_pair_button.setEnabled(False)
        pair_layout.addWidget(self.open_pair_button)
        layout.addWidget(pair_box)

        options = QHBoxLayout()
        options.addWidget(QLabel("Preview factor"))
        self.preview_factor = QSpinBox()
        self.preview_factor.setRange(1, 64)
        self.preview_factor.setValue(4)
        options.addWidget(self.preview_factor)
        options.addStretch(1)
        layout.addLayout(options)

        self.open_series_button = QPushButton("Open MS/IR series directory…")
        layout.addWidget(self.open_series_button)
        layout.addStretch(1)

        self.fixed_button.clicked.connect(self._choose_fixed)
        self.moving_button.clicked.connect(self._choose_moving)
        self.open_pair_button.clicked.connect(self._open_pair)
        self.open_series_button.clicked.connect(self._open_series)

    def _choose_raster(self, title: str) -> Optional[Path]:
        filename, _ = QFileDialog.getOpenFileName(
            self, title, "", "GeoTIFF (*.tif *.tiff)"
        )
        return Path(filename).resolve() if filename else None

    def _choose_fixed(self) -> None:
        selected = self._choose_raster("Choose the FIXED GeoTIFF")
        if selected is not None:
            self.fixed_path = selected
            self._refresh_pair_status()

    def _choose_moving(self) -> None:
        selected = self._choose_raster("Choose the MOVING GeoTIFF")
        if selected is not None:
            self.moving_path = selected
            self._refresh_pair_status()

    def _refresh_pair_status(self) -> None:
        fixed = self.fixed_path.name if self.fixed_path else "not selected"
        moving = self.moving_path.name if self.moving_path else "not selected"
        self.pair_status.setText(f"FIXED: {fixed}\nMOVING: {moving}")
        self.open_pair_button.setEnabled(
            self.fixed_path is not None and self.moving_path is not None
        )

    def _install_controller(self, fixed_path: Path, moving_path: Path) -> Any:
        from napari_tps_register import RasterioSource, RegistrationController

        if self.controller is not None:
            self.controller.load_image_pair(fixed_path, moving_path)
            return self.controller
        self.controller = RegistrationController(
            self.viewer,
            RasterioSource(fixed_path),
            RasterioSource(moving_path),
            self.preview_factor.value(),
        )
        self.layout().insertWidget(0, self.controller.widget)
        for widget in (
            self.fixed_button.parentWidget(),
            self.open_series_button,
        ):
            widget.setVisible(False)
        return self.controller

    def _open_pair(self) -> None:
        if self.fixed_path is None or self.moving_path is None:
            return
        try:
            self._install_controller(self.fixed_path, self.moving_path)
        except Exception as error:
            QMessageBox.critical(self, "Unable to open images", str(error))

    def _open_series(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Open an MS/IR series directory", str(Path.cwd())
        )
        if not selected:
            return
        try:
            from series_registration import discover_series

            directory = Path(selected).resolve()
            images, warnings = discover_series(directory)
            if len(images) < 2:
                details = f"\n\n{' '.join(warnings)}" if warnings else ""
                raise ValueError(
                    "At least two recognized <scene>_<MS|IR>.tif images are required."
                    + details
                )
            preferred = sorted(
                (image for image in images if image.modality == "MS"),
                key=lambda image: image.scene,
            )
            initial = preferred if len(preferred) >= 2 else sorted(
                images, key=lambda image: (image.scene, image.modality)
            )
            controller = self._install_controller(initial[0].path, initial[1].path)
            controller.series_widget.load_directory(directory)
        except Exception as error:
            QMessageBox.critical(self, "Unable to open series", str(error))
