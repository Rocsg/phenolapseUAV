"""npe2 widget entry point for phenolapseUAV."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from qtpy.QtCore import QEvent, Qt
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
        self.launch_widget = QWidget()
        launch_layout = QVBoxLayout(self.launch_widget)
        launch_layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel("Choose an alignment workflow")
        intro.setStyleSheet("font-size: 16px; font-weight: bold; padding: 4px;")
        intro.setWordWrap(True)
        launch_layout.addWidget(intro)

        self.pair_mode_button = QPushButton(
            "1. Alignment of two images\nor orthoimages"
        )
        self.pair_mode_button.setMinimumHeight(54)
        self.pair_mode_button.setCheckable(True)
        launch_layout.addWidget(self.pair_mode_button)

        self.series_mode_button = QPushButton(
            "2. Alignment of a time-lapse series\nof images or orthoimages"
        )
        self.series_mode_button.setMinimumHeight(64)
        self.series_mode_button.setCheckable(True)

        self.pair_box = QGroupBox("Choose the two images")
        pair_layout = QVBoxLayout(self.pair_box)
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
        self.pair_box.setVisible(False)
        launch_layout.addWidget(self.pair_box)

        launch_layout.addWidget(self.series_mode_button)

        self.series_box = QGroupBox("Raw time-lapse series")
        self.series_box.setMinimumHeight(125)
        series_layout = QVBoxLayout(self.series_box)
        series_help = QLabel(
            "Choose the directory containing the raw images or orthoimages."
        )
        series_help.setWordWrap(True)
        series_help.setMinimumHeight(42)
        series_help.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        series_layout.addWidget(series_help)
        self.open_series_button = QPushButton("Choose raw-data directory…")
        series_layout.addWidget(self.open_series_button)
        self.series_box.setVisible(False)
        launch_layout.addWidget(self.series_box)

        self.preview_options = QWidget()
        options = QHBoxLayout(self.preview_options)
        options.setContentsMargins(0, 8, 0, 0)
        options.addWidget(QLabel("Preview factor (live)"))
        self.preview_factor = QSpinBox()
        self.preview_factor.setRange(1, 64)
        self.preview_factor.setValue(8)
        self.preview_factor.setToolTip(
            "Reload the in-memory previews at this subsampling factor. "
            "Landmarks remain stored at full resolution."
        )
        options.addWidget(self.preview_factor)
        options.addStretch(1)
        self.preview_options.setVisible(False)
        launch_layout.addWidget(self.preview_options)
        launch_layout.addStretch(1)
        layout.addWidget(self.launch_widget)

        self.pair_mode_button.clicked.connect(self._show_pair_workflow)
        self.series_mode_button.clicked.connect(self._show_series_workflow)
        self.fixed_button.clicked.connect(self._choose_fixed)
        self.moving_button.clicked.connect(self._choose_moving)
        self.open_pair_button.clicked.connect(self._open_pair)
        self.open_series_button.clicked.connect(self._open_series)
        self.preview_factor.valueChanged.connect(self._change_preview_factor)

        self._welcome_canvas = self._canvas_widget()
        self.welcome_message: Optional[QLabel] = None
        if self._welcome_canvas is not None:
            self.welcome_message = QLabel(self._welcome_canvas)
            self.welcome_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.welcome_message.setWordWrap(True)
            self.welcome_message.setText(
                "Welcome to PhenoLapseUAV\n\n"
                "Choose one of the two alignment options (right panel)."
            )
            self.welcome_message.setStyleSheet(
                "background: rgba(20, 20, 20, 220); color: white; "
                "font-size: 24px; font-weight: bold; "
                "border: 3px solid #66bbff; border-radius: 12px; padding: 24px;"
            )
            self.welcome_message.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )
            self._welcome_canvas.installEventFilter(self)
            self._position_welcome_message()
            self.welcome_message.show()

    def _canvas_widget(self) -> Optional[QWidget]:
        qt_viewer = getattr(self.viewer.window, "qt_viewer", None)
        if qt_viewer is None:
            qt_viewer = getattr(self.viewer.window, "_qt_viewer", None)
        canvas = getattr(qt_viewer, "canvas", None)
        native = getattr(canvas, "native", None)
        return native if isinstance(native, QWidget) else None

    def _position_welcome_message(self) -> None:
        if self.welcome_message is None or self._welcome_canvas is None:
            return
        width = min(700, max(360, self._welcome_canvas.width() - 80))
        self.welcome_message.setFixedWidth(width)
        self.welcome_message.adjustSize()
        self.welcome_message.move(
            max(0, (self._welcome_canvas.width() - self.welcome_message.width()) // 2),
            max(0, (self._welcome_canvas.height() - self.welcome_message.height()) // 2),
        )
        self.welcome_message.raise_()

    def eventFilter(self, watched: Any, event: Any) -> bool:
        if watched is self._welcome_canvas and event.type() == QEvent.Type.Resize:
            self._position_welcome_message()
        return super().eventFilter(watched, event)

    def _select_workflow(self, workflow: str) -> None:
        pair_mode = workflow == "pair"
        self.pair_mode_button.setChecked(pair_mode)
        self.series_mode_button.setChecked(not pair_mode)
        self.pair_box.setVisible(pair_mode)
        self.series_box.setVisible(not pair_mode)
        self.preview_options.setVisible(True)
        if self.welcome_message is not None:
            self.welcome_message.hide()

    def _show_pair_workflow(self, *_: Any) -> None:
        self._select_workflow("pair")

    def _show_series_workflow(self, *_: Any) -> None:
        self._select_workflow("series")

    def closeEvent(self, event: Any) -> None:
        if self._welcome_canvas is not None:
            self._welcome_canvas.removeEventFilter(self)
        if self.welcome_message is not None:
            self.welcome_message.deleteLater()
            self.welcome_message = None
        super().closeEvent(event)

    def _change_preview_factor(self, factor: int) -> None:
        if self.controller is None:
            return
        previous = self.controller.preview_factor
        try:
            self.controller.set_preview_factor(factor)
        except Exception as error:
            self.preview_factor.blockSignals(True)
            try:
                self.preview_factor.setValue(previous)
            finally:
                self.preview_factor.blockSignals(False)
            QMessageBox.critical(self, "Unable to reload previews", str(error))

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

    @staticmethod
    def _pair_state_path(fixed_path: Path, moving_path: Path) -> Path:
        from series_registration import registered_output_directory

        identity = f"{fixed_path.resolve()}::{moving_path.resolve()}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        fixed_name = fixed_path.stem[:60]
        moving_name = moving_path.stem[:60]
        return (
            registered_output_directory(moving_path.parent)
            / "STATE"
            / f"{fixed_name}__TO__{moving_name}_{digest}.json"
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
        self.launch_widget.setVisible(False)
        return self.controller

    def _open_pair(self) -> None:
        if self.fixed_path is None or self.moving_path is None:
            return
        try:
            state_path = self._pair_state_path(self.fixed_path, self.moving_path)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            controller = self._install_controller(self.fixed_path, self.moving_path)
            controller.configure_standalone_state(state_path)
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
