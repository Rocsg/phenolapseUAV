"""Clickable series graph for the napari manual-registration controller.

This module deliberately contains only series discovery, graph UI, and project
persistence. Image loading and landmark registration remain owned by
``RegistrationController``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QFont, QPen
from qtpy.QtWidgets import (
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from napari_tps_register import RegistrationController


IMAGE_SUFFIXES = {".tif", ".tiff"}
KNOWN_MODALITIES = ("MS", "IR")
STATE_FILENAME = "manual_registration_state.json"


def automatic_state_path(directory: Path) -> Path:
    """Return OUTPUT/<relative input path>/STATE_FILENAME for a series directory."""
    directory = directory.resolve()
    input_root = next(
        (parent for parent in (directory, *directory.parents) if parent.name.lower() == "input"),
        None,
    )
    if input_root is not None:
        siblings = {
            child.name.lower(): child
            for child in input_root.parent.iterdir()
            if child.is_dir()
        }
        output_root = siblings.get("output", input_root.parent / "OUTPUT")
        relative = directory.relative_to(input_root)
        return output_root / relative / STATE_FILENAME
    return directory.parent / "OUTPUT" / directory.name / STATE_FILENAME


@dataclass(frozen=True)
class SeriesImage:
    path: Path
    scene: str
    modality: str


@dataclass(frozen=True)
class RegistrationEdge:
    fixed_path: Path
    moving_path: Path

    @property
    def key(self) -> str:
        return f"{self.fixed_path.resolve()}::{self.moving_path.resolve()}"


def parse_series_image(path: Path) -> SeriesImage:
    """Parse ``..._<MS|IR>.tif`` without guessing an unknown modality."""
    parts = path.stem.split("_")
    modality_indices = [
        index for index, value in enumerate(parts) if value.upper() in KNOWN_MODALITIES
    ]
    if not modality_indices:
        raise ValueError(f"Filename has no explicit MS/IR modality: {path.name}")
    index = modality_indices[-1]
    modality = parts[index].upper()
    scene_parts = parts[:index] + parts[index + 1 :]
    scene = "_".join(scene_parts).strip("_")
    if not scene:
        raise ValueError(f"Filename has no scene identifier: {path.name}")
    return SeriesImage(path.resolve(), scene, modality)


def discover_series(directory: Path) -> tuple[list[SeriesImage], list[str]]:
    images: list[SeriesImage] = []
    warnings: list[str] = []
    occupied: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            image = parse_series_image(path)
        except ValueError as error:
            warnings.append(str(error))
            continue
        slot = (image.scene, image.modality)
        if slot in occupied:
            warnings.append(
                f"Duplicate {image.scene}/{image.modality}: ignored {path.name} "
                f"(already using {occupied[slot].name})"
            )
            continue
        occupied[slot] = path
        images.append(image)
    return images, warnings


class ImageNode(QGraphicsRectItem):
    """Selectable graphics item carrying one image path."""

    def __init__(self, image: SeriesImage, x: float, y: float) -> None:
        super().__init__(0, 0, 132, 30)
        self.setPos(x, y)
        self.image = image
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self.setBrush(QBrush(QColor("#30343b")))
        self.setPen(QPen(QColor("#666b73"), 1.5))
        label = QGraphicsSimpleTextItem(image.path.name, self)
        label.setBrush(QBrush(QColor("#f0f0f0")))
        label.setFont(QFont("Sans Serif", 7))
        label.setPos(5, 7)
        label.setToolTip(str(image.path))
        label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setToolTip(str(image.path))

    def set_role(self, role: str) -> None:
        if role == "fixed":
            self.setPen(QPen(QColor("#ff5555"), 4))
        elif role == "moving":
            self.setPen(QPen(QColor("#55dd66"), 4))
        else:
            self.setPen(QPen(QColor("#666b73"), 1.5))


class SeriesRegistrationWidget(QGroupBox):
    """Series graph dock section linked to a registration controller."""

    def __init__(self, controller: "RegistrationController") -> None:
        super().__init__("Manual registration series")
        self.controller = controller
        self.directory: Optional[Path] = None
        self.images: list[SeriesImage] = []
        self.nodes: dict[Path, ImageNode] = {}
        self.edge_lines: dict[str, QGraphicsLineItem] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, RegistrationEdge] = {}
        self.current_edge: Optional[RegistrationEdge] = None
        self.first_selection: Optional[Path] = None
        self.state_path: Optional[Path] = None
        self._loading_pair = False

        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        open_directory = QPushButton("Open directory")
        open_directory.clicked.connect(self.open_directory_dialog)
        buttons.addWidget(open_directory)
        layout.addLayout(buttons)

        self.status = QLabel(
            "Open a directory, then click FIXED (red) and MOVING (green)."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.scene = QGraphicsScene(self)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumHeight(210)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.view)
        controller.session_changed_callbacks.append(self._session_changed)

    def open_directory_dialog(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Open series directory", str(self.directory or Path.cwd())
        )
        if selected:
            self.load_directory(Path(selected))

    def load_directory(self, directory: Path) -> None:
        if self.directory is not None:
            self.save_state(show_status=False)
        images, warnings = discover_series(directory)
        self.directory = directory.resolve()
        self.state_path = automatic_state_path(self.directory)
        self.images = images
        self.sessions.clear()
        self.edges.clear()
        self.current_edge = None
        self.first_selection = None
        restored = 0
        state_warning = ""
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                if payload.get("format") != "napari-tps-register-series":
                    raise ValueError("unrecognized file format")
                available = {image.path for image in images}
                for item in payload.get("registrations", []):
                    edge = RegistrationEdge(
                        Path(item["fixed_path"]).resolve(),
                        Path(item["moving_path"]).resolve(),
                    )
                    if edge.fixed_path not in available or edge.moving_path not in available:
                        continue
                    self.edges[edge.key] = edge
                    self.sessions[edge.key] = item.get("session", {})
                    restored += 1
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                state_warning = f" — state not loaded: {error}"
        self._rebuild_scene()
        warning_text = f" — {len(warnings)} warning(s)" if warnings else ""
        self.status.setText(
            f"{len(images)} image(s), {restored} pair(s) restored"
            f"{warning_text}{state_warning}. Click FIXED, then MOVING."
        )
        tooltip_lines = [f"Automatic state file: {self.state_path}"]
        tooltip_lines.extend(warnings)
        self.status.setToolTip("\n".join(tooltip_lines))

    def _rebuild_scene(self) -> None:
        self.scene.blockSignals(True)
        self.scene.clear()
        self.nodes.clear()
        self.edge_lines.clear()
        scenes = sorted({image.scene for image in self.images})
        modalities = [
            modality
            for modality in KNOWN_MODALITIES
            if any(image.modality == modality for image in self.images)
        ]
        x_for_modality = {
            modality: 150.0 * index + 105
            for index, modality in enumerate(modalities)
        }
        y_for_scene = {scene: 48.0 * index + 32 for index, scene in enumerate(scenes)}

        for modality, x in x_for_modality.items():
            header = self.scene.addText(modality)
            header.setFont(QFont("Sans Serif", 8, QFont.Weight.Bold))
            header.setDefaultTextColor(
                QColor("#ff7777") if modality == "MS" else QColor("#77dd88")
            )
            header.setPos(x + 54, 3)
        for scene, y in y_for_scene.items():
            row = self.scene.addText(scene)
            row.setFont(QFont("Sans Serif", 8))
            row.setDefaultTextColor(QColor("#c5c8cc"))
            row.setPos(2, y + 6)

        by_slot = {(image.scene, image.modality): image for image in self.images}
        # Dark potential edges connect adjacent acquisitions within a modality
        # and same-scene MS/IR pairs. Edited registrations brighten these edges.
        for modality in modalities:
            present = [image for image in self.images if image.modality == modality]
            present.sort(key=lambda image: image.scene)
            for first, second in zip(present, present[1:]):
                self._add_background_wire(first, second, x_for_modality, y_for_scene)
        for scene in scenes:
            if (scene, "MS") in by_slot and (scene, "IR") in by_slot:
                self._add_background_wire(
                    by_slot[(scene, "MS")],
                    by_slot[(scene, "IR")],
                    x_for_modality,
                    y_for_scene,
                )

        for image in self.images:
            node = ImageNode(
                image,
                x_for_modality[image.modality],
                y_for_scene[image.scene],
            )
            self.scene.addItem(node)
            self.nodes[image.path] = node
        self.scene.blockSignals(False)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-10, -10, 10, 10))
        self._refresh_edges()

    def _add_background_wire(
        self,
        first: SeriesImage,
        second: SeriesImage,
        x_positions: dict[str, float],
        y_positions: dict[str, float],
    ) -> None:
        x1 = x_positions[first.modality] + 66
        y1 = y_positions[first.scene] + 15
        x2 = x_positions[second.modality] + 66
        y2 = y_positions[second.scene] + 15
        line = self.scene.addLine(x1, y1, x2, y2, QPen(QColor("#25282d"), 3))
        line.setZValue(-10)

    def _selection_changed(self) -> None:
        selected = [item for item in self.scene.selectedItems() if isinstance(item, ImageNode)]
        if not selected:
            return
        node = selected[-1]
        self.scene.blockSignals(True)
        for item in self.nodes.values():
            item.setSelected(False)
        node.setSelected(True)
        self.scene.blockSignals(False)
        self._choose_image(node.image.path)

    def _choose_image(self, path: Path) -> None:
        if self.first_selection is None:
            self.first_selection = path
            self._refresh_node_roles()
            self.status.setText(
                f"FIXED: {path.name}. Now click the MOVING image."
            )
            return
        if path == self.first_selection:
            self.first_selection = None
            self._refresh_node_roles()
            self.status.setText("FIXED selection canceled.")
            return
        self._store_current_session()
        self._write_state_file()
        edge = RegistrationEdge(self.first_selection, path)
        self.edges[edge.key] = edge
        self.current_edge = edge
        self.first_selection = None
        self._loading_pair = True
        try:
            self.controller.load_image_pair(
                edge.fixed_path,
                edge.moving_path,
                self.sessions.get(edge.key),
            )
        finally:
            self._loading_pair = False
        self._refresh_node_roles()
        self._ensure_edge_line(edge)
        self._refresh_edges()
        self._store_current_session()
        self._write_state_file()
        self.status.setText(
            f"Active edge: FIXED {edge.fixed_path.name} → MOVING {edge.moving_path.name}"
        )

    def _store_current_session(self) -> None:
        if self.current_edge is None or self._loading_pair:
            return
        self.sessions[self.current_edge.key] = self.controller.session_dict()
        self.edges[self.current_edge.key] = self.current_edge

    def _session_changed(self) -> None:
        if self.current_edge is None or self._loading_pair:
            return
        self._store_current_session()
        self._refresh_edges()
        self._write_state_file()

    def _refresh_node_roles(self) -> None:
        for path, node in self.nodes.items():
            role = ""
            if self.current_edge and path == self.current_edge.fixed_path:
                role = "fixed"
            if self.current_edge and path == self.current_edge.moving_path:
                role = "moving"
            if self.first_selection == path:
                role = "fixed"
            node.set_role(role)

    def _ensure_edge_line(self, edge: RegistrationEdge) -> None:
        if edge.key in self.edge_lines:
            return
        fixed_node = self.nodes.get(edge.fixed_path)
        moving_node = self.nodes.get(edge.moving_path)
        if fixed_node is None or moving_node is None:
            return
        first = fixed_node.sceneBoundingRect().center()
        second = moving_node.sceneBoundingRect().center()
        line = self.scene.addLine(
            first.x(), first.y(), second.x(), second.y(), QPen(QColor("#33373d"), 4)
        )
        line.setZValue(-5)
        self.edge_lines[edge.key] = line

    def _refresh_edges(self) -> None:
        for key, edge in self.edges.items():
            self._ensure_edge_line(edge)
            line = self.edge_lines.get(key)
            if line is None:
                continue
            payload = self.sessions.get(key, {})
            landmarks = payload.get("landmarks_full_resolution_yx", [])
            complete = sum(
                bool(item.get("enabled", True))
                and item.get("fixed_yx") is not None
                and item.get("moving_yx") is not None
                for item in landmarks
            )
            if complete >= 10:
                color = "#55dd88"
            elif complete >= 7:
                color = "#e3c94f"
            elif complete >= 4:
                color = "#e38a3f"
            elif complete:
                color = "#8b6543"
            else:
                color = "#33373d"
            line.setPen(QPen(QColor(color), 4))

    def project_dict(self) -> dict[str, Any]:
        self._store_current_session()
        return {
            "format": "napari-tps-register-series",
            "version": 1,
            "directory": str(self.directory) if self.directory else None,
            "images": [
                {
                    "path": str(image.path),
                    "scene": image.scene,
                    "modality": image.modality,
                }
                for image in self.images
            ],
            "registrations": [
                {
                    "fixed_path": str(edge.fixed_path),
                    "moving_path": str(edge.moving_path),
                    "session": self.sessions.get(key, {}),
                }
                for key, edge in self.edges.items()
            ],
        }

    def _write_state_file(self) -> bool:
        if self.state_path is None:
            return False
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.project_dict(), indent=2), encoding="utf-8"
            )
            temporary.replace(self.state_path)
            return True
        except OSError as error:
            self.status.setText(f"Failed to save state: {error}")
            return False

    def save_state(self, *_: Any, show_status: bool = True) -> bool:
        """Persist every directed edge and landmark session automatically."""
        self._store_current_session()
        saved = self._write_state_file()
        if show_status:
            if saved:
                self.status.setText(f"State saved: {self.state_path}")
            elif self.state_path is None:
                self.status.setText("Open a series directory first.")
        return saved
