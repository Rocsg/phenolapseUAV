"""Adaptive sizing shared by plugin and command-line registration panels."""

from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QShowEvent
from qtpy.QtWidgets import QDockWidget, QMainWindow, QSizePolicy, QWidget


class RegistrationPanel(QWidget):
    """Fill the dock and request one third of the window on first display."""

    def __init__(self) -> None:
        super().__init__()
        self._dock_size_initialized = False

    def showEvent(self, event: QShowEvent) -> None:
        # napari overrides this policy when it installs a dock widget.
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        super().showEvent(event)
        if not self._dock_size_initialized:
            QTimer.singleShot(0, self._initialize_dock_size)

    def _initialize_dock_size(self) -> None:
        if self._dock_size_initialized:
            return
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QDockWidget):
            if isinstance(parent, RegistrationPanel):
                return
            parent = parent.parentWidget()
        if not isinstance(parent, QDockWidget) or parent.isFloating():
            return
        window = parent.window()
        if not isinstance(window, QMainWindow):
            return
        if window.dockWidgetArea(parent) not in (
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        ):
            return
        window.resizeDocks([parent], [max(1, window.width() // 3)], Qt.Orientation.Horizontal)
        self._dock_size_initialized = True
