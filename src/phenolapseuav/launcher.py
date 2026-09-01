"""Standalone graphical launcher for the PhenoLapseUAV napari widget."""

from __future__ import annotations


def main() -> int:
    """Open napari with the PhenoLapseUAV registration panel already docked."""
    import napari

    from phenolapseuav._widget import PhenolapseUAVWidget

    viewer = napari.Viewer(title="PhenoLapseUAV")
    widget = PhenolapseUAVWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="PhenoLapseUAV registration",
        area="right",
    )
    napari.run()
    return 0
