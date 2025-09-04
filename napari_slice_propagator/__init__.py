"""
Napari plugin for semi-automated 3D segmentation with slice propagation.
"""

__version__ = "0.1.0"

from ._widget import SlicePropagatorWidget

__all__ = ["SlicePropagatorWidget"]