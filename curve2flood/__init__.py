"""Curve2Flood: flood maps from ARC rating curves (VDT databases or curve files)."""
from typing import TYPE_CHECKING

from ._log import LOG
from .config import Config
from .core import Curve2Flood_MainFunction, remove_cells_not_connected

if TYPE_CHECKING:
    # For type checkers and the API docs: it is imported when first asked for (see __getattr__)
    from .spreaders.fldpln.library import build_fldpln_library

__all__ = ["LOG", "Config", "Curve2Flood_MainFunction", "build_fldpln_library", "remove_cells_not_connected"]


def __getattr__(name):
    # The library builder, and the pandas it reads with, are imported when first asked for
    if name == "build_fldpln_library":
        from .spreaders.fldpln.library import build_fldpln_library
        return build_fldpln_library
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
