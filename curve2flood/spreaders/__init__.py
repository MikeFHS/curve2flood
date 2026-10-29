"""The flood mappers: kernel-weighted, multi-point interpolation and FLDPLN."""
from .fldpln import make_fldpln_flood_map
from .kernel_weighted import create_kernel_weighted_spread_map, spread_stream_cells

__all__ = [
    "build_fldpln_library",
    "create_kernel_weighted_spread_map",
    "filter_outliers",
    "make_fldpln_flood_map",
    "multi_point_interpolation",
    "spread_stream_cells",
]


def __getattr__(name):
    # Imported when first asked for: the multi-point mapper needs pandas and SciPy, and the FLDPLN library builder
    # pandas, which the other mappers don't
    if name in ("filter_outliers", "multi_point_interpolation"):
        from . import multi_point
        return getattr(multi_point, name)
    
    if name == "build_fldpln_library":
        from .fldpln.library import build_fldpln_library
        return build_fldpln_library
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
