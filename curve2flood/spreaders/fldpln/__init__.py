"""FLDPLN: build a floodplain library, and map floods with it.

FLDPLN (Kastens 2008, *Some New Developments on Two Separate Topics:
Statistical Cross Validation and Floodplain Mapping*, chapter 3) asks, for
every stream pixel, which floodplain pixels its water can reach and how deep
the water must be at the stream pixel to get there: the depth to flood (DTF).
Water fills the filled DEM upstream (backfill) and spills over low divides
into neighbouring valleys (spillover).  The library holds one row per
(flood source pixel ``FSP``, floodplain pixel ``FPP``, ``DTF``, ``fill``),
where ``fill`` is how far the filled DEM sits above the DEM at ``FPP``,
and ``segment``, the stretch of stream the ``FSP`` belongs to.  Every pixel
of a segment starts at the same depth, so the library is exact for one water
depth per segment, which is all mapping an event then needs.

Two solvers build the library:

``solver="iterative"`` (default)
    The depth-stepped algorithm, raising the water ``dh`` at a time.  Without
    ``iterative_spill`` water spills over a divide once per step, as FLDPLN
    recommends for coarse DEMs (30 m); a larger ``dh`` damps spillover more.

``solver="exact"``
    DTF is the smallest total climb along any 8-connected path from the stream
    to the pixel (downhill steps are free, but for ``spill_decay``), found in
    one Dijkstra sweep.  This
    is the limit of the depth-stepped algorithm with ``iterative_spill`` as
    ``dh`` goes to 0: water keeps spilling and running downhill for as long as
    it can. 

The exact solver solves each stream segment in a window around it that grows
when the floodplain reaches its edge, so the result never depends on the
window size.  A library build solves with the iterative solver on the whole
raster, in arrays each thread keeps from one segment to the next
(``Workspace``), which gives what the windows give without solving any
segment twice.

Flow directions use WhiteboxTools D8 codes, and the stream table has columns
``start_pixel, end_pixel, length, stream_id``.  Pixels are flat row-major
indices, ``row * ncols + col``, of the grid a Parquet library records in its
metadata (``library_metadata``), so a flood map needs no DEM to place them.
"""
from .library_writer import library_metadata
from .mapping import library_sources, make_fldpln_flood_map

__all__ = ["build_fldpln_library", "library_metadata", "library_sources", "make_fldpln_flood_map"]


def __getattr__(name):
    # The builder, and the pandas it reads with, are imported when first asked for: mapping a flood needs neither
    # Saves import time
    if name == "build_fldpln_library":
        from .library import build_fldpln_library
        return build_fldpln_library
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")