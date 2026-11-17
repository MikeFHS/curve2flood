from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

from . import library_writer
from .solvers import _next_downstream

if TYPE_CHECKING:
    import pandas as pd

# Rows of a Parquet library read at a time.  A quarter of a row group: pyarrow keeps the buffers it decoded a batch
# in for the next. This saves memory, at no cost in time.
_READ_ROWS = 1 << 18

@njit(cache=True, nogil=True)
def _walk_reaches(starts: np.ndarray, lengths: np.ndarray, fdr: np.ndarray, nrows: int,
                  ncols: int) -> tuple[np.ndarray, np.ndarray]:
    """Each reach's stream pixels, walked ``lengths[i]`` pixels down the FDR from ``starts[i]`` (or to a sink or the
    raster's edge): all of them, and where each reach's begin (``offsets``, one more than the reaches)."""
    total = 0
    for i in range(starts.size):
        total += max(lengths[i], 0)

    pixels = np.empty(total, dtype=np.int64)
    offsets = np.zeros(starts.size + 1, dtype=np.int64)
    n = 0
    for i in range(starts.size):
        pixel = starts[i]
        for _ in range(lengths[i]):
            pixels[n] = pixel
            n += 1
            pixel = _next_downstream(pixel, fdr, nrows, ncols)
            if pixel == -1:
                break
        offsets[i + 1] = n

    return pixels[:n], offsets

@dataclass(frozen=True)
class LibrarySources:
    """What a flood map needs of a library before its depths, the same for every flow (see ``library_sources``).

    ``key`` is the column each row's depth is looked up by: ``segment``, or
    ``FSP`` for a library built before reaches were split into segments.
    ``pixels`` holds each segment's stream pixels, every ``(segment, FSP)``
    once (only ``FSP`` without segments).  ``own_fsp`` are the stream pixels
    with a row for their own cell, ascending, and ``own_fill`` that cell's
    fill.  ``fill_cells`` are the cells whose fill is above 0 (or NaN),
    ascending, and ``fill_values`` their fill.
    """
    key: str
    pixels: pl.DataFrame
    own_fsp: np.ndarray
    own_fill: np.ndarray
    fill_cells: np.ndarray
    fill_values: np.ndarray


def library_sources(fldpln_library: pl.LazyFrame | str | os.PathLike, ncells: int,
                    fill_below_dem: np.ndarray | None = None, scratch: np.ndarray | None = None) -> LibrarySources:
    """Read a library's stream pixels and fills, once whatever the flows (see ``make_fldpln_flood_map``).

    A cell's fill is the same in every row, so it is read here, a cell at a
    time, and the depths need only ``(key, FPP, DTF)``.  ``fill_below_dem``,
    a float32 raster of ``ncells`` cells, gets ``min(fill, 0)`` at every cell
    of the library: how far the filled DEM sits below the DEM, for the water
    surface.  ``scratch``, float32 of ``ncells`` cells, is worked in (made
    when None) and left with every cell's fill, 0 off the library.
    """
    columns = _library_columns(fldpln_library)
    if 'fill' not in columns:
        raise ValueError("The FLDPLN library has no fill column. Rebuild it with build_fldpln_library, "
                         "which now takes the DEM the floods are mapped on.")

    key = 'segment' if 'segment' in columns else 'FSP'
    group = ['segment', 'FSP'] if key == 'segment' else ['FSP']
    fill_of = np.zeros(ncells, np.float32) if scratch is None else scratch
    fill_of[:] = 0.0
    below = np.empty(0, np.float32) if fill_below_dem is None else fill_below_dem.reshape(-1)
    last = np.zeros(3, np.int64)
    # Where each part's runs start and its own rows are: few, so only the first pages are ever touched
    starts = own = np.empty(0, np.int32)
    pairs = {column: [] for column in group}
    own_fsp = []
    for part in _library_parts(fldpln_library, group + ['FPP', 'fill']):
        fsp, fill = part['FSP'], part['fill']
        if starts.size < fsp.size:
            starts, own = np.empty(fsp.size, np.int32), np.empty(fsp.size, np.int32)
        n_starts, n_own = _scan_sources(part[key], fsp, part['FPP'], fill, _scale(fill), last, starts, own, fill_of,
                                        below)
        for column in group:
            pairs[column].append(part[column][starts[:n_starts]])
        own_fsp.append(fsp[own[:n_own]])

    pixels = pl.DataFrame({column: np.concatenate(pairs[column]) if pairs[column] else np.empty(0, np.int32)
                           for column in group}).unique()
    own_fsp = np.unique(np.concatenate(own_fsp)) if own_fsp else np.empty(0, np.int32)
    # A cell's fill is the same in every row, its own row's too
    own_fill = fill_of[own_fsp]
    raised = fill_of <= 0
    np.logical_not(raised, out=raised)
    fill_cells = np.flatnonzero(raised)
    del raised
    return LibrarySources(key, pixels, own_fsp, own_fill, fill_cells, fill_of[fill_cells])


@njit(cache=True, nogil=True)
def _scan_sources(segment, fsp, fpp, fill, scale, last, starts, own, fill_of, below):
    """``library_sources`` of one part of the library's rows: where each run of rows of one ``(segment, FSP)``
    starts, into ``starts``, and which rows are their source's own cell's (``FPP == FSP``), into ``own``; returns how
    many of each.  ``last`` is the previous part's last ``(segment, FSP)`` and 1, or 0s before the first part, and
    is moved on to this part's.

    Each row's fill, float32 ``fill / scale``, goes to its cell of
    ``fill_of`` unless 0 (``fill_of`` starts at 0), and of ``below``, unless
    empty, as ``min(fill, 0)``.
    """
    zero = np.float32(0.0)
    n_starts = 0
    n_own = 0
    for i in range(fsp.size):
        source = fsp[i]
        k = segment[i]
        if last[2] == 0 or source != last[0] or k != last[1]:
            starts[n_starts] = i
            n_starts += 1
            last[0] = source
            last[1] = k
            last[2] = 1

        cell = fpp[i]
        if cell == source:
            own[n_own] = i
            n_own += 1

        raw = fill[i]
        if raw != 0:
            fill_of[cell] = np.float32(raw / scale)
        if below.size > 0:
            below[cell] = min(zero, np.float32(raw / scale))

    return n_starts, n_own


@njit(cache=True, nogil=True)
def _deepest_heads(best: np.ndarray, key: np.ndarray, fpp: np.ndarray, dtf: np.ndarray, scale: float,
                   dof: np.ndarray):
    """Raise ``best[p]`` to the head of each library row ``(key, p, DTF)`` above 0: ``dof[key] - DTF``, how deep the
    water stands over ``p``'s filled surface, in float32 (``DTF`` is float32 ``dtf / scale``).

    ``dof`` is NaN for a key without a depth, and a key outside it has none
    either.  The order of the rows doesn't matter.
    """
    zero = np.float32(0.0)
    for i in range(key.size):
        k = key[i]
        if k < 0 or k >= dof.size:
            continue

        head = dof[k] - np.float32(dtf[i] / scale)
        if head > zero and head > best[fpp[i]]:
            best[fpp[i]] = head


def _add_fill(best: np.ndarray, sources: LibrarySources):
    """Add to the heads in ``best`` (``_deepest_heads``; -inf where none) their cell's fill where it is above 0: the
    depth over the DEM.  A NaN fill makes the depth NaN.  Each cell's fill is the same in every row, and float32
    rounding keeps the order of sums, so this is the deepest of its rows' ``head + fill``."""
    heads = best[sources.fill_cells]
    reached = heads > -np.inf
    best[sources.fill_cells[reached]] = heads[reached] + sources.fill_values[reached]


def make_fldpln_flood_map(
        vdt_df: pl.DataFrame,
        fdr: np.ndarray,
        fldpln_library: pl.LazyFrame | str | os.PathLike,
        stream_info_df: pl.DataFrame | pd.DataFrame,
        fill_below_dem: np.ndarray | None = None,
        min_depth: float = 0.0,
        sources: LibrarySources | None = None) -> np.ndarray:
    """Flood depth raster for one event, from VDT stages and a floodplain library.

    1. Along each reach (its pixels in ``stream_info_df``, walked down the
       flow directions), the stage of stream pixel ``s`` (its water depth over
       the filled DEM) is the VDT's water surface minus the filled DEM.  The
       filled DEM there is the VDT's ``Elev`` (the DEM under the pixel, the
       bed ARC's depths are measured from) plus the library's ``fill`` of the
       pixel's own cell, its row with ``FPP == FSP``, so the filled DEM itself
       is never read.  A pixel without that row counts as having no VDT
       value, and so does a pixel without a VDT row: it has no stage.
    2. Each segment ``R`` floods at one depth ``DoF(R)``, as the library
       assumes: the median of its stream pixels' stages, every pixel counted
       once, however many floodplain cells it is the source of.  The depth is
       the one the VDT gives the segment, not one chosen for what it floods.
       In a segment with VDT values, a pixel without one takes the stage
       interpolated between the segment's own pixels with one, and held
       beyond the first and last, so ``DoF(R)`` never leaves the range of
       the stages ARC gave the segment. A library without a ``segment`` column (built before reaches were
       split) floods each reach at one depth.
    3. Each library row ``(s, p, DTF, fill)`` whose water reaches ``p``'s
       filled surface (``DoF(R) - DTF(s, p) > 0``, ``R`` being ``s``'s
       segment) gives ``p`` the depth ``DoF(R) - DTF(s, p) + max(fill(p), 0)``:
       the water above the filled DEM, plus how far the filled DEM sits above
       the DEM there.  Water that doesn't reach a filled pit's rim never spills
       into it, however deep the pit.  ``p`` keeps the deepest.  Where the DEM
       sits above the filled DEM (bathymetry raised above the ground the
       library was routed on), the fill is 0: the water was routed across
       the filled surface, so it covers the cell wherever it reaches that
       surface, and the depth is measured from it.  Otherwise a raised cell
       inside the channel, routed through, read as a dry hole.
    4. A cell floods where its depth is above 0 and at least ``min_depth``.
       Water that only just tops a low rise beside a stream on a slope runs
       down everything below it in a film: the library's DTF to all of it is
       that of the rise, so each cell gets the same few millimetres.

    Returns the depth where the cell floods, NaN elsewhere, on ``fdr``'s grid;
    no DEM is needed.  ``vdt_df`` needs ``COMID``, ``Row``, ``Col``, ``Elev``
    and ``WSE``.  ``fldpln_library`` is a Parquet library's path or a
    LazyFrame (a CSV library), read a part at a time: its stream pixels and
    fills first (``library_sources``, unless ``sources`` gives them, read
    once for every flow), and then its depths.  ``fill_below_dem``, a float32
    raster, gets ``min(fill, 0)`` at every cell of the library: how far the
    filled DEM sits below the DEM, for the water surface; with ``sources``,
    ``library_sources`` wrote it.  ``stream_info_df`` is a polars or pandas
    DataFrame.
    """
    nrows, ncols = fdr.shape
    # C order: the library's flat pixel ids index it
    depth_array = np.full((nrows, ncols), np.nan, dtype=np.float32)

    # Each segment's stream pixels, and the fill of their own cells.  Every
    # stream pixel of a reach in the library is the source of at least its
    # own cell, so it has a row there, which gives its fill.
    if sources is None:
        sources = library_sources(fldpln_library, depth_array.size, fill_below_dem, scratch=depth_array.reshape(-1))
        depth_array.fill(np.nan)

    if sources.pixels.is_empty():
        return depth_array

    key, library_pixels, own_fsp, own_fill = sources.key, sources.pixels, sources.own_fsp, sources.own_fill

    # The VDT's WSE and Elev at its stream pixels, sorted by pixel; a pixel listed twice keeps its last row
    vdt_fsp = vdt_df['Row'].to_numpy().astype(np.int64) * ncols + vdt_df['Col'].to_numpy()
    order = np.argsort(vdt_fsp, kind='stable')
    vdt_fsp, vdt_wse, vdt_elev = vdt_fsp[order], vdt_df['WSE'].to_numpy()[order], vdt_df['Elev'].to_numpy()[order]
    stream_ids = list(set(vdt_df['COMID']))

    row_of = {stream_id: i for i, stream_id in enumerate(stream_info_df['stream_id'].to_numpy().tolist())}
    for stream_id in stream_ids:
        if stream_id not in row_of:
            raise ValueError(f"Stream ID {stream_id} not found in stream_info_df.")

    # Walk each reach's stream pixels, with the VDT's WSE and Elev where it has them
    rows = np.array([row_of[s] for s in stream_ids], dtype=np.int64)
    starts = stream_info_df['start_pixel'].to_numpy().astype(np.int64)[rows]
    lengths = stream_info_df['length'].to_numpy().astype(np.int64)[rows]
    pixels, offsets = _walk_reaches(starts, lengths, fdr.ravel(), nrows, ncols)
    at = np.searchsorted(vdt_fsp, pixels, side='right') - 1
    known = at >= 0
    known[known] = vdt_fsp[at[known]] == pixels[known]
    wse = np.full(pixels.size, np.nan)
    wse[known] = vdt_wse[at[known]]
    elevation = np.full(pixels.size, np.nan)
    elevation[known] = vdt_elev[at[known]]
    # The filled DEM: the VDT's Elev plus the fill of the pixel's own cell
    fill = np.full(pixels.size, np.nan)
    if own_fsp.size:
        at = np.minimum(np.searchsorted(own_fsp, pixels), own_fsp.size - 1)
        has_fill = own_fsp[at] == pixels
        fill[has_fill] = own_fill[at[has_fill]]

    # Each stream pixel's depth over the filled DEM, missing without a VDT value
    pixel_stage = wse - (elevation + fill)
    measured = ~np.isnan(pixel_stage)
    if not measured.any():
        return depth_array

    # A pixel on two reaches keeps its deeper stage
    stage = (pl.DataFrame({'FSP': pixels, 'DoF': pl.Series(pixel_stage, dtype=pl.Float32, nan_to_null=True),
                           'measured': measured},
                          schema={'FSP': pl.Int32, 'DoF': pl.Float32, 'measured': pl.Boolean})
             .group_by('FSP').agg(pl.col('DoF').max(), pl.col('measured').max()))

    # Each stream pixel's reach and its place along it, from upstream: a
    # segment is a piece of one reach
    sizes = np.diff(offsets)
    place = pl.DataFrame({'FSP': pixels, 'reach': np.repeat(np.array(stream_ids, dtype=np.int64), sizes),
                          'position': np.arange(pixels.size) - np.repeat(offsets[:-1], sizes)},
                         schema={'FSP': pl.Int32, 'reach': pl.Int64, 'position': pl.Int32}
                         ).unique('FSP', keep='first', maintain_order=True)
    if key == 'FSP':
        # A library built before reaches were split into segments: each reach is one
        reach_of = place.select('FSP', segment='reach')
        library_pixels = library_pixels.join(reach_of, on='FSP', how='inner')

    library_pixels = (library_pixels.join(stage, on='FSP', how='inner')
                      .join(place.select('FSP', 'position'), on='FSP', how='inner')
                      .sort(['segment', 'position']))
    if library_pixels.is_empty():
        return depth_array
    
    # A pixel without a VDT value takes its stage from its own segment's pixels with one; a segment without any has
    # none, and floods nothing
    stages = _interpolate_within_segments(library_pixels['segment'].to_numpy(), library_pixels['position'].to_numpy(),
                                          library_pixels['measured'].to_numpy(), library_pixels['DoF'].to_numpy())
    segments, depths = _median_by_group(library_pixels['segment'].to_numpy(), stages)
    depths = depths.astype(np.float32)
    if key == 'FSP':
        df = pl.DataFrame({'segment': segments, 'DoF': depths},
                          schema_overrides={'segment': library_pixels.schema['segment']})
        df = reach_of.join(df, on='segment', how='inner')
        segments, depths = df['FSP'].to_numpy(), df['DoF'].to_numpy()

    # Each key's depth, looked up by the key itself
    valid = segments >= 0
    if not valid.any():
        return depth_array
    
    dof = np.full(int(segments[valid].max()) + 1, np.nan, dtype=np.float32)
    dof[segments[valid]] = depths[valid]

    # Read the library a part at a time and keep only each pixel's deepest
    # water so far, in ``depth_array`` itself: joined whole, the library and
    # its join were several times the size of every raster in the run.
    best = depth_array.reshape(-1)
    best.fill(-np.inf)
    for part in _library_parts(fldpln_library, [key, 'FPP', 'DTF']):
        dtf = part['DTF']
        _deepest_heads(best, part[key], part['FPP'], dtf, _scale(dtf), dof)

    _add_fill(best, sources)
    # NaN where the depth isn't above 0 and at least min_depth (a NaN depth is NaN already)
    min_depth = np.float32(min_depth)
    if min_depth > 0:
        best[best < min_depth] = np.nan
    elif min_depth <= 0:
        best[best <= 0] = np.nan
    else:
        best.fill(np.nan)

    return depth_array

def _median_by_group(group: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each group's median: its middle value, or the lower of the two middle ones for an even count.

    Returns the groups, sorted, and their medians.
    """
    order = np.lexsort((value, group))
    group, value = group[order], value[order]
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    sizes = np.diff(np.r_[starts, len(group)])
    return group[starts], value[starts + (sizes - 1) // 2]

def _interpolate_within_segments(segment: np.ndarray, position: np.ndarray, measured: np.ndarray,
                                 stage: np.ndarray) -> np.ndarray:
    """Each pixel's stage, with the unmeasured pixels' taken from the measured pixels of their own segment.

    The pixels are sorted by segment, then by ``position`` along the stream.
    Between two measured pixels of a segment the stage is interpolated
    linearly by position; before the first and after the last it is that
    pixel's.  A segment without a measured pixel keeps ``stage``.
    """
    n = len(segment)
    index = np.arange(n)
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    sizes = np.diff(np.r_[starts, n])
    first, end = np.repeat(starts, sizes), np.repeat(starts + sizes, sizes)
    # The nearest measured pixel of the same segment before or at, and after or at, each pixel
    before = np.maximum.accumulate(np.where(measured, index, -1))
    after = np.minimum.accumulate(np.where(measured, index, n)[::-1])[::-1]
    before = np.where(before >= first, before, after)
    after = np.where(after < end, after, before)
    stage = np.asarray(stage, dtype=float)
    out = stage.copy()
    known = ~measured & (before < end)
    b, a = before[known], after[known]
    span = (position[a] - position[b]).astype(float)
    t = np.divide(position[known] - position[b], span, out=np.zeros(len(b)), where=span > 0)
    out[known] = stage[b] + t * (stage[a] - stage[b])
    return out

def _library_columns(library: pl.LazyFrame | str | os.PathLike) -> list[str]:
    """The columns of a library: a LazyFrame, or a Parquet file's path."""
    if isinstance(library, pl.LazyFrame):
        return library.collect_schema().names()
    return pq.read_schema(library).names

def _library_parts(library: pl.LazyFrame | str | os.PathLike, columns: list[str]):
    """The library's ``columns`` a part at a time, as NumPy arrays by name: ``_READ_ROWS`` rows of a Parquet file,
    ``_LIBRARY_SLICE_ROWS`` of a LazyFrame.  DTF and fill are float32 metres, or int32 hundredths of a metre: their
    values are ``_metres`` of them.

    A Parquet file is read through pyarrow, a batch at a time: two to five
    times as fast as slicing ``pl.scan_parquet``, and half the memory.
    """
    if isinstance(library, pl.LazyFrame):
        rows = library_writer._LIBRARY_SLICE_ROWS
        n = library.select(pl.len()).collect().item()
        for offset in range(0, n, rows):
            part = library.slice(offset, rows).select(columns).collect()
            part = part.with_columns([pl.col(c).cast(pl.Float32) for c, d in part.schema.items() if d == pl.Decimal])
            yield {c: part[c].to_numpy() for c in columns}
        return
    
    for batch in pq.ParquetFile(library, pre_buffer=False).iter_batches(batch_size=_READ_ROWS, columns=columns):
        yield {name: _column_values(batch.column(i)) for i, name in enumerate(batch.schema.names)}

def _column_values(column: pa.Array) -> np.ndarray:
    """A library column's values as NumPy: a decimal of hundredths stored in 32 bits (``DECIMAL_COLS``) as its int32
    hundredths, and any other decimal cast to float32 by polars.

    ``float32(integer / 100)``, worked out where a value is used, is what
    polars' cast gives for every one of them (all twenty million of
    decimal32(7, 2) were checked), with no float32 copy of the column.
    Numbers without nulls are read in place, from the column's buffer:
    pyarrow's ``to_numpy`` imports pandas, a third of a second.
    """
    kind = column.type
    if pa.types.is_decimal(kind):
        if kind.bit_width == 32 and kind.scale == 2 and column.null_count == 0:
            return _in_place(column, np.dtype(np.int32))
        
        return pl.from_arrow(column).cast(pl.Float32).to_numpy()
    
    if column.null_count == 0 and (pa.types.is_integer(kind) or pa.types.is_floating(kind)):
        letter = "f" if pa.types.is_floating(kind) else "i" if pa.types.is_signed_integer(kind) else "u"
        return _in_place(column, np.dtype(f"{letter}{kind.bit_width // 8}"))
    
    return column.to_numpy(zero_copy_only=False)

def _in_place(column: pa.Array, dtype: np.dtype) -> np.ndarray:
    """A fixed-width column without nulls as NumPy ``dtype``, read from its data buffer."""
    return np.frombuffer(column.buffers()[1], dtype=dtype)[column.offset:column.offset + len(column)]

def _scale(values: np.ndarray) -> float:
    """What a DTF or fill column of ``_library_parts`` is divided by to give metres: 100 for integer hundredths."""
    return 100.0 if values.dtype.kind in "iu" else 1.0

def _metres(values: np.ndarray) -> np.ndarray:
    """A DTF or fill column of ``_library_parts`` in metres, as float32: worked out in float64 and rounded once."""
    return (values / _scale(values)).astype(np.float32)
