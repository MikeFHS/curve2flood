import contextlib
import importlib.metadata
import itertools
import math
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm
from numba import njit, prange
from osgeo import gdal

from curve2flood._log import LOG
from curve2flood._raster import Grid, decoding_threads, open_raster, read_array
from curve2flood.config import is_parquet

from .library_writer import METADATA_VERSION, LibraryWriter
from .solvers import (
    SOLVERS,
    _neighbour_bits_of_row,
    _next_downstream,
    _solve_exact,
    _solve_iterative,
)
from .sparsefill import SparseFill
from .workspace import Workspace

# The most threads a parallel build takes by default: each holds a copy of the
# solver's arrays of the whole raster, and more than two hardly build faster
_DEFAULT_THREADS = 4

# Window half-widths tried in turn, in cells, before falling back to the raster.
_WINDOW_MARGINS = (128, 384, 1152)

def build_fldpln_library(
    dem: str,
    filled_dem: str,
    stream_info_file: str,
    flow_direction_file: str,
    library_file: str,
    dh: float = 0.5,
    fldmn: float = 0.1,
    fldmx: float = 25.0,
    iterative_spill: bool = False,
    vdt_file: str | None = None,
    bg_mask: np.ndarray | None = None,
    stream_ids: list[int] | None = None,
    global_max_wse: float = 0.0,
    bg: float = -9999,
    parallel: bool = False,
    pbar: bool = True,
    processes: int | None = None,
    spill_decay: float = 0.0,
    solver: str = "iterative",
    max_cells_per_segment: int | None = None,
    max_reach_length: float | None = 5000.0,
    max_profile_deviation: float | None = 1.0,
    water_mask: np.ndarray | None = None
):
    """Build the FLDPLN floodplain library for every stream segment and save it.

    Each row is a stream pixel ``FSP`` (Flood Source Pixel), a cell it floods ``FPP`` (FloodPlain Pixel), the Depth
    to Flood ``DTF``, the ``segment`` the stream pixel belongs to and the
    cell's ``fill``: the filled DEM minus ``dem``.  Cells where ``dem`` has no
    data never flood, and are left out.

    Every source of a segment starts at the same depth, so each cell keeps the
    one stream pixel that reaches it at the smallest depth: the library is
    exact for one depth per segment, which is how the mapper floods it.
    Reaches are first cut where the filled DEM under them bends or steps (with
    ``max_profile_deviation``) and into pieces no longer than
    ``max_reach_length``, each solved and flooded on its own.

    Each segment's rows are written as soon as it is solved, sorted by
    ``FSP`` and then ``FPP``, so the build holds only a few segments' rows
    however big the library.  The file is written as ``<library_file>.partial``
    and renamed once complete: a build that fails leaves no library.

    Parameters
    ----------
    dem: str
        The DEM floods are mapped on (with the bathymetry, if any).  Used to
        derive the ``fill`` column.
    filled_dem: str
        Filled (depression-free) DEM raster, the one the flow directions were
        made from.
    stream_info_file: str
        CSV or Parquet table of the segments, with columns ``start_pixel``,
        ``end_pixel``, ``length`` (pixels) and ``stream_id``.
    flow_direction_file: str
        D8 flow-direction raster with WhiteboxTools codes.
    library_file: str
        Where to save the library: Parquet (``.parquet``/``.pq``, compact) or CSV.
    dh: float
        Depth step of the iterative solver (m), 0.5 by default.  Ignored by the
        exact solver.
    fldmn: float
        Iterative solver only: the least depth (m) a flooded pixel counts as
        when it decides where a spill stops, as the KU FLDPLN's minimum flood
        depth.  0.1 by default.  DTFs are stored as found, from 0 at the
        stream: the KU FLDPLN raises those below ``fldmn`` to it, which mapped
        every stream pixel and flat channel bed ``fldmn`` too shallow.
    fldmx: float
        Largest depth to flood to (m).  25 by default.
    iterative_spill: bool
        Iterative solver only: keep spilling until the floodplain stops growing.
        Recommended for fine DEMs (10 m or less); off by default.
    vdt_file: str | None
        A VDT database.  Each segment then floods only to the deepest water its
        last ``wse_*`` column gives (at most ``fldmx``), and segments without
        VDT cells are skipped.
    bg_mask: np.ndarray | None
        Cells that never flood, such as lakes; same shape as the DEM.
    stream_ids: list[int] | None
        Only build these segments.
    global_max_wse: float
        Highest water-surface elevation allowed; 0 for no limit.
    bg: float
        No-data value of the filled DEM.
    parallel: bool
        Solve several segments at once.
    pbar: bool
        Show a progress bar.
    processes: int | None
        Number of threads when ``parallel``; defaults to the CPU count, at
        most 4. Tests indicate that more threads do not help and consume more memory.
    spill_decay: float
        Head (m) water spilling over a divide loses per pixel down its path. 
        The path's ``j``-th pixel floods
        from ``j * spill_decay`` more DTF than its first, it backfills with
        that much less water, and the path ends where the water runs out.
        Without it a spill keeps its whole depth however far it runs, so on
        flat ground it floods everything below its source's water surface as
        a level pool.  The exact solver takes it off the water's head on every
        step down, to lower ground whichever way or along the flow directions,
        as a spill loses it down its path (and from the first step down, which
        the depth-stepped spill takes for free).  0 (default) for none.
    solver: str
        ``"iterative"`` (default) or ``"exact"``; see the module docstring.
    max_cells_per_segment: int | None
        Most rows one segment may contribute.  A segment that would flood more
        stops at the depth where it fills up, so deeper floods there are mapped
        only that far: this trades the largest floods on flat ground for memory.
        None (default) for no limit.  Needs ``solver="exact"``.
    max_reach_length: float | None
        Longest segment (m), measured along the flow directions with the
        filled DEM's cell size.  A longer reach is cut into
        ``ceil(length / max_reach_length)`` segments of about equal length.
        5000 (default). None keeps each reach whole.
    max_profile_deviation: float | None
        Also cut each reach where the filled DEM under the stream bends or
        steps, so that the bed is about straight along every segment: its
        profile along the river is simplified top-down (Douglas-Peucker)
        until no stream pixel is more than this many metres above or below
        its piece's straight line.  The pieces are then cut by
        ``max_reach_length``.  One depth per segment fits a straight bed
        best: the stage jumps at a step and where a filled flat begins.
        None makes no such cuts.
    water_mask: np.ndarray | None
        The river's water, such as the bathymetry water mask; same shape as
        the DEM, positive for water.  Each segment's ``TD(R)``, the cells below
        its end that its water may not spill into, is then the channel below
        it and not only the stream line: every water cell whose D8 path first
        meets the stream network inside ``TD(R)`` joins it.  None (default) keeps
        the line alone.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")
    if max_cells_per_segment is not None and (int(max_cells_per_segment) != max_cells_per_segment or max_cells_per_segment < 1):
        raise ValueError(f"max_cells_per_segment must be a positive whole number or None, got {max_cells_per_segment!r}.")
    if max_cells_per_segment is not None and solver != "exact":
        raise ValueError("max_cells_per_segment needs solver='exact'.")
    if max_reach_length is not None and not max_reach_length > 0:
        raise ValueError(f"max_reach_length must be positive or None, got {max_reach_length!r}.")
    if max_profile_deviation is not None and not max_profile_deviation > 0:
        raise ValueError(f"max_profile_deviation must be positive or None, got {max_profile_deviation!r}.")
    split = max_reach_length is not None or max_profile_deviation is not None
    max_cells = int(max_cells_per_segment or 0)

    if is_parquet(stream_info_file):
        stream_info = pd.read_parquet(stream_info_file)
    else:
        stream_info = pd.read_csv(stream_info_file)

    if stream_info.shape[1] != 4:
        raise ValueError(f"The stream info file {stream_info_file} has {stream_info.shape[1]} columns, not the 4 of "
                         "start pixel, end pixel, length and stream id, in that order.")
    # The whole network, for the channel cells: water draining to a reach left out below is still that reach's
    network = stream_info.values.astype(np.int64)

    if stream_ids is not None:
        stream_info = stream_info[stream_info.iloc[:, 3].isin(stream_ids)]

    # The grid the library's pixel ids number, which the DEM and the flow directions must share
    grid = Grid.of(filled_dem)
    grid.check_matches(Grid.of(dem), "The filled DEM", "the DEM")
    grid.check_matches(Grid.of(flow_direction_file), "The filled DEM", "the flow directions")
    filled_dem_array = read_array(filled_dem, np.float32)
    flow_direction_array = read_array(flow_direction_file, np.uint8)

    if bg_mask is not None:
        # A 0/1 integer mask would index rows 0 and 1 instead of the masked cells
        bg_mask = np.asarray(bg_mask, dtype=bool)
        if bg_mask.shape != filled_dem_array.shape:
            raise ValueError(f"bg_mask has shape {bg_mask.shape}, but the filled DEM has {filled_dem_array.shape}.")

    channel = None
    if water_mask is not None:
        water_mask = np.asarray(water_mask)
        if water_mask.shape != filled_dem_array.shape:
            raise ValueError(f"water_mask has shape {water_mask.shape}, but the filled DEM has "
                             f"{filled_dem_array.shape}.")
        channel = _channel_cells(flow_direction_array, network, water_mask > 0)  # NaN and no-data aren't water
        LOG.info(f"TD(R) takes in the channel: {channel[2].size:,} water cells drain first to "
                 f"{channel[0].size:,} stream pixels.")

    stream_ids = stream_info.iloc[:, 3].unique()
    vdt_depth: dict[int, float] = {}
    if vdt_file is None:
        max_depths = [fldmx] * len(stream_ids)
    else:
        # Read only the columns used: the VDT carries ~120 more
        vdt_is_parquet = is_parquet(vdt_file)
        if vdt_is_parquet:
            vdt_columns = pq.read_schema(vdt_file).names
        else:
            vdt_columns = pd.read_csv(vdt_file, nrows=0).columns
        # Find last wse_* column
        wse_cols = [col for col in vdt_columns if col.startswith('wse_')]
        if not wse_cols:
            raise ValueError(f"The VDT database {vdt_file} has no wse_* columns.")
        
        max_wse_col = wse_cols[-1]
        usecols = ['COMID', 'Row', 'Col', max_wse_col]
        if vdt_is_parquet:
            vdt_df = pd.read_parquet(vdt_file, columns=usecols)
        else:
            vdt_df = pd.read_csv(vdt_file, usecols=usecols)

        vdt_df = vdt_df[vdt_df['COMID'].isin(stream_ids)]
        # making a copy to resolve a fragmentation warning from pandas
        vdt_df = vdt_df[usecols].copy()
        elevation = filled_dem_array[vdt_df['Row'], vdt_df['Col']]
        vdt_df['depth'] = vdt_df[max_wse_col] - elevation
        # Background cells, like masked lakes, are never flooded from, and their depth would be measured from bg
        background = elevation == bg
        if bg_mask is not None:
            background |= bg_mask[vdt_df['Row'], vdt_df['Col']]
        vdt_df = vdt_df[~background]
        ids_max_depths = vdt_df.groupby('COMID', sort=False, as_index=False)['depth'].max()
        stream_ids = ids_max_depths['COMID'].to_numpy(np.int64)
        max_depths = np.minimum(ids_max_depths['depth'].to_numpy(), fldmx)
        if split:
            # Each VDT cell's deepest water, for the segments of split reaches
            vdt_pixel = vdt_df['Row'].to_numpy(np.int64) * filled_dem_array.shape[1] + vdt_df['Col'].to_numpy(np.int64)
            vdt_depth = pd.Series(vdt_df['depth'].to_numpy(), index=vdt_pixel).groupby(level=0).max().to_dict()

    if len(stream_ids) == 0:
        LOG.warning("No stream segments found to process. Exiting.")
        return

    if pbar:
        pbar = tqdm.tqdm
    else:
        pbar = lambda x, **kwargs: x

    nrows, ncols = filled_dem_array.shape
    cell_size = _cell_size_metres(filled_dem) if split else None
    # The profiles are read before bg_mask is applied: a lake on the stream is a flat, not a gap
    segments, segment_depths, n_split = _segments(
        stream_info.values.astype(np.int64), stream_ids, max_depths, flow_direction_array.ravel(), nrows, ncols,
        max_reach_length, cell_size, vdt_depth, fldmx, filled_dem_array.ravel(), max_profile_deviation, bg)
    if n_split:
        why = []
        if max_profile_deviation is not None:
            why.append(f"where the filled DEM under them departs from a straight line by more than "
                       f"{max_profile_deviation:g} m")
        if max_reach_length is not None:
            why.append(f"into pieces of at most {max_reach_length / 1000:g} km")
        LOG.info(f"Cut {n_split} of {len(stream_ids)} reaches {' and '.join(why)}: {len(segments)} segments in all.")
    if bg_mask is not None:
        filled_dem_array[bg_mask] = bg
    # One segment table for every task, not a copy per segment
    args = [
        (
            int(segment),
            segments,
            dh,
            fldmn,
            segment_depths[i],
            iterative_spill,
            global_max_wse,
            bg,
            spill_decay,
            solver,
            max_cells
        )
        for i, segment in enumerate(segments[:, 3])
    ]
    # Sort args from largest max depth to smallest, (longest time to shortest)
    args.sort(key=lambda x: x[4], reverse=True)

    if processes is None:
        processes = max(min(os.cpu_count() or 1, _DEFAULT_THREADS, len(args)), 1)
        if processes == 1:
            parallel = False

    fill_depth = _fill_depth(dem, filled_dem_array, bg)

    # The iterative solver solves every segment on the whole raster, each
    # thread with its own workspace (see Workspace); the rasters with the
    # windows' halo are shared
    halo = None
    if solver == "iterative":
        halo_fil = _crop(filled_dem_array, 0, nrows, 0, ncols, bg).ravel()
        # The filled DEM inside the halo stands in for it: the workspace solves on the halo's, and the rest reads
        # only its shape
        filled_dem_array = halo_fil.reshape(nrows + 2, ncols + 2)[1:-1, 1:-1]
        halo_fdr = _crop(flow_direction_array, 0, nrows, 0, ncols, np.uint8(0)).ravel()
        halo = (halo_fil, halo_fdr, *_neighbour_bits_parallel(halo_fil, halo_fdr, ncols + 2, bg),
                _channel_lookup(channel, nrows * ncols))
    local = threading.local()

    def solve(task):
        workspace = getattr(local, "workspace", None)
        if workspace is None and halo is not None:
            workspace = local.workspace = Workspace(*halo)
        return _segment_rows(filled_dem_array, flow_direction_array, *task, channel=channel, workspace=workspace)

    # Each segment's rows are written as soon as they are solved, and in the
    # order of args however many threads solve them: holding every segment's
    # rows to sort them together took 12 bytes a row, and twice that to sort,
    # which on a big river's floodplain is billions of rows.  They are sorted
    # and compressed in the writer's thread, while the next segments solve
    metadata = {
        "version": METADATA_VERSION, "curve2flood": _version(), "nrows": grid.nrows, "ncols": grid.ncols,
        "geotransform": list(grid.geotransform), "crs": grid.projection,
        "cell_size_m": list(_cell_size_metres(filled_dem)),
        "build": {"dh": dh, "fldmn": fldmn, "fldmx": fldmx, "iterative_spill": iterative_spill, "solver": solver,
                  "spill_decay": spill_decay, "max_cells_per_segment": max_cells_per_segment,
                  "max_reach_length": max_reach_length, "max_profile_deviation": max_profile_deviation,
                  "global_max_wse": global_max_wse, "with_vdt": vdt_file is not None,
                  "with_bg_mask": bg_mask is not None, "with_water_mask": water_mask is not None},
    }
    limits = []
    with LibraryWriter(library_file, background=True, metadata=metadata) as writer, \
            (ThreadPoolExecutor(processes) if parallel else contextlib.nullcontext()) as pool:
        if parallel:
            results = pbar(_ordered_map(pool, solve, args, 2 * processes), total=len(args),
                           desc="Processing streams in parallel")
        else:
            results = (solve(task) for task in pbar(args, desc="Processing streams"))
        for task, (fsp, fpp, dtf, limit) in zip(args, results):
            if limit is not None:
                limits.append(limit)
            writer.write_later(_library_rows, task[0], fsp, fpp, dtf, fill_depth)

    if limits:
        LOG.info(f"{len(limits)} of {len(args)} stream segments reached max_cells_per_segment={max_cells:,}; "
                 f"their floodplains stop at depths of {min(limits):.2f}-{max(limits):.2f} m "
                 f"(median {np.median(limits):.2f} m).")
        
def _library_rows(segment: int, fsp: np.ndarray, fpp: np.ndarray, dtf: np.ndarray,
                  fill_depth: "np.ndarray | SparseFill") -> dict[str, np.ndarray]:
    """One segment's library rows, sorted by FSP and then FPP (which compresses well), with each cell's fill from
    the flat ``fill_depth`` (or ``SparseFill``).  Cells without a DEM value (a NaN fill) never flood, and are left out."""
    order = np.lexsort((fpp, fsp))
    # Looked up in the cells' own order, ascending as the solvers give them, and then put in the rows'
    fill = fill_depth[fpp][order]
    fpp = fpp[order]
    has_dem = np.isfinite(fill)
    if not has_dem.all():
        order, fpp, fill = order[has_dem], fpp[has_dem], fill[has_dem]
    return {"FSP": fsp[order], "FPP": fpp, "DTF": dtf[order],
            "segment": np.full(order.size, segment, dtype=np.int32), "fill": fill}

def _version() -> str:
    """The installed curve2flood's version, or "unknown" when it runs from a copy that isn't installed."""
    try:
        return importlib.metadata.version("curve2flood")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def _channel_cells(flow_direction: np.ndarray, stream_info: np.ndarray, water: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The water cells off the stream network, grouped by the network pixel their D8 path first meets: the
    group of pixel ``keys[i]`` is ``cells[starts[i]:starts[i + 1]]``, ``keys`` sorted.  Cells whose path meets no
    network pixel are left out."""
    nrows, ncols = flow_direction.shape
    flat_fdr = np.ascontiguousarray(flow_direction, dtype=np.uint8).ravel()
    on_network = _network_pixels(np.ascontiguousarray(stream_info, dtype=np.int64), flat_fdr, nrows, ncols)
    candidates = np.flatnonzero(np.asarray(water, dtype=bool).ravel() & ~on_network)
    hits = _first_network_pixel(candidates, flat_fdr, on_network, nrows, ncols)
    meets = hits >= 0
    hits, candidates = hits[meets], candidates[meets]
    order = np.argsort(hits, kind="stable")
    hits, cells = hits[order], candidates[order]
    first = np.flatnonzero(np.r_[True, hits[1:] != hits[:-1]]) if hits.size else np.empty(0, np.int64)
    return hits[first], np.r_[first, hits.size].astype(np.int64), cells


# ---------------------------------------------------------------------------
# The channel beside TD(R)
#
# TD(R) is a one-pixel line, but a river is wider: water that may not enter the
# stream line below its segment's end runs down the channel beside it, past the
# end, as far as the channel goes (descending is free).  Given a
# water mask, TD(R) also takes the water cells whose D8 path first meets the
# stream network inside it: the channel downstream of R, not its tributaries'.
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _network_pixels(stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    """Which pixels are on the stream network: each reach's, walked ``length`` pixels down the FDR from its start
    as ``_segment_pixels`` walks them."""
    on_network = np.zeros(nrows * ncols, dtype=np.bool_)
    for i in range(stream_info.shape[0]):
        current = stream_info[i, 0]
        length = stream_info[i, 2]
        for step in range(length):
            if step > 0:
                current = _next_downstream(current, fdr, nrows, ncols)
                if current == -1:
                    break
            on_network[current] = True

    return on_network


def _first_network_pixel(starts: np.ndarray, fdr: np.ndarray, on_network: np.ndarray, nrows: int,
                         ncols: int) -> np.ndarray:
    """The first network pixel on the D8 path from each of ``starts`` (itself if it is one), or -1 for a path that
    ends at a sink, the raster's edge or a loop first.  Each path is walked once: the cells on it remember where it
    led, in int32 when the raster's pixel ids fit, int64 otherwise."""
    dtype = np.int32 if nrows * ncols < 2 ** 31 else np.int64
    return _first_network_pixel_with(starts, fdr, on_network, nrows, ncols, np.full(nrows * ncols, -2, dtype=dtype),
                                     np.empty(nrows * ncols, dtype=dtype))

@njit(cache=True, nogil=True)
def _first_network_pixel_with(starts: np.ndarray, fdr: np.ndarray, on_network: np.ndarray, nrows: int, ncols: int,
                              found: np.ndarray, path: np.ndarray) -> np.ndarray:
    """``_first_network_pixel`` with its arrays: ``found``, -2 for every pixel, and ``path``, as long."""
    unknown, walking = -2, -3
    result = np.empty(starts.size, dtype=np.int64)
    for i in range(starts.size):
        current = starts[i]
        n = 0
        while True:
            if found[current] == walking:   # a loop in the FDR
                hit = -1
                break
            if found[current] != unknown:
                hit = found[current]
                break
            if on_network[current]:
                hit = current
                break

            found[current] = walking
            path[n] = current
            n += 1
            nxt = _next_downstream(current, fdr, nrows, ncols)
            if nxt == -1:
                hit = -1
                break
            current = nxt

        for k in range(n):
            found[path[k]] = hit

        result[i] = hit

    return result



# ---------------------------------------------------------------------------
# Reaches into segments
#
# A library is exact only for one depth per segment (every source starts at
# the same depth), and a single depth fits a long reach badly, so reaches over
# a length are cut into shorter segments, solved and mapped on their own.
# ---------------------------------------------------------------------------

def _cell_size_metres(raster: str) -> tuple[float, float]:
    """The raster's cell size ``(dx, dy)`` in metres, as Curve2Flood measures the DEM's."""
    from curve2flood.core import convert_cell_size  # core imports this module

    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    yur = gt[3]
    yll = yur - ds.RasterYSize * abs(gt[5])
    size_x, size_y = np.hypot(gt[1], gt[2]), np.hypot(gt[4], gt[5])
    if not projection:
        # No coordinate system: take the geotransform's units as metres
        return float(size_x), float(size_y)
    dx, dy = convert_cell_size(size_x, size_y, yll, yur, projection)
    return float(dx), float(dy)


def _segments(stream_info: np.ndarray, stream_ids: np.ndarray, max_depths, flat_fdr: np.ndarray,
              nrows: int, ncols: int, max_length: float | None, cell_size: tuple[float, float] | None,
              vdt_depth: dict[int, float], fldmx: float, flat_fil: np.ndarray | None = None,
              max_deviation: float | None = None, bg: float = -9999) -> tuple[np.ndarray, list[float], int]:
    """The segments to solve: ``(start_pixel, end_pixel, length, segment)`` rows numbered from 0, each one's depth
    cap, and how many reaches were split.

    A segment cut from a reach floods as deep as the deepest VDT water among its own pixels (at most ``fldmx``), or
    as deep as its reach when it has no VDT cell (the mapper floods no such segment, but the library may be mapped with
    another VDT database). ``flat_fil`` is the
    filled DEM, for ``max_deviation``; stream pixels without a value there are left out of the profile.
    """
    rows, caps = [], []
    n_split = 0
    by_id = {int(r[3]): r for r in stream_info}
    for stream_id, cap in zip(stream_ids, max_depths):
        row = by_id[int(stream_id)]
        if max_length is None and max_deviation is None:
            pieces = [(int(row[0]), int(row[1]), int(row[2]), None)]
        else:
            pixels = np.asarray(_segment_pixels(int(stream_id), stream_info, flat_fdr, nrows, ncols), dtype=np.int64)
            steps = _step_lengths(pixels, flat_fdr, *cell_size)
            elevation = None
            if max_deviation is not None:
                elevation = flat_fil[pixels].astype(np.float64)
                has_value = elevation != bg
                if not has_value.all():
                    distance = np.cumsum(steps) - steps
                    elevation = (np.interp(distance, distance[has_value], elevation[has_value])
                                 if has_value.sum() > 1 else np.zeros(len(pixels)))
            pieces = _split_reach(row, pixels, steps, max_length, elevation, max_deviation)
        n_split += len(pieces) > 1
        for start, end, length, pixels in pieces:
            piece_cap = cap
            if len(pieces) > 1:
                depths = [vdt_depth[p] for p in pixels.tolist() if p in vdt_depth]
                piece_cap = min(max(depths), fldmx) if depths else cap
            rows.append((start, end, length, len(rows)))
            caps.append(piece_cap)
    return np.array(rows, dtype=np.int64).reshape(-1, 4), caps, n_split


def _fill_depth(dem: str, filled_dem: np.ndarray, bg: float) -> "np.ndarray | SparseFill":
    """``filled_dem - dem`` as float32, NaN where either has no data, indexed by flat pixel id.

    Filling changes few cells of a DEM, so while
    at most a third of them differ, only those are kept (``SparseFill``),
    worked out a strip of rows at a time; a whole raster of them took four
    bytes a cell for the whole build.  Otherwise it is the flat raster.
    """
    ds = open_raster(dem)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    bg, has_nodata = np.float32(bg), nodata is not None
    nodata = np.float32(nodata if has_nodata else 0.0)
    nrows, ncols = filled_dem.shape
    block_rows = band.GetBlockSize()[1]
    step = block_rows * max(1, 512 // block_rows)
    counts = np.zeros(nrows + 1, dtype=np.int64)
    cols, values = [], []
    dense = None
    for r0 in range(0, nrows, step):
        n = min(step, nrows - r0)
        with decoding_threads():
            fill = band.ReadAsArray(0, r0, ncols, n).astype(np.float32, copy=False)

        band.FlushCache()
        _subtract_from_filled(filled_dem[r0:r0 + n], fill, bg, has_nodata, nodata)
        if dense is None:
            per_row, columns, nonzero = _nonzero_by_row(fill)
            counts[r0 + 1:r0 + n + 1] = per_row
            cols.append(columns)
            values.append(nonzero)
            if 3 * counts.sum() > filled_dem.size:
                # Too many to keep apart: the raster, with the strips so far
                dense = _densify(nrows, ncols, np.cumsum(counts[:r0 + n + 1]), np.concatenate(cols),
                                 np.concatenate(values))
                cols = values = None
        else:
            dense[r0:r0 + n] = fill

    if dense is not None:
        return dense.ravel()
    
    return SparseFill(np.cumsum(counts), np.concatenate(cols), np.concatenate(values), ncols)

def _crop(arr2d: np.ndarray, r0: int, r1: int, c0: int, c1: int, fill) -> np.ndarray:
    """Crop ``[r0, r1) x [c0, c1)`` into an array with a one-cell ``fill`` halo."""
    out = np.full((r1 - r0 + 2, c1 - c0 + 2), fill, dtype=arr2d.dtype)
    out[1:-1, 1:-1] = arr2d[r0:r1, c0:c1]
    return out

@njit(cache=True, parallel=True)
def _neighbour_bits_parallel(fil: np.ndarray, fdr: np.ndarray, ncols: int, bg: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """``_neighbour_bits``, its rows in parallel, for the whole raster of a build: not from the build's threads."""
    inflow = np.zeros(fdr.size, dtype=np.uint8)
    has_data = np.zeros(fdr.size, dtype=np.uint8)
    for row in prange(1, fdr.size // ncols - 1):
        _neighbour_bits_of_row(fil, fdr, ncols, bg, row, inflow, has_data)
    return inflow, has_data

def _channel_lookup(channel: tuple[np.ndarray, np.ndarray, np.ndarray] | None, n: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_channel_cells``' groups, ``(keys, starts, cells)``, for ``_mark_downstream``, of a raster of ``n`` cells:
    empty for no channel.  The group of a pixel is found among the sorted ``keys``, not looked up in a raster of
    every pixel: TD(R) has a few thousand pixels, and the raster took four bytes a cell for the whole build."""
    if channel is None:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    return channel

# ---------------------------------------------------------------------------
# Segment driver
# ---------------------------------------------------------------------------

def fldpln_library_for_segment(filled_dem: np.ndarray,
                               flow_direction: np.ndarray,
                               stream_id: int,
                               stream_info: np.ndarray,
                               dh: float,
                               fldmn: float,
                               fldmx: float,
                               iterative_spill: bool,
                               global_max_wse: float = 0.0,
                               bg: float = -9999,
                               spill_decay: float = 0.0,
                               solver: str = "iterative",
                               max_cells: int = 0,
                               channel: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """Floodplain table for one stream segment.

    Parameters
    ----------
    filled_dem : np.ndarray
        Filled (depression-free) DEM.
    flow_direction : np.ndarray
        D8 flow directions, WhiteboxTools codes.
    stream_id : int
        The segment to solve.
    stream_info : np.ndarray
        One row per segment: start pixel, end pixel, length (pixels), stream id.
    dh : float
        Depth step of the iterative solver (m); must be positive.  Ignored by
        the exact solver.
    fldmn : float
        Iterative solver only: the least depth (m) a flooded pixel counts as
        when it decides where a spill stops.  DTFs are reported as found, not
        raised to it (see ``build_fldpln_library``).
    fldmx : float
        Largest depth to flood to (m).
    iterative_spill : bool
        Iterative solver only: keep spilling until the floodplain stops growing.
    global_max_wse : float, optional
        Highest water-surface elevation allowed; 0 for no limit.
    bg : float, optional
        No-data value of the DEM; such cells never flood.
    spill_decay : float, optional
        Head (m) a spill loses per pixel down its spillover path: the path's
        ``j``-th pixel floods from ``j * spill_decay`` more DTF than its first
        and backfills with that much less water, and the path ends where its
        water runs out.  The exact solver takes it off the water's head on
        every step down: to lower ground, whichever neighbour that is, or
        along the flow directions.  0 for none.
    solver : {"exact", "iterative"}, optional
        See the module docstring.
    max_cells : int, optional
        Most rows the segment may have (0 for no limit).  A segment that would
        flood more stops at the shallowest depth that exceeds it, recorded in
        ``attrs["depth_limit"]``.  Exact solver only.
    channel : tuple, optional
        The water cells grouped by the network pixel they first drain to
        (``_channel_cells``).  TD(R) then also takes the groups of its pixels:
        the channel below the segment, not only its stream line.

    Returns
    -------
    pandas.DataFrame
        ``FSP``, ``FPP`` and ``DTF`` columns.  ``attrs["depth_limit"]`` is the
        depth ``max_cells`` stopped at, or None.
    """
    fsp, fpp, dtf, depth_limit = _segment_rows(
        filled_dem, flow_direction, stream_id, stream_info, dh, fldmn, fldmx, iterative_spill, global_max_wse, bg,
        spill_decay, solver, max_cells, channel)
    table = pd.DataFrame({"FSP": fsp, "FPP": fpp, "DTF": dtf}, columns=["FSP", "FPP", "DTF"])
    table.attrs["depth_limit"] = depth_limit
    return table


def _segment_rows(filled_dem: np.ndarray, flow_direction: np.ndarray, stream_id: int, stream_info: np.ndarray,
                  dh: float, fldmn: float, fldmx: float, iterative_spill: bool, global_max_wse: float = 0.0,
                  bg: float = -9999, spill_decay: float = 0.0, solver: str = "iterative", max_cells: int = 0,
                  channel: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                  workspace: Workspace | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """``fldpln_library_for_segment`` as arrays: ``(FSP, FPP, DTF, depth_limit)``.  The library build takes these
    rather than a table per segment, whose construction cost more than solving most segments.  With a
    ``workspace`` (of these rasters and this ``channel``), the iterative solver solves the segment on the whole
    raster, not in windows."""
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")
    if solver == "iterative" and dh <= 0:
        raise ValueError("dh must be positive.")
    if max_cells and solver != "exact":
        raise ValueError("max_cells needs solver='exact': the iterative solver can lower a cell's DTF "
                         "in a later depth step, so its floodplain cannot be cut at a depth.")

    nrows, ncols = filled_dem.shape
    if flow_direction.shape != (nrows, ncols):
        raise ValueError("flow_direction shape must match filled_dem shape.")

    flow_direction = np.ascontiguousarray(flow_direction, dtype=np.uint8)

    # The 0.99999 is carried over from the MATLAB reference
    global_max_wse = np.finfo(np.float32).max if not global_max_wse else 0.99999 * global_max_wse

    flat_fdr = flow_direction.ravel()
    stream_pixels = _segment_pixels(stream_id, stream_info, flat_fdr, nrows, ncols)
    downstream = _downstream_exclusion(stream_id, stream_info, flat_fdr, nrows, ncols)

    if stream_pixels.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), None

    if workspace is not None and solver == "iterative":
        # The whole raster with its halo: ids of a window from (0, 0).  The
        # workspace widens TD(R) with its own channel cells.
        fpp, dtf, fsp = workspace.solve(_to_local(stream_pixels, ncols, 0, 0, ncols + 2), downstream, dh, fldmn,
                                        fldmx, iterative_spill, global_max_wse, bg, spill_decay, ncols)
        return fsp, fpp, dtf, None

    if channel is not None:
        downstream = _widen_downstream(downstream, channel)
    filled_dem = np.ascontiguousarray(filled_dem, dtype=np.float32)

    # Each side of the window has its own margin and grows only when the
    # floodplain reaches it: a floodplain usually runs off one way, and growing
    # every side would solve a window several times its size.  The last margin
    # reaches the raster edge, which can never be touched, so this ends.
    margins = _WINDOW_MARGINS + (max(nrows, ncols),)
    level = [0, 0, 0, 0]
    while True:
        r0, r1, c0, c1 = _window_bounds(stream_pixels, nrows, ncols,
                                        tuple(margins[i] for i in level))
        wr, wc = r1 - r0 + 2, c1 - c0 + 2

        win_fil = _crop(filled_dem, r0, r1, c0, c1, bg).ravel()
        win_fdr = _crop(flow_direction, r0, r1, c0, c1, np.uint8(0)).ravel()
        win_excluded = np.zeros(wr * wc, dtype=np.bool_)
        if downstream.size:
            inside = _in_window(downstream, ncols, r0, r1, c0, c1)
            if inside.any():
                win_excluded[_to_local(downstream[inside], ncols, r0, c0, wc)] = True
        # Sources are seeded regardless of the mask, exactly as the reference
        # does.  They are not un-excluded: when stream_info's end pixel sits
        # upstream of where the length-step FDR walk ends, a segment's own
        # pixels legitimately fall inside TD(R), and the reference lets the
        # exclusion stand.
        sources = _to_local(stream_pixels, ncols, r0, c0, wc)

        depth_limit = -1.0
        if solver == "exact":
            cells, values, source_of, depth_limit = _solve_exact(
                win_fil, win_fdr, wc, sources, win_excluded,
                fldmn, fldmx, global_max_wse, bg, spill_decay, max_cells)
        else:
            cells, values, source_of = _solve_iterative(
                win_fil, win_fdr, wc, sources, win_excluded,
                dh, fldmn, fldmx, iterative_spill, global_max_wse, bg, spill_decay)

        touched = _touched_edges(cells, wr, wc, r0, r1, c0, c1, nrows, ncols)
        if not any(touched):
            break
        LOG.debug("Stream %s reached its window's %s edge; retrying larger.", stream_id,
                  "/".join(side for side, t in zip(("top", "bottom", "left", "right"), touched) if t))
        level = [min(i + t, len(margins) - 1) for i, t in zip(level, touched)]
        # Free this window before cropping the next one
        del win_fil, win_fdr, win_excluded, cells, values, source_of

    fpp = _to_global(cells, ncols, r0, c0, wc)
    fsp = _to_global(source_of, ncols, r0, c0, wc)
    return fsp, fpp, values, (float(depth_limit) if depth_limit >= 0 else None)

def _ordered_map(pool: ThreadPoolExecutor, fn, tasks, ahead: int):
    """``pool.map(fn, tasks)``, with at most ``ahead`` tasks submitted and not yet taken.

    ``pool.map`` submits every task at once, and the results of those that
    finish early wait for the ones before them, all at once if one is slow.
    """
    tasks = iter(tasks)
    pending = deque(pool.submit(fn, task) for task in itertools.islice(tasks, ahead))
    while pending:
        yield pending.popleft().result()
        for task in itertools.islice(tasks, 1):
            pending.append(pool.submit(fn, task))


# ---------------------------------------------------------------------------
# Windowing
#
# Every solver works on a crop of the rasters surrounded by a one-cell halo of
# background.  The halo lets the inner loops use constant flat neighbour offsets
# with no bounds checks and no row/column division.  Because the halo blocks
# flow, a floodplain that reaches the usable edge would be truncated -- so the
# caller detects that and re-solves in a larger window.
# ---------------------------------------------------------------------------

def _window_bounds(pixels: np.ndarray, nrows: int, ncols: int,
                   margins: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Usable window ``(r0, r1, c0, c1)`` around ``pixels``, clipped to the raster.

    ``margins`` are the top, bottom, left and right margins, in cells.
    """
    top, bottom, left, right = margins
    rows = pixels // ncols
    cols = pixels % ncols
    r0 = max(int(rows.min()) - top, 0)
    r1 = min(int(rows.max()) + bottom + 1, nrows)
    c0 = max(int(cols.min()) - left, 0)
    c1 = min(int(cols.max()) + right + 1, ncols)
    return r0, r1, c0, c1


def _to_local(pixels: np.ndarray, ncols: int, r0: int, c0: int, wc: int) -> np.ndarray:
    """Global flat pixel ids -> window-local flat ids."""
    rows = pixels // ncols
    cols = pixels % ncols
    return ((rows - r0 + 1) * wc + (cols - c0 + 1)).astype(np.int32)

def _to_global(local: np.ndarray, ncols: int, r0: int, c0: int, wc: int) -> np.ndarray:
    """Window-local flat ids -> global flat pixel ids."""
    lr = local // wc
    lc = local - lr * wc
    return ((lr + r0 - 1) * ncols + (lc + c0 - 1)).astype(np.int32)

def _in_window(pixels: np.ndarray, ncols: int,
               r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    rows = pixels // ncols
    cols = pixels % ncols
    return (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)

def _touched_edges(local: np.ndarray, wr: int, wc: int,
                   r0: int, r1: int, c0: int, c1: int,
                   nrows: int, ncols: int) -> tuple[bool, bool, bool, bool]:
    """Which usable window edges that are not raster edges a solved cell sits on.

    Returned as top, bottom, left, right, matching ``_window_bounds``' margins.
    """
    if local.size == 0:
        return False, False, False, False
    
    lr = local // wc
    lc = local - lr * wc
    return (r0 > 0 and bool(np.any(lr == 1)),
            r1 < nrows and bool(np.any(lr == wr - 2)),
            c0 > 0 and bool(np.any(lc == 1)),
            c1 < ncols and bool(np.any(lc == wc - 2)))


@njit(cache=True, nogil=True)
def _segment_pixels(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                    nrows: int, ncols: int) -> np.ndarray:
    """Return stream pixels for a segment id, walking the FDR downstream."""
    mask = stream_info[:, 3] == stream_id
    if not np.any(mask):
        raise ValueError("stream_id not found in stream_info.")

    start = stream_info[mask, 0][0]
    length = stream_info[mask, 2][0]

    pixels = np.empty(max(length, 0), dtype=np.int64)
    if length <= 0:
        return pixels

    pixels[0] = start
    n = 1
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        pixels[n] = nxt
        n += 1
        current = nxt
    return pixels[:n]


def _step_lengths(pixels: np.ndarray, flat_fdr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Length (m) of the D8 step each pixel takes downstream; about one cell where it has none."""
    codes = flat_fdr[pixels]
    steps = np.full(len(pixels), (dx + dy) / 2)
    steps[np.isin(codes, (2, 32))] = dx                     # east, west
    steps[np.isin(codes, (8, 128))] = dy                    # south, north
    steps[np.isin(codes, (1, 4, 16, 64))] = np.hypot(dx, dy)
    return steps

def _profile_cuts(distance: np.ndarray, elevation: np.ndarray, max_deviation: float) -> np.ndarray:
    """Where to cut a reach so that the filled DEM under it is about straight along each piece.

    ``distance`` is each stream pixel's distance along the river and ``elevation`` its filled DEM, from upstream
    down. The profile is simplified top-down (Douglas-Peucker): a chord is split at its point farthest above or below
    it, until no point departs from its chord by more than ``max_deviation`` (m). Each pixel then belongs to the chord
    it lies inside; a vertex lies on two chords and joins the gentler one, so the top and the foot of a step stay with
    their pools and a step from one pixel to the next is a cut between the two. Returns the index of the first pixel
    of every piece but the first.
    """
    n = len(elevation)
    if n < 3:
        return np.zeros(0, dtype=np.int64)
    keep = np.zeros(n, dtype=bool)
    keep[[0, n - 1]] = True
    chords = [(0, n - 1)]
    while chords:
        a, b = chords.pop()
        if b - a < 2:
            continue
        along = (distance[a + 1:b] - distance[a]) / (distance[b] - distance[a])
        off = np.abs(elevation[a + 1:b] - elevation[a] - (elevation[b] - elevation[a]) * along)
        k = int(np.argmax(off))
        if off[k] > max_deviation:
            k += a + 1
            keep[k] = True
            chords += [(a, k), (k, b)]
    vertices = np.flatnonzero(keep)
    if len(vertices) <= 2:
        return np.zeros(0, dtype=np.int64)
    slope = np.abs(np.diff(elevation[vertices]) / np.diff(distance[vertices]))
    chord_of = np.r_[np.repeat(np.arange(len(vertices) - 1), np.diff(vertices)), len(vertices) - 2]
    inner = np.arange(1, len(vertices) - 1)
    chord_of[vertices[inner]] = np.where(slope[inner - 1] <= slope[inner], inner - 1, inner)
    return np.flatnonzero(np.diff(chord_of)) + 1

def _equal_lengths(pixels: np.ndarray, steps: np.ndarray, max_length: float | None) -> list[np.ndarray]:
    """Cut a run of stream pixels longer than ``max_length`` into ``ceil(length / max_length)`` of about equal
    length along the river."""
    total = float(steps.sum())
    if max_length is None or total <= max_length or len(pixels) < 2:
        return [pixels]
    n = min(math.ceil(total / max_length), len(pixels))
    upstream = np.cumsum(steps) - steps  # river distance from the run's start to each pixel
    piece = np.minimum((upstream / (total / n)).astype(np.int64), n - 1)
    return [pixels[piece == k] for k in range(n) if (piece == k).any()]

def _split_reach(row: np.ndarray, pixels: np.ndarray, steps: np.ndarray, max_length: float | None,
                 elevation: np.ndarray | None = None,
                 max_deviation: float | None = None) -> list[tuple[int, int, int, np.ndarray]]:
    """Cut a reach into segments along the river.

    ``row`` is the reach's stream table row, ``pixels`` its stream pixels from upstream down, ``steps`` their step
    lengths and ``elevation`` their filled DEM. With ``max_deviation``, the reach is first cut where its profile bends
    or steps (see ``_profile_cuts``); then every piece longer than ``max_length`` is cut into
    ``ceil(length / max_length)`` of about equal length. Returns ``(start_pixel, end_pixel, length, pixels)`` for
    each segment, from upstream down. A reach left whole keeps its own row; the last segment of a split reach keeps
    the reach's ``end_pixel``, and every other one ends at its own last pixel.
    """
    start, end, length = int(row[0]), int(row[1]), int(row[2])
    cuts = []
    if max_deviation is not None:
        cuts = _profile_cuts(np.cumsum(steps) - steps, elevation, max_deviation).tolist()
    bounds = [0, *cuts, len(pixels)]
    parts = [p for a, b in itertools.pairwise(bounds) for p in _equal_lengths(pixels[a:b], steps[a:b], max_length)]
    if len(parts) == 1:
        return [(start, end, length, pixels)]
    return [(int(p[0]), end if i == len(parts) - 1 else int(p[-1]), len(p), p) for i, p in enumerate(parts)]


@njit(cache=True, parallel=True)
def _nonzero_by_row(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cells of ``values`` that aren't 0 (NaN is not), row by row: how many in each row, their columns and their
    values."""
    nrows, ncols = values.shape
    per_row = np.zeros(nrows, dtype=np.int64)
    for r in prange(nrows):
        n = 0
        for c in range(ncols):
            if values[r, c] != 0:
                n += 1
        per_row[r] = n

    starts = np.zeros(nrows + 1, dtype=np.int64)
    starts[1:] = np.cumsum(per_row)
    cols = np.empty(starts[-1], dtype=np.int32)
    nonzero = np.empty(starts[-1], dtype=values.dtype)
    for r in prange(nrows):
        k = starts[r]
        for c in range(ncols):
            if values[r, c] != 0:
                cols[k] = c
                nonzero[k] = values[r, c]
                k += 1

    return per_row, cols, nonzero

def _densify(nrows: int, ncols: int, starts: np.ndarray, cols: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The raster of a ``SparseFill``'s rows (``starts`` of rows so far, ``cols``, ``values``), 0 elsewhere."""
    dense = np.zeros((nrows, ncols), dtype=np.float32)
    rows = np.repeat(np.arange(starts.size - 1), np.diff(starts))
    dense[rows, cols] = values
    return dense


@njit(cache=True, parallel=True)
def _subtract_from_filled(filled_dem: np.ndarray, fill: np.ndarray, bg: np.float32, has_nodata: bool,
                          nodata: np.float32):
    """``fill`` (the DEM, in place) set to ``filled_dem - fill``, or NaN where the DEM isn't finite, is -9998 or
    less, or ``nodata`` (with ``has_nodata``), or the filled DEM is ``bg``."""
    for r in prange(fill.shape[0]):
        for c in range(fill.shape[1]):
            dem = fill[r, c]
            if (not np.isfinite(dem) or dem <= -9998 or filled_dem[r, c] == bg
                    or (has_nodata and dem == nodata)):
                fill[r, c] = np.nan
            else:
                fill[r, c] = filled_dem[r, c] - dem

def save_library(table: pd.DataFrame | dict[str, np.ndarray], path: str, metadata: dict | None = None):
    """Save a library (a table, or its columns' arrays by name) as compact Parquet, with its ``metadata`` (see
    ``library_metadata``) if given.

    DTF and fill are stored as 2-decimal numbers, and each column gets the
    encoding that compresses it best; the file is several times smaller than
    plain Brotli-compressed Parquet.  It is compressed with zstd: Brotli made
    it 6% smaller, but took five times as long to write and twice as long to
    read.  Rows are converted and written a row group at a time.
    """
    with LibraryWriter(path, parquet=True, metadata=metadata) as writer:
        writer.write({c: np.asarray(table[c]) for c in table})


@njit(cache=True, nogil=True)
def _downstream_exclusion(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                          nrows: int, ncols: int) -> np.ndarray:
    """Pixels downstream of the segment end -- ``TD(R)``, excluded from spillover.

    The walk stops at a sink, the raster edge, or where the FDR loops back on
    itself.  A loop is found by Brent's method, comparing each pixel with one
    remembered at doubling intervals, so that no set of the walked pixels is
    kept.
    """
    end_pixel = stream_info[stream_info[:, 3] == stream_id, 1][0]

    excluded = np.empty(256, dtype=np.int64)
    n = 0
    current = end_pixel
    remembered = end_pixel
    interval = 1
    steps = 0
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1 or nxt == end_pixel:
            return excluded[:n]
        
        if n == excluded.size:
            bigger = np.empty(2 * n, dtype=np.int64)
            bigger[:n] = excluded
            excluded = bigger

        excluded[n] = nxt
        n += 1
        if nxt == remembered:
            # Round a loop: every pixel on it has been walked once
            return np.unique(excluded[:n])
        
        steps += 1
        if steps == interval:
            remembered = nxt
            interval *= 2
            steps = 0

        current = nxt




def _widen_downstream(downstream: np.ndarray, channel: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """TD(R) (``downstream``, its stream pixels) with the channel cells that drain first into them."""
    keys, starts, cells = channel
    if downstream.size == 0 or keys.size == 0:
        return downstream
    at = np.minimum(np.searchsorted(keys, downstream), keys.size - 1)
    groups = at[keys[at] == downstream]
    if groups.size == 0:
        return downstream
    
    # The groups' cells, gathered in one index rather than a slice per group
    sizes = starts[groups + 1] - starts[groups]
    offsets = np.cumsum(sizes) - sizes
    index = np.repeat(starts[groups] - offsets, sizes) + np.arange(sizes.sum())
    return np.concatenate([downstream, cells[index]])

