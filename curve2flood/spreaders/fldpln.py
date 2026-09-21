"""FLDPLN floodplain-library construction.

Implements the FLDPLN model of Kastens (2008), *Some New Developments on Two
Separate Topics: Statistical Cross Validation and Floodplain Mapping*, chapter 3
("A New Method for Floodplain Modeling"), and descends from the MATLAB reference
``fldpln_model_v5ram.m``.

Two conventions differ deliberately from that reference; keep them in mind when
diffing against it:

* **Flow-direction encoding.**  The reference uses MATLAB/ESRI-like D8 codes
  ``INFLOW = [2, 4, 8, 1, 16, 128, 64, 32]``.  This module uses WhiteboxTools
  codes (see ``INFLOW`` / ``OUTFLOW`` below).  The neighbour ordering itself is
  unchanged.
* **Stream-table columns.**  The reference indexes ``seg_info`` by row number and
  reads segment length from column 4.  Here ``stream_info`` is looked up by
  stream id and the columns are ``start pixel (0), end pixel (1), length (2),
  stream id (3)``.

Two solvers are available.

``solver="exact"`` (default)
    A single monotone priority-queue sweep.  Chapter 3 fixes the cost of every
    move FLDPLN can make:

    * BFA step 2 (p.113) assigns ``DTF(p) = E(p) - E(x)`` over the backfill
      watershed ``B(x,h)``.  Exit paths on a filled DEM are non-increasing in
      elevation (p.111), so that difference is exactly the sum of the positive
      rises along the D8 path from ``x`` to ``p``.
    * Spillover step 5 (p.120) assigns ``DTF(y) = DTF(w) + max(0, E(y) - E(w))``
      for any 8-neighbour ``w`` of ``y`` -- the same positive-rise cost.
    * Trajectory descent (step 6a) runs downhill and so costs nothing.

    ``DTF(p)`` is therefore the minimum, over 8-connected paths from the stream
    set to ``p``, of the summed positive rises: a shortest-path problem with
    non-negative edge weights.  Theorem 3.1's observation that "backfill and
    spillover flooding processes can never assign to a floodplain pixel a DTF
    value lower than the DTF value from the input pixel acting as the floodwater
    source" is exactly the non-negativity that makes Dijkstra's algorithm
    correct, and ``_forward_path``'s ``spill_dtf >= flddat[nxt]`` stop is its
    settled-node test.  The ``dh`` loop is thus a bucket queue over that metric,
    and one sweep computes the ``dh -> 0`` limit directly.  ``dh`` is unused here.

``solver="iterative"``
    The depth-stepped loop of algorithm steps 1-8, honouring ``dh``.  Kept
    because ``dh`` is a modelling choice and not only a discretisation: Kastens
    notes (pp.123-124) that on coarse DEMs unrestricted spill can overestimate
    extent, which a larger ``dh`` damps.

Both solvers run on a padded window around the segment rather than the whole
raster.  The window grows and the segment is re-solved if the floodplain reaches
its edge, so results never depend on the window size.
"""

from __future__ import annotations

import geopandas as gpd
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from multiprocessing import shared_memory

import tqdm
import numpy as np
import polars as pl
import pandas as pd
import pyarrow as pa
import networkx as nx
import pyarrow.parquet as pq
import pyarrow.compute as pc
from numba import njit
from osgeo import gdal
from scipy.ndimage import median_filter
from numba.extending import register_jitable

from curve2flood import LOG

FLOAT_COLS = ["DTF", "fill depth"]
SCALE = 100

_SHARED_MEMORYS = {}

# Neighbor order used in the MATLAB code:
# 1 2 3
# 4 x 5
# 6 7 8
# INFLOW/OUTFLOW are indexed by this ordering, and the solvers' flat neighbour
# offsets mirror it as ``dr * ncols + dc``.  test_fldpln_solvers.py pins that.
NEIGHBOR_DELTAS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Whitebox D8
# 64  128 1
# 32  x   2
# 16  8   4

# ESRI D8
# 32  64 128
# 16  x   1
# 8   4   2

# Whitebox.  INFLOW[pos] is the code a neighbour at NEIGHBOR_DELTAS[pos] carries
# when it drains into the centre cell; OUTFLOW[pos] is the code the centre cell
# carries when it drains into that neighbour.
INFLOW = np.array([4, 8, 16, 2, 32, 1, 128, 64], dtype=np.int32)
OUTFLOW = np.array([64, 128, 1, 32, 2, 16, 8, 4], dtype=np.int32)

# MATLAB
# INFLOW = np.array([2, 4, 8, 1, 16, 128, 64, 32], dtype=np.int32)
# OUTFLOW = np.array([32, 64, 128, 16, 1, 8, 4, 2], dtype=np.int32)

SOLVERS = ("exact", "iterative")

# Window half-widths tried in turn, in cells, before falling back to the raster.
_WINDOW_MARGINS = (128, 384, 1152)

# The exact solver carries DTF as an integer count of these units, so that the
# priority queue can be keyed on a single packed int64.  Tenths of a millimetre
# keep the per-edge rounding an order of magnitude below the 3 decimals the
# library is written with, even after accumulating over a long path, while
# leaving an int32 range worth hundreds of kilometres of depth.
_DTF_UNITS_PER_METRE = 10000.0


@register_jitable(cache=True, nogil=True, forceinline=True)
def _pixel_to_rc(pixel: int, ncols: int) -> tuple[int, int]:
    return pixel // ncols, pixel % ncols

@register_jitable(cache=True, nogil=True, forceinline=True)
def _rc_to_pixel(row: int, col: int, ncols: int) -> int:
    return np.int32(row * ncols + col)

@register_jitable(cache=True, nogil=True)
def _next_downstream(pixel: int, fdr: np.ndarray, nrows: int, ncols: int) -> int:
    # ESRI D8
    # OUTFLOW = {
    #     np.uint8(32): (-1, -1),
    #     np.uint8(64): (-1, 0),
    #     np.uint8(128): (-1, 1),
    #     np.uint8(16): (0, -1),
    #     np.uint8(1): (0, 1),
    #     np.uint8(8): (1, -1),
    #     np.uint8(4): (1, 0),
    #     np.uint8(2): (1, 1),
    # }

    # Whitebox D8
    OUTFLOW = {
        np.uint8(64): (-1, -1),
        np.uint8(128): (-1, 0),
        np.uint8(1): (-1, 1),
        np.uint8(32): (0, -1),
        np.uint8(2): (0, 1),
        np.uint8(16): (1, -1),
        np.uint8(8): (1, 0),
        np.uint8(4): (1, 1),
    }

    fd = fdr[pixel]
    if fd == 0:
        return -1

    row, col = _pixel_to_rc(pixel, ncols)
    dr, dc = OUTFLOW[fd]
    rr = row + dr
    cc = col + dc
    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
        return -1
    return _rc_to_pixel(rr, cc, ncols)


@njit(cache=True, nogil=True)
def _segment_pixels(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                    nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id, walking the FDR downstream."""
    mask = stream_info[:, 3] == stream_id
    if not np.any(mask):
        raise ValueError("stream_id not found in stream_info.")

    start = stream_info[mask, 0][0]
    length = stream_info[mask, 2][0]

    pixels = []
    if length <= 0:
        return pixels

    pixels.append(start)
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        pixels.append(nxt)
        current = nxt
    return pixels

@njit(cache=True, nogil=True)
def _downstream_exclusion(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray,
                          nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end -- ``TD(R)``, excluded from spillover."""
    end_pixel = stream_info[stream_info[:, 3] == stream_id, 1][0]

    excluded: set[int] = set()
    current = end_pixel
    seen = {current}
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1 or nxt in seen:
            break
        excluded.add(nxt)
        seen.add(nxt)
        current = nxt

    return excluded


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
                   margin: int) -> tuple[int, int, int, int]:
    """Usable window ``(r0, r1, c0, c1)`` around ``pixels``, clipped to the raster."""
    rows = pixels // ncols
    cols = pixels % ncols
    r0 = max(int(rows.min()) - margin, 0)
    r1 = min(int(rows.max()) + margin + 1, nrows)
    c0 = max(int(cols.min()) - margin, 0)
    c1 = min(int(cols.max()) + margin + 1, ncols)
    return r0, r1, c0, c1

def _crop(arr2d: np.ndarray, r0: int, r1: int, c0: int, c1: int, fill) -> np.ndarray:
    """Crop ``[r0, r1) x [c0, c1)`` into an array with a one-cell ``fill`` halo."""
    out = np.full((r1 - r0 + 2, c1 - c0 + 2), fill, dtype=arr2d.dtype)
    out[1:-1, 1:-1] = arr2d[r0:r1, c0:c1]
    return out

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

def _touches_edge(local: np.ndarray, wr: int, wc: int,
                  r0: int, r1: int, c0: int, c1: int,
                  nrows: int, ncols: int) -> bool:
    """True if a solved cell sits on a usable window edge that is not a raster edge."""
    if local.size == 0:
        return False
    lr = local // wc
    lc = local - lr * wc
    if r0 > 0 and np.any(lr == 1):
        return True
    if r1 < nrows and np.any(lr == wr - 2):
        return True
    if c0 > 0 and np.any(lc == 1):
        return True
    if c1 < ncols and np.any(lc == wc - 2):
        return True
    return False


# ---------------------------------------------------------------------------
# Exact solver -- one monotone sweep over the positive-rise metric
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True, inline="always")
def _heap_push(heap: np.ndarray, n: int, key: int) -> int:
    i = n
    heap[i] = key
    while i > 0:
        parent = (i - 1) >> 1
        if heap[parent] <= heap[i]:
            break
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent
    return n + 1

@njit(cache=True, nogil=True, inline="always")
def _heap_pop(heap: np.ndarray, n: int) -> tuple[int, int]:
    top = heap[0]
    n -= 1
    heap[0] = heap[n]
    i = 0
    while True:
        left = 2 * i + 1
        right = left + 1
        small = i
        if left < n and heap[left] < heap[small]:
            small = left
        if right < n and heap[right] < heap[small]:
            small = right
        if small == i:
            break
        heap[small], heap[i] = heap[i], heap[small]
        i = small
    return top, n

@njit(cache=True, nogil=True)
def _solve_exact(fil: np.ndarray, fdr: np.ndarray, ncols: int,
                 sources: np.ndarray, excluded: np.ndarray,
                 fldmn: float, fldmx: float, global_max_wse: float,
                 bg: float, spill_decay: float):
    """Dijkstra over ``w(u -> v) = max(0, E(v) - E(u))`` on the 8-neighbourhood.

    Returns ``(cells, dtf, fsp)`` in window-local flat ids.  ``spill_decay`` is
    charged on a move to the D8 successor, which is where the depth-stepped
    solver charges it (``_forward_path``); backfill moves run strictly upstream
    and so can never be that edge.

    ``TD(R)`` is not a wall.  The reference applies it per operation, and each
    one maps onto an edge type here: ``_backfill_from_source`` never consults the
    exclusion set, so backfill edges into ``TD(R)`` stay open; ``_spill_candidates``
    refuses to select an excluded cell, so lateral spill edges into it are shut;
    and ``_forward_path`` may end a trajectory on one excluded cell but not run
    along ``TD(R)``, so a descent edge into it is open only from outside.
    """
    n = fil.size
    unreached = np.int32(2147483647)
    cap = np.int64(fldmx * _DTF_UNITS_PER_METRE + 0.5)
    decay = np.int64(spill_decay * _DTF_UNITS_PER_METRE + 0.5)

    dist = np.full(n, unreached, dtype=np.int32)
    fsp = np.full(n, np.int32(-1), dtype=np.int32)
    settled = np.zeros(n, dtype=np.bool_)

    offsets = np.empty(8, dtype=np.int64)
    offsets[0] = -ncols - 1
    offsets[1] = -ncols
    offsets[2] = -ncols + 1
    offsets[3] = -1
    offsets[4] = 1
    offsets[5] = ncols - 1
    offsets[6] = ncols
    offsets[7] = ncols + 1

    heap = np.empty(max(1024, 2 * sources.size), dtype=np.int64)
    heap_n = 0
    for i in range(sources.size):
        source = sources[i]
        if fil[source] == bg:
            continue
        dist[source] = 0
        fsp[source] = source
        if heap_n + 1 >= heap.size:
            bigger = np.empty(heap.size * 2, dtype=np.int64)
            bigger[:heap_n] = heap[:heap_n]
            heap = bigger
        heap_n = _heap_push(heap, heap_n, np.int64(source))

    n_settled = 0
    while heap_n > 0:
        key, heap_n = _heap_pop(heap, heap_n)
        pixel = np.int32(key & np.int64(0xFFFFFFFF))
        depth = np.int64(key >> 32)
        if settled[pixel] or depth != dist[pixel]:
            continue
        settled[pixel] = True
        n_settled += 1

        elevation = fil[pixel]
        from_excluded = excluded[pixel]
        code = fdr[pixel]

        for pos in range(8):
            neighbor = pixel + offsets[pos]
            if settled[neighbor]:
                continue
            descends = OUTFLOW[pos] == code          # pixel drains into neighbour
            if excluded[neighbor]:
                # TD(R) constrains the reference's spillover step, not its
                # backfill: _backfill_from_source never consults the exclusion
                # set, _spill_candidates refuses to select an excluded cell, and
                # _forward_path may end a trajectory on one but not run along it.
                backfills = fdr[neighbor] == INFLOW[pos]
                if not backfills and not (descends and not from_excluded):
                    continue
            neighbor_elev = fil[neighbor]
            if neighbor_elev == bg or neighbor_elev > global_max_wse:
                continue
            rise = neighbor_elev - elevation
            cost = np.int64(rise * _DTF_UNITS_PER_METRE + 0.5) if rise > 0.0 else np.int64(0)
            if descends:
                cost += decay
            candidate = depth + cost
            if candidate > cap or candidate >= dist[neighbor]:
                continue
            dist[neighbor] = np.int32(candidate)
            fsp[neighbor] = fsp[pixel]
            if heap_n + 1 >= heap.size:
                bigger = np.empty(heap.size * 2, dtype=np.int64)
                bigger[:heap_n] = heap[:heap_n]
                heap = bigger
            heap_n = _heap_push(heap, heap_n, (candidate << 32) | np.int64(neighbor))

    cells = np.empty(n_settled, dtype=np.int32)
    out_dtf = np.empty(n_settled, dtype=np.float32)
    out_fsp = np.empty(n_settled, dtype=np.int32)
    k = 0
    for pixel in range(n):
        if settled[pixel]:
            cells[k] = pixel
            value = dist[pixel] / _DTF_UNITS_PER_METRE
            out_dtf[k] = fldmn if value < fldmn else value
            out_fsp[k] = fsp[pixel]
            k += 1
    return cells, out_dtf, out_fsp


# ---------------------------------------------------------------------------
# Depth-stepped solver -- algorithm steps 1-8
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _backfill_from_source(source: int, base_dtf: float, max_wse: float,
                          fil: np.ndarray, fdr: np.ndarray, offsets: np.ndarray,
                          bg: float, skip_flooded: bool, in_flood: np.ndarray,
                          blocked: np.ndarray, blocked_stamp: int,
                          stack: np.ndarray, visited: np.ndarray, visit_stamp: int,
                          out_pixels: np.ndarray, out_dtf: np.ndarray) -> int:
    """BFA step 1: backfill against the FDR from ``source`` (chapter 3, p.113).

    ``DTF`` is measured from ``source``'s elevation, as the dissertation defines
    it; on a filled DEM the inverse-D8 walk is elevation-monotone, so that equals
    the accumulated positive rise.  Returns the number of cells written.
    """
    source_elev = fil[source]
    visited[source] = visit_stamp
    stack[0] = source
    top = 1
    count = 0
    while top > 0:
        top -= 1
        center = stack[top]
        for pos in range(8):
            neighbor = center + offsets[pos]
            if visited[neighbor] == visit_stamp:
                continue
            if blocked_stamp != 0 and blocked[neighbor] == blocked_stamp:
                continue
            if skip_flooded and in_flood[neighbor]:
                continue
            neighbor_elev = fil[neighbor]
            if neighbor_elev == bg or neighbor_elev > max_wse:
                continue
            if fdr[neighbor] != INFLOW[pos]:
                continue
            visited[neighbor] = visit_stamp
            out_pixels[count] = neighbor
            out_dtf[count] = base_dtf + max(0.0, neighbor_elev - source_elev)
            count += 1
            stack[top] = neighbor
            top += 1
    return count

@njit(cache=True, nogil=True)
def _is_interior_boundary(pixel: int, fil: np.ndarray, offsets: np.ndarray,
                          in_flood: np.ndarray, bg: float) -> bool:
    """``pixel`` is on the interior boundary if some neighbour is dry, non-background."""
    for pos in range(8):
        neighbor = pixel + offsets[pos]
        if not in_flood[neighbor] and fil[neighbor] != bg:
            return True
    return False

@njit(cache=True, nogil=True)
def _refresh_boundary(boundary: np.ndarray, n_boundary: int,
                      added: np.ndarray, n_added: int,
                      fil: np.ndarray, offsets: np.ndarray,
                      in_flood: np.ndarray, on_boundary: np.ndarray,
                      bg: float) -> int:
    """Re-derive the full interior boundary ``dI F`` (algorithm steps 3 and 8).

    The floodplain only grows, so a flooded cell that is not on the boundary can
    never rejoin it.  Retesting the previous boundary plus the newly flooded
    cells therefore yields the exact ``dI F``, at ``O(|dI F| + |added|)`` instead
    of a rescan of the whole floodplain.
    """
    kept = 0
    for i in range(n_boundary):
        pixel = boundary[i]
        if _is_interior_boundary(pixel, fil, offsets, in_flood, bg):
            boundary[kept] = pixel
            kept += 1
        else:
            on_boundary[pixel] = False
    for i in range(n_added):
        pixel = added[i]
        if on_boundary[pixel]:
            continue
        if _is_interior_boundary(pixel, fil, offsets, in_flood, bg):
            on_boundary[pixel] = True
            boundary[kept] = pixel
            kept += 1
    return kept

@njit(cache=True, nogil=True)
def _order_by_depth_then_elevation(keys: np.ndarray, depth: np.ndarray,
                                   elevation: np.ndarray) -> np.ndarray:
    """Ascending ``depth``, ties broken by descending ``elevation``.

    The reference orders both the interior boundary and the spillover candidates
    this way (algorithm step 6: "sort the y_k so that they are decreasing in
    spillover flood depth ... decreasing in elevation").  Order matters because
    assimilation keeps the first-arriving minimum, so it decides which FSP claims
    a cell -- and, as Kastens notes, it "help[s] limit redundant processing".
    """
    by_elevation = keys[np.argsort(-elevation[keys], kind="mergesort")]
    return by_elevation[np.argsort(depth[by_elevation], kind="mergesort")]

@njit(cache=True, nogil=True)
def _forward_path(start: int, spill_dtf: float, fil: np.ndarray, fdr: np.ndarray,
                  offsets: np.ndarray, flood_depths: np.ndarray,
                  bg: float, excluded: np.ndarray, spill_decay: float,
                  path: np.ndarray, seen: np.ndarray, seen_stamp: int) -> int:
    """Algorithm step 6a: the trajectory ``T(y_k)``, with its two halt criteria.

    ``flood_depths`` is the reference's ``flddat``: a snapshot of the floodplain
    taken at the top of the spill round and deliberately *not* refreshed while
    the round's candidates are processed.  Reading live depths here would halt
    trajectories a cell or two earlier and change the result.
    """
    current = start
    length = 0
    while True:
        if seen[current] == seen_stamp:
            break
        seen[current] = seen_stamp
        path[length] = current
        length += 1

        code = fdr[current]
        nxt = -1
        if code != 0:
            for pos in range(8):
                if OUTFLOW[pos] == code:
                    nxt = current + offsets[pos]
                    break
        if nxt == -1:
            break
        if excluded[nxt]:
            path[length] = nxt
            length += 1
            break
        if fil[nxt] == bg:
            break
        if flood_depths[nxt] > 0.0 and spill_dtf >= flood_depths[nxt]:
            break
        current = nxt
        spill_dtf += spill_decay
    return length

@njit(cache=True, nogil=True)
def _solve_iterative(fil: np.ndarray, fdr: np.ndarray, ncols: int,
                     sources: np.ndarray, excluded: np.ndarray,
                     dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
                     global_max_wse: float, bg: float, spill_decay: float):
    """Depth-stepped FLDPLN (algorithm steps 1-8).

    Returns ``(cells, dtf, fsp)`` in window-local flat ids.
    """
    n = fil.size
    unreached = np.float32(3.4e38)
    dtf = np.full(n, unreached, dtype=np.float32)
    fsp = np.full(n, np.int32(-1), dtype=np.int32)
    in_flood = np.zeros(n, dtype=np.bool_)
    on_boundary = np.zeros(n, dtype=np.bool_)

    # The reference's ``flddat``.  It tracks ``dtf`` during the backfill phase
    # but is frozen while a spill round processes its candidates, so writes made
    # then are queued in ``dirty`` and folded in at the start of the next round.
    flood_depths = np.zeros(n, dtype=np.float32)
    dirty = np.empty(n, dtype=np.int32)
    dirty_stamp = np.zeros(n, dtype=np.int32)
    n_dirty = 0
    round_id = 0

    offsets = np.empty(8, dtype=np.int64)
    offsets[0] = -ncols - 1
    offsets[1] = -ncols
    offsets[2] = -ncols + 1
    offsets[3] = -1
    offsets[4] = 1
    offsets[5] = ncols - 1
    offsets[6] = ncols
    offsets[7] = ncols + 1

    # Generation stamps stand in for the transient sets of the reference
    # implementation, so nothing has to be cleared between rounds.
    visited = np.zeros(n, dtype=np.int32)
    seen = np.zeros(n, dtype=np.int32)
    on_path = np.zeros(n, dtype=np.int32)
    stamp = 0

    boundary = np.empty(n, dtype=np.int32)
    added = np.empty(n, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    fill_pixels = np.empty(n, dtype=np.int32)
    fill_dtf = np.empty(n, dtype=np.float32)
    path = np.empty(n + 2, dtype=np.int32)

    # Spillover candidates, one slot per cell plus a touched list (step 4).
    cand_stamp = np.zeros(n, dtype=np.int32)
    cand_avail = np.empty(n, dtype=np.float32)
    cand_belev = np.empty(n, dtype=np.float32)
    cand_fsp = np.empty(n, dtype=np.int32)
    cand_dtf = np.empty(n, dtype=np.float32)
    touched = np.empty(n, dtype=np.int32)

    n_added = 0
    for i in range(sources.size):
        source = sources[i]
        if fil[source] == bg:
            continue
        dtf[source] = 0.0
        fsp[source] = source
        in_flood[source] = True
        flood_depths[source] = fldmn
        added[n_added] = source
        n_added += 1
    n_boundary = _refresh_boundary(boundary, 0, added, n_added, fil, offsets,
                                   in_flood, on_boundary, bg)

    fldht = 0.0
    iterations = int(np.ceil(fldmx / dh))
    for _ in range(iterations):
        fldht += min(dh, fldmx - fldht)

        # Steps 1-2: backfill from every interior boundary pixel and assimilate.
        order = _order_by_depth_then_elevation(boundary[:n_boundary], dtf, fil)
        n_added = 0
        for i in range(n_boundary):
            pixel = order[i]
            pixel_dtf = dtf[pixel]
            pixel_fsp = fsp[pixel]
            max_wse = min(global_max_wse, fil[pixel] + fldht - pixel_dtf)
            stamp += 1
            count = _backfill_from_source(
                pixel, pixel_dtf, max_wse, fil, fdr, offsets, bg,
                True, in_flood, on_path, 0, stack, visited, stamp,
                fill_pixels, fill_dtf)
            for j in range(count):
                target = fill_pixels[j]
                value = fill_dtf[j]
                if value < dtf[target]:
                    dtf[target] = value
                    fsp[target] = pixel_fsp
                    flood_depths[target] = max(fldmn, value)
                    if not in_flood[target]:
                        in_flood[target] = True
                        added[n_added] = target
                        n_added += 1
        n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, fil,
                                       offsets, in_flood, on_boundary, bg)

        # Steps 3-7: spillover, optionally repeated until the floodplain settles.
        # The reference filters the boundary to ``dtf < fldht`` at the end of a
        # round and feeds only that to the next one, so from the second round on
        # a boundary pixel that has already used up its head cannot spill again.
        shallow_only = False
        spill = True
        while spill:
            n_new = 0
            n_added = 0

            # Refresh the frozen flood-depth snapshot from the previous round.
            for i in range(n_dirty):
                target = dirty[i]
                flood_depths[target] = max(fldmn, dtf[target])
            n_dirty = 0
            round_id += 1

            # Steps 4-5: best spillover source for each exterior boundary pixel.
            stamp += 1
            n_touched = 0
            for i in range(n_boundary):
                pixel = boundary[i]
                pixel_elev = fil[pixel]
                if pixel_elev == bg:
                    continue
                pixel_dtf = dtf[pixel]
                if shallow_only and pixel_dtf >= fldht:
                    continue
                pixel_fsp = fsp[pixel]
                limit = min(global_max_wse, pixel_elev + fldht - pixel_dtf)
                for pos in range(8):
                    neighbor = pixel + offsets[pos]
                    if in_flood[neighbor] or excluded[neighbor]:
                        continue
                    neighbor_elev = fil[neighbor]
                    if neighbor_elev == bg or neighbor_elev > limit:
                        continue
                    climb = max(0.0, neighbor_elev - pixel_elev)
                    available = fldht - pixel_dtf - climb
                    if cand_stamp[neighbor] != stamp:
                        cand_stamp[neighbor] = stamp
                        touched[n_touched] = neighbor
                        n_touched += 1
                    elif not (available > cand_avail[neighbor] or
                              (available == cand_avail[neighbor] and
                               pixel_elev > cand_belev[neighbor])):
                        continue
                    cand_avail[neighbor] = available
                    cand_belev[neighbor] = pixel_elev
                    cand_fsp[neighbor] = pixel_fsp
                    cand_dtf[neighbor] = pixel_dtf + climb

            # Step 6: flood each spillover trajectory and its backfill watershed,
            # shallowest spill first.
            ordered = _order_by_depth_then_elevation(touched[:n_touched],
                                                     cand_dtf, cand_belev)
            for i in range(n_touched):
                start = ordered[i]
                spill_dtf = cand_dtf[start]
                spill_fsp = cand_fsp[start]
                start_elev = fil[start]
                stamp += 1
                length = _forward_path(start, spill_dtf, fil, fdr, offsets,
                                       flood_depths, bg, excluded, spill_decay,
                                       path, seen, stamp)
                if length == 0:
                    continue
                path_stamp = stamp
                for j in range(length):
                    on_path[path[j]] = path_stamp
                path_depth = min(global_max_wse, start_elev + fldht - spill_dtf) - start_elev

                for j in range(length):
                    source = path[j]
                    if spill_dtf < dtf[source]:
                        dtf[source] = spill_dtf
                        fsp[source] = spill_fsp
                        if dirty_stamp[source] != round_id:
                            dirty_stamp[source] = round_id
                            dirty[n_dirty] = source
                            n_dirty += 1
                        if not in_flood[source]:
                            in_flood[source] = True
                            added[n_added] = source
                            n_added += 1
                            n_new += 1
                for j in range(length):
                    source = path[j]
                    stamp += 1
                    count = _backfill_from_source(
                        source, 0.0, fil[source] + path_depth, fil, fdr, offsets, bg,
                        False, in_flood, on_path, path_stamp, stack, visited, stamp,
                        fill_pixels, fill_dtf)
                    for k in range(count):
                        target = fill_pixels[k]
                        value = fill_dtf[k] + spill_dtf
                        if value < dtf[target]:
                            dtf[target] = value
                            fsp[target] = spill_fsp
                            if dirty_stamp[target] != round_id:
                                dirty_stamp[target] = round_id
                                dirty[n_dirty] = target
                                n_dirty += 1
                            if not in_flood[target]:
                                in_flood[target] = True
                                added[n_added] = target
                                n_added += 1
                                n_new += 1

            n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, fil,
                                           offsets, in_flood, on_boundary, bg)

            # Step 7: keep spilling while the boundary can still afford to.
            spill = False
            if iterative_spill and n_new > 0:
                for i in range(n_boundary):
                    if dtf[boundary[i]] < fldht:
                        spill = True
                        break
            shallow_only = True

    n_out = 0
    for pixel in range(n):
        if in_flood[pixel]:
            n_out += 1
    cells = np.empty(n_out, dtype=np.int32)
    out_dtf = np.empty(n_out, dtype=np.float32)
    out_fsp = np.empty(n_out, dtype=np.int32)
    k = 0
    for pixel in range(n):
        if in_flood[pixel]:
            cells[k] = pixel
            value = dtf[pixel]
            out_dtf[k] = fldmn if value < fldmn else value
            out_fsp[k] = fsp[pixel]
            k += 1
    return cells, out_dtf, out_fsp


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
                               solver: str = "exact") -> pd.DataFrame:
    """Build an FLDPLN floodplain table for a single stream segment.

    Parameters
    ----------
    filled_dem : np.ndarray
        Filled elevation raster.  The only surface the solvers see.
    flow_direction : np.ndarray
        D8 flow-direction raster, assumes whitebox conventions.
    stream_id : int
        Stream segment identifier.
    stream_info : np.ndarray
        Segment metadata, as a 2D array with one row per segment and columns including start pixel (0), end pixel (1), length (2), and linkno (3).
    dh : float
        Flood increment step; must be positive. Ignored when ``solver="exact"``.
    fldmn : float
        Minimum flood depth.
    fldmx : float
        Maximum flood depth to evaluate.
    iterative_spill : bool
        If True, continue spilling while shallow boundary conditions exist.
        Ignored when ``solver="exact"``, which always reaches the steady state.
    global_max_wse : float, optional
        Upper bound on water-surface elevation. 0 means no bound.
    bg : float, optional
        Background/no-data value in DEM.
    spill_decay : float, optional
        When spilling, the amount to increase the DTF for each downstream pixel. This throttles spilling.
    solver : {"exact", "iterative"}, optional
        ``"exact"`` solves the depth-to-flood field in one monotone sweep -- the
        ``dh -> 0`` limit of the depth-stepped algorithm.  ``"iterative"`` runs
        the depth-stepped loop and honours ``dh``.
    Returns
    -------
    pandas.DataFrame
        Table with FSP, FPP and DTF columns.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")
    if solver == "iterative" and dh <= 0:
        raise ValueError("dh must be positive.")

    nrows, ncols = filled_dem.shape
    if flow_direction.shape != (nrows, ncols):
        raise ValueError("flow_direction shape must match filled_dem shape.")

    filled_dem = np.ascontiguousarray(filled_dem, dtype=np.float32)
    flow_direction = np.ascontiguousarray(flow_direction, dtype=np.uint8)

    # Carried over from the MATLAB reference, where mxht0 was a flood height
    # rather than an absolute water-surface elevation.  Kept for fidelity.
    global_max_wse = np.finfo(np.float32).max if not global_max_wse else 0.99999 * global_max_wse

    flat_fdr = flow_direction.ravel()
    stream_pixels = np.asarray(
        _segment_pixels(stream_id, stream_info, flat_fdr, nrows, ncols), dtype=np.int64)
    downstream = _downstream_exclusion(stream_id, stream_info, flat_fdr, nrows, ncols)
    downstream = np.fromiter(downstream, dtype=np.int64, count=len(downstream))

    header = ["FSP", "FPP", "DTF"]
    if stream_pixels.size == 0:
        return pd.DataFrame(
            {"FSP": np.empty(0, np.int32), "FPP": np.empty(0, np.int32),
             "DTF": np.empty(0, np.float32)},
            columns=header)

    margins = _WINDOW_MARGINS + (max(nrows, ncols),)
    for attempt, margin in enumerate(margins):
        r0, r1, c0, c1 = _window_bounds(stream_pixels, nrows, ncols, margin)
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

        if solver == "exact":
            cells, values, source_of = _solve_exact(
                win_fil, win_fdr, wc, sources, win_excluded,
                fldmn, fldmx, global_max_wse, bg, spill_decay)
        else:
            cells, values, source_of = _solve_iterative(
                win_fil, win_fdr, wc, sources, win_excluded,
                dh, fldmn, fldmx, iterative_spill, global_max_wse, bg, spill_decay)

        if attempt == len(margins) - 1 or not _touches_edge(
                cells, wr, wc, r0, r1, c0, c1, nrows, ncols):
            break
        LOG.debug("Stream %s reached its %d-cell window; retrying larger.",
                  stream_id, margin)

    fpp = _to_global(cells, ncols, r0, c0, wc)
    fsp = _to_global(source_of, ncols, r0, c0, wc)
    return pd.DataFrame({"FSP": fsp, "FPP": fpp, "DTF": values}, columns=header)


def _set_shared(name: str, shm: shared_memory.SharedMemory):
    """
    We need the shared memory objects to persist somewhere; otherwise, the memory is freed and the numpy arrays point to invalid memory.
    These must last the lifetime of the program! Reason being, bathymetry is the last thing written, and we need the shared memory to persist until then.
    """
    global _SHARED_MEMORYS
    _SHARED_MEMORYS[name] = shm

def read_array_and_set_shared(file: str, dtype: np.dtype, set_shared: bool,
                              name: str = None):
    ds = gdal.Open(file)
    shape = (ds.RasterYSize, ds.RasterXSize)

    dtype = np.dtype(dtype)
    if not set_shared:
        arr = np.empty(
            shape, 
            dtype=dtype
        )
        arr[:] = ds.ReadAsArray()
        return arr
    
    size = int(dtype.itemsize * np.prod(shape))
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr[:] = ds.ReadAsArray()
    globals()[name] = arr
    _set_shared(name, shm)
    return arr

def run_parallel(args):
    return fldpln_library_for_segment(
            globals()['filled_dem_array'],
            globals()['flow_direction_array'],
            *args
        )

def close_shared_memory(names: list[str]):
    """
    Close and unlink shared memory segments.

    Parameters
    ----------
    names : list[str]
        Names of the shared memory segments to close.
    """
    for name in names:
        shm = _SHARED_MEMORYS.get(name)
        if shm is not None:
            shm.close()
            shm.unlink()
            del globals()[name]
            del _SHARED_MEMORYS[name]

def init_parallel(
    names: list[str],
    shapes: list[tuple],
    dtypes: list[np.dtype],
):
    """
    Worker initializer for multiprocessing.

    Attaches shared memory segments into NumPy arrays and stores them into
    module-level globals so the per-cell worker function can run without
    pickling large arrays.

    Parameters
    ----------
    names, shapes, dtypes
        Metadata produced by :func:`get_init_parallel_args`.
    """
    shms = [shared_memory.SharedMemory(name=name) for name in names]

    for shm, name, shape, dtype in zip(shms, names, shapes, dtypes):
        _set_shared(name, shm)
        globals()[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

def build_fldpln_library(
    filled_dem: str,
    stream_info_file: str,
    flow_direction_file: str,
    library_file: str,
    dh: float,
    fldmn: float,
    fldmx: float,
    iterative_spill: bool,
    vdt_file: str = None,
    bg_mask: np.ndarray | None = None,
    stream_ids: list[int] | None = None,
    global_max_wse: float = 0.0,
    bg: float = -9999,
    parallel: bool = False,
    pbar: bool = True,
    processes: int | None = None,
    spill_decay: float = 0.0,
    solver: str = "exact"
):
    """
    Build floodplain library.

    Parameters
    ----------
    filled_dem: str
        Path to the filled DEM raster.  The only surface the solvers see, and the
        one the mapper measures its stage against.
    stream_info_file: str
        Path to the stream info CSV file.
    flow_direction_file: str
        Path to the flow direction raster.
    dh: float
        Depth increment (meters).
    fldmn: float
        Minimum depth for a cell to be considered flooded (meters).
    fldmx: float
        Maximum floodplain depth to iterate up to (meters).
    iterative_spill: bool
        Whether to use iterative spill routing when generating outputs. Recommended if the DEM is high resolution (<= 10 m)
    vdt_file: str | None
        Optional VDT file used to derive per-stream maximum depths.
    bg_mask: np.ndarray | None
        Optional background mask to exclude certain pixels from flooding.
    stream_ids: list[int] | None
        Optional subset of COMIDs to process.
    global_max_wse: float
        Global maximum water-surface elevation override.
    bg: float
        Background/no-data value for output rasters.
    parallel: bool
        If True, process stream reaches using multiprocessing.
    pbar
        If True, display a progress bar.
    processes
        Number of worker processes to use when parallel is enabled.
    spill_decay: float
        DTF added per downstream pixel when a spillover path is routed. Throttles
        how far a single spill travels.
    solver: str
        "exact" (default) solves each segment in a single monotone sweep -- the
        dh -> 0 limit of the depth-stepped algorithm -- and ignores dh.
        "iterative" runs the depth-stepped loop and honours dh.
    """
    try:
        _build_fldpln_library(
            filled_dem, stream_info_file, flow_direction_file, library_file,
            dh, fldmn, fldmx, iterative_spill, vdt_file, stream_ids,
            global_max_wse, bg, parallel, pbar, processes, bg_mask,
            spill_decay, solver
        )
    finally:
        close_shared_memory(['filled_dem_array', 'flow_direction_array'])

def _build_fldpln_library(
    filled_dem: str,
    stream_info_file: str,
    flow_direction_file: str,
    library_file: str,
    dh: float,
    fldmn: float,
    fldmx: float,
    iterative_spill: bool,
    vdt_file: str = None,
    stream_ids: list[int] | None = None,
    global_max_wse: float = 0.0,
    bg: float = -9999,
    parallel: bool = False,
    pbar: bool = True,
    processes: int | None = None,
    bg_mask: np.ndarray | None = None,
    spill_decay: float = 0.0,
    solver: str = "exact"
):
    """
    Build floodplain library.

    Parameters
    ----------
    filled_dem: str
        Path to the filled DEM raster.  The only surface the solvers see, and the
        one the mapper measures its stage against.
    stream_info_file: str
        Path to the stream info CSV file.
    flow_direction_file: str
        Path to the flow direction raster.
    dh: float
        Depth increment (meters).
    fldmn: float
        Minimum depth for a cell to be considered flooded (meters).
    fldmx: float
        Maximum floodplain depth to iterate up to (meters).
    iterative_spill: bool
        Whether to use iterative spill routing when generating outputs. Recommended if the DEM is high resolution (<= 10 m)
    vdt_file: str | None
        Optional VDT file used to derive per-stream maximum depths.
    stream_ids: list[int] | None
        Optional subset of COMIDs to process.
    global_max_wse: float
        Global maximum water-surface elevation override.
    bg: float
        Background/no-data value for output rasters.
    parallel: bool
        If True, process stream reaches using multiprocessing.
    pbar
        If True, display a progress bar.
    processes
        Number of worker processes to use when parallel is enabled.
    spill_decay: float
        DTF added per downstream pixel when a spillover path is routed. Throttles
        how far a single spill travels.
    solver: str
        "exact" (default) solves each segment in a single monotone sweep -- the
        dh -> 0 limit of the depth-stepped algorithm -- and ignores dh.
        "iterative" runs the depth-stepped loop and honours dh.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")

    if Path(stream_info_file).suffix in {'.parquet', '.pq'}:
        stream_info = pd.read_parquet(stream_info_file)
    else:
        stream_info = pd.read_csv(stream_info_file)
    assert stream_info.ndim == 2, "Stream info file must be a 2D table"
    assert stream_info.shape[1] == 4, "Stream info file must have 4 columns: start pixel (0), end pixel (1), length (2), and stream ID (3)"

    if stream_ids is not None:
        stream_info = stream_info[stream_info.iloc[:, 3].isin(stream_ids)]

    filled_dem_array = read_array_and_set_shared(filled_dem, np.float32, set_shared=parallel, name='filled_dem_array')
    flow_direction_array = read_array_and_set_shared(flow_direction_file, np.uint8, set_shared=parallel, name='flow_direction_array')

    if bg_mask is not None:
        filled_dem_array[bg_mask] = bg

    stream_ids = stream_info.iloc[:, 3].unique()
    if vdt_file is None:
        max_depths = [fldmx] * len(stream_ids)
    else:
        if Path(vdt_file).suffix in {'.parquet', '.pq'}:
            vdt_df = pd.read_parquet(vdt_file)
        else:
            vdt_df = pd.read_csv(vdt_file)

        vdt_df = vdt_df[vdt_df['COMID'].isin(stream_ids)]
        # Find last wse_* column
        wse_cols = [col for col in vdt_df.columns if col.startswith('wse_')]
        assert wse_cols, "No wse_* columns found in VDT file"
        max_wse_col = wse_cols[-1]
        # making a copy to resolve a fragmentation warning from pandas
        vdt_df = vdt_df[['COMID', 'Row', 'Col', max_wse_col]].copy()
        vdt_df['depth'] = vdt_df[max_wse_col] - filled_dem_array[vdt_df['Row'], vdt_df['Col']]
        ids_max_depths = vdt_df.groupby('COMID', sort=False, as_index=False)['depth'].max().values
        stream_ids = ids_max_depths[:, 0].astype(np.int32)
        max_depths = np.minimum(ids_max_depths[:, 1], fldmx)

    if len(stream_ids) == 0:
        LOG.warning("No stream segments found to process. Exiting.")
        return

    if pbar:
        pbar = tqdm.tqdm
    else:
        pbar = lambda x, **kwargs: x

    if processes is None:
        processes = max(min(mp.cpu_count(), len(stream_ids)), 1)
        if processes == 1:
            parallel = False

    # Materialise the stream table once: building it inside the comprehension
    # made a fresh copy per segment, and every copy was pickled to a worker.
    stream_info_values = stream_info.values
    args = [
        (
            stream_id,
            stream_info_values,
            dh,
            fldmn,
            max_depths[i],
            iterative_spill,
            global_max_wse,
            bg,
            spill_decay,
            solver
        )
        for i, stream_id in enumerate(stream_ids)
    ]
    # Sort args from largest max depth to smallest, (longest time to shortest)
    args.sort(key=lambda x: x[4], reverse=True)

    dfs: list[pd.DataFrame] = []
    if parallel:
        names = ['filled_dem_array', 'flow_direction_array']
        shapes = [filled_dem_array.shape, flow_direction_array.shape]
        dtypes = [filled_dem_array.dtype, flow_direction_array.dtype]
        with mp.Pool(processes, init_parallel, (names, shapes, dtypes)) as pool:
            for df in pbar(pool.imap_unordered(run_parallel, args), total=len(args), desc="Processing streams in parallel"):
                dfs.append(df)
    else:
        for i, _ in enumerate(pbar(stream_ids, desc="Processing streams")):
            dfs.append(fldpln_library_for_segment(
                filled_dem_array,
                flow_direction_array,
                *args[i]
            ))

    close_shared_memory(['filled_dem_array', 'flow_direction_array'])

    df = pd.concat(dfs, ignore_index=True)
    dfs = None
    df = df.sort_values(['FSP', 'FPP'], kind='stable', ignore_index=True) # Sorting like this allows very nice compression
    if Path(library_file).suffix in {'.parquet', '.pq'}:
        save_library(df, library_file)
    else:
        df.to_csv(library_file, index=False)

def save_library(df: pd.DataFrame, path: str):
    """
    We can make the parquet file 6x smaller than just brotli-compressed parquet by using the right encodings.
    """
    cols, enc = {}, {"FPP": "DELTA_BINARY_PACKED"}
    DEC = pa.decimal32(5, 2)
    for c in df.columns:
        if c in FLOAT_COLS:
            cols[c] = pc.cast(pa.array(df[c].to_numpy().astype(np.float32).round(2)), DEC)
            enc[c] = "BYTE_STREAM_SPLIT"
        else:
            cols[c] = pa.array(df[c].to_numpy())

    pq.write_table(pa.table(cols), path, use_dictionary=["FSP"],
                   column_encoding=enc, store_decimal_as_integer=True, compression="brotli")

def make_dtf_map(filled_dem_file: str, fldpln_library_file: str, output_file: str):
    filled_dem_array: np.ndarray = gdal.Open(filled_dem_file).ReadAsArray()
    nrows, ncols = filled_dem_array.shape

    if Path(fldpln_library_file).suffix in {'.parquet', '.pq'}:
        df = pd.read_parquet(fldpln_library_file)
    else:
        df = pd.read_csv(fldpln_library_file)

    df = df.groupby('FPP', as_index=False).agg({'DTF': 'min'})
    df['Row'] = (df['FPP'] // ncols).astype(int)
    df['Col'] = (df['FPP'] % ncols).astype(int)
    df['DTF'] = df['DTF'].clip(lower=0)

    dtf_array = np.full_like(filled_dem_array, np.nan, dtype=np.float32)
    dtf_array[df['Row'], df['Col']] = df['DTF']

    out_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(
        output_file,
        ncols,
        nrows,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=DEFLATE']
    )
    out_ds.SetGeoTransform(gdal.Open(filled_dem_file).GetGeoTransform())
    out_ds.SetProjection(gdal.Open(filled_dem_file).GetProjection())
    out_ds.WriteArray(dtf_array)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

def limit_rise(arr, max_rise=0.5):
    """
    Limit upward rises in a 1D array.

    ``out[i] = min(arr[i], out[i-1] + max_rise)``, which is the same as
    ``min_j<=i (arr[j] + max_rise * (i - j))``; subtracting the ramp turns that
    into a running minimum, so the whole thing vectorises.

    Parameters
    ----------
    arr : array-like
        Elevation values, evenly spaced and free of NaN.
    max_rise : float
        Maximum allowed rise per step.

    Returns
    -------
    np.ndarray
    """
    out = np.asarray(arr, dtype=float)
    if out.size == 0:
        return out.copy()
    ramp = max_rise * np.arange(out.size, dtype=float)
    return np.minimum.accumulate(out - ramp) + ramp

def _fill_missing_profile(valid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Carry a conditioned profile into the stream pixels the VDT does not cover.

    ``valid`` marks the pixels that carried a VDT stage; ``values`` is the
    conditioned profile over just those pixels.  Which rule is used barely
    matters -- linear and nearest score within 0.0005 MCC of this one over 51
    sites -- but filling the gaps at all does: leaving them empty costs 0.007.
    """
    out = np.full(len(valid), np.nan, dtype=float)
    out[valid] = values
    return pd.Series(out).ffill().bfill().to_numpy(dtype=float)

def longest_path_decomposition(G: nx.DiGraph, stream_wse_dict: dict[int, list[tuple[int, float]]]) -> list[list[int]]:
    """
    Decompose a DAG into disjoint longest headwater->outlet paths.

    Returns
    -------
    list[list]
        Each element is a list of stream IDs ordered upstream->downstream.
    """
    paths = []
    length = {sid: len(stream_wse_dict[sid]) for sid in G}

    while G.nodes:
        topo = list(nx.topological_sort(G))

        longest = {}

        for node in reversed(topo):
            children = list(G.successors(node))

            if not children:
                longest[node] = (length[node], [node])
            else:
                best = max(
                    (longest[c] for c in children),
                    key=lambda x: x[0]
                )
                longest[node] = (length[node] + best[0], [node] + best[1])

        # Find longest path beginning at a headwater
        headwaters = [n for n in G if G.in_degree(n) == 0]

        _, path = max(
            (longest[h] for h in headwaters),
            key=lambda x: x[0]
        )

        paths.append(path)

        G.remove_nodes_from(path)

    return paths

def _make_fldpln_flood_map(
        dem: np.ndarray,
        filled_dem: np.ndarray,
        vdt_df: pl.DataFrame,
        fdr: np.ndarray,
        fldpln_library: pl.LazyFrame,
        stream_info_df: pd.DataFrame,
        stream_gdf: gpd.GeoDataFrame,
        reach_id_field: str,
        downstream_reach_id_field: str,
        max_wse_rise: float = 0.5,
        median_filter_size: int = 53,
        max_drop_below_source: float = 2.0):
    """Turn a VDT water surface into a flood map using a floodplain library.

    Each stream pixel's stage is conditioned along its longest-path chain, then
    ``wse(p) = filled_dem[p] + DoF(s) - DTF(s, p)`` maximised over the stream
    pixels ``s`` that reach ``p``.  That is the same as ``WSE(s) - L(s, p)``
    where ``L = DTF - (fil[p] - fil[s])`` is the minimum total descent from
    ``s`` to ``p``, so the stream pixel's own filled elevation cancels: the
    water lies flat across a floodplain, loses exactly the lip height crossing
    a levee, and fills a depression to the level of its rim.

    That cancellation is only meaningful while ``p`` lies near the level of
    ``s``.  ``_solve_exact`` charges nothing for a downhill step, so the library
    pairs every stream pixel with the whole slope beneath it, and the formula
    then stands the source's own stage on top of whatever ground it finds there.
    ``max_drop_below_source`` is how far below the source's filled elevation a
    floodplain pixel may still lie once its stage is spent: a pixel is kept only
    while ``fil[s] - fil[p] <= DoF(s) + max_drop_below_source``.  Without it a
    steep reach paints its entire hillside wet -- on a 140 m/4 km reach, 58% of
    the wet pixels sat more than 20 m below the stream pixel that wetted them
    and 44% more than 100 m below, some of them 15 km away.

    ``median_filter_size`` is the only conditioning knob left.  Measured over
    51 benchmark sites and 809 flow events (MCC inside each site's boundary
    raster), dropping the filter costs 0.017 while every window from 11 to 101
    lands within 0.003 of the default 53.  The rise cap, the gap-filling rule
    and the choice of DoF signal each moved the mean by less than 0.008 and won
    on roughly half the sites, so they are fixed constants here; the remaining
    error is a per-site stage bias of about +-1.5 m that none of them can
    reach.
    """
    nrows, ncols = filled_dem.shape
    median_filter_size = int(median_filter_size)
    if median_filter_size < 1:
        median_filter_size = 1
    if median_filter_size % 2 == 0:
        median_filter_size += 1

    vdt_df = vdt_df.with_columns(FSP=(pl.col('Row') * ncols + pl.col('Col')).cast(pl.Int32))
    fsp_wse_dict = dict(zip(vdt_df['FSP'], vdt_df['WSE']))
    stream_ids = set(vdt_df['COMID'])

    stream_gdf = stream_gdf.sort_values('topological_order')
    G = nx.from_pandas_edgelist(
        stream_gdf,
        source=reach_id_field,
        target=downstream_reach_id_field,
        create_using=nx.DiGraph
    )
    if -1 in G:
        G.remove_node(-1)  # Remove the dummy downstream node

    fdr = fdr.ravel()
    stream_rows: dict[list[tuple]] = defaultdict(list)
    stream_info_dict = stream_info_df.set_index('stream_id').to_dict(orient='index')

    # We need to traverse each stream segment, and add missing FSPs to the vdt_df with interpolated DoF values.
    for stream_id in stream_ids:
        # if stream_id not in  {770294095, 770275374}:
        #     continue
        if stream_id not in stream_info_dict:
            raise ValueError(f"Stream ID {stream_id} not found in stream_info_df.")

        fsp = stream_info_dict[stream_id]['start_pixel']
        length = stream_info_dict[stream_id]['length']

        for _ in range(length):
            if fsp in fsp_wse_dict:
                raw_wse = fsp_wse_dict[fsp]
                stream_rows[stream_id].append((fsp, raw_wse))
            else:
                stream_rows[stream_id].append((fsp, np.nan))

            fsp = _next_downstream(fsp, fdr, nrows, ncols)
            if fsp == -1:
                break

    wse_array = np.full_like(dem, np.nan, dtype=np.float32)
    if not stream_rows:
        return wse_array
    
    # Let us combine stream ids that follow the main stems in the Graph
    paths = longest_path_decomposition(G, stream_rows)

    row_chunks = []
    for path in paths:
        path_rows = []
        for stream_id in path:
            path_rows.extend(stream_rows[stream_id])
    
        if not path_rows:
            continue

        Y = np.array([wse for _, wse in path_rows])
        mask = ~np.isnan(Y)
        if mask.sum() == 0:
            continue
        filled_dem_profile = np.array([filled_dem[_pixel_to_rc(pixel, ncols)]
                                       for pixel, _ in path_rows])

        # Two conditioned profiles, differing only in the space they are
        # smoothed and gap-filled in: ``wse_dof`` works on the water surface and
        # ``depths_interped`` on the stage above the filled DEM.  The rise cap
        # only ever lowers, so ``wse_dof <= depths_interped`` at about two
        # thirds of stream pixels, and the minimum is the conservative branch.
        wse_smoothed = median_filter(Y[mask], size=median_filter_size)
        wse_interped = _fill_missing_profile(
            mask, limit_rise(wse_smoothed, max_rise=max_wse_rise))
        wse_dof = wse_interped - filled_dem_profile

        depths_smoothed = median_filter((Y - filled_dem_profile)[mask],
                                        size=median_filter_size)
        depths_interped = _fill_missing_profile(mask, depths_smoothed)

        dof_profile = np.minimum(wse_dof, depths_interped)

        row_chunks.extend([
            (fsp, dof, src_fil)
            for (fsp, _), dof, src_fil in zip(path_rows, dof_profile, filled_dem_profile)
            if np.isfinite(dof)
        ])

    if not row_chunks:
        return wse_array

    df = pl.LazyFrame(
        row_chunks,
        schema={'FSP': pl.Int32, 'DoF': pl.Float32, 'SrcFil': pl.Float32},
        orient='row',
    )

    # merge the two dataframes on the row and column indices
    joined = fldpln_library.join(df, on='FSP', how='inner').collect()
    # Keep only the pairs whose source can still stand over the pixel once its
    # stage is spent.  The library's downhill edges are free, so without this a
    # stream pixel reaches every cell that drains past it, however far below.
    joined = joined.with_columns(
        FppFil=pl.Series('FppFil', filled_dem.ravel()[joined['FPP'].to_numpy()])
    ).filter(
        pl.col('SrcFil') - pl.col('FppFil') <= pl.col('DoF') + max_drop_below_source
    )

    fldpln_library = joined.lazy().with_columns(
        DTF=(pl.col('DoF') - pl.col('DTF'))
    )
    # ``DoF - DTF`` is the head left over at the floodplain pixel, measured from
    # the filled DEM, so it is allowed to be negative there -- the pixel is still
    # wet whenever the filled surface sits above the mapping DEM by more.  Take
    # the deepest of the sources that reach the pixel.
    fldpln_library = fldpln_library.group_by('FPP').agg(pl.max('DTF'))

    fldpln_library: pl.DataFrame = fldpln_library.collect()

    rows = fldpln_library.with_columns(Row=(pl.col('FPP') // ncols).cast(pl.Int32))['Row']
    cols = fldpln_library.with_columns(Col=(pl.col('FPP') % ncols).cast(pl.Int32))['Col']
    # The water surface needs no DEM: a library built with a ``fill depth``
    # column added ``dem + (DoF - DTF) + (filled_dem - dem)``, and ``dem``
    # cancels.  Libraries that still carry that column simply go unused, so the
    # two forms give identical maps.
    wse_array[rows, cols] = fldpln_library['DTF'] + filled_dem[rows, cols]
    mask = (dem > -9998) & (wse_array > dem)
    wse_array[~mask] = np.nan

    return wse_array
