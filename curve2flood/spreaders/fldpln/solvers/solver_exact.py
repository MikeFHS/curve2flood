import numpy as np
from numba import njit

from .common import INFLOW, OUTFLOW

# The exact solver carries DTF as an integer count of these units, so that the
# priority queue can be keyed on a single packed int64.  Tenths of a millimetre
# keep the per-edge rounding well below the centimetres the library is written
# in (DECIMAL_COLS), even after accumulating over a long path, while leaving an
# int32 range worth hundreds of kilometres of depth.
_DTF_UNITS_PER_METRE = 10000.0

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
                 bg: float, spill_decay: float, max_cells: int = 0):
    """Dijkstra over ``w(u -> v) = max(0, E(v) - E(u))`` on the 8-neighbourhood.

    Returns ``(cells, dtf, fsp, depth_limit)`` in window-local flat ids.
    ``spill_decay`` is the head the water loses on each step down: to lower
    ground, whichever neighbour that is, or to a pixel's D8 successor, as the
    depth-stepped solver's spills lose it a pixel at a time down their paths.
    Steps up and across stay free of it.

    ``max_cells`` caps the floodplain at that many cells (0 is no cap).  Cells
    settle in order of DTF, so once the cap is full the sweep stops at the DTF
    ``d`` of the next cell and keeps only the cells shallower than ``d``: the
    floodplain an ``fldmx`` just below ``d`` would give.  ``depth_limit`` is
    ``d`` in metres, or -1 when the cap was not reached.  Cells flooded at zero
    depth are always kept.

    The pixels downstream of the segment (``excluded``) are not a wall.  As in
    the depth-stepped algorithm, backfill may enter them, lateral spillover may
    not, and a downhill path may step onto one but not continue along them.
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

    stop = np.int64(-1)
    n_settled = 0
    while heap_n > 0:
        key, heap_n = _heap_pop(heap, heap_n)
        pixel = np.int32(key & np.int64(0xFFFFFFFF))
        depth = np.int64(key >> 32)
        if settled[pixel] or depth != dist[pixel]:
            continue
        if max_cells > 0 and n_settled >= max_cells and depth > 0:
            stop = depth
            break

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
            # The water loses head running down, whichever way it goes, as a spill does down its path
            if descends or rise < 0.0:
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
        if settled[pixel] and (stop < 0 or dist[pixel] < stop):
            cells[k] = pixel
            out_dtf[k] = dist[pixel] / _DTF_UNITS_PER_METRE  # as found, not raised to fldmn (see build_fldpln_library)
            out_fsp[k] = fsp[pixel]
            k += 1

    depth_limit = stop / _DTF_UNITS_PER_METRE if stop >= 0 else -1.0
    return cells[:k], out_dtf[:k], out_fsp[:k], depth_limit
