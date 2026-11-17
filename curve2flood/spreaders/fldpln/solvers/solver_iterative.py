import numpy as np
from numba import njit

from .common import OUTFLOW, _iterative_arrays, _neighbour_bits_of_row

# ---------------------------------------------------------------------------
# Depth-stepped solver -- algorithm steps 1-8
# ---------------------------------------------------------------------------

# The depth-stepped solver tests a cell's eight neighbours over and over: are
# they dry, do they have data, do they drain into it?  It reads the answers as
# bit masks, bit ``pos`` for the neighbour at ``NEIGHBOR_DELTAS[pos]``: one
# byte rather than eight scattered reads.

_UNREACHED = np.float32(3.4e38)

@njit(cache=True, nogil=True)
def _neighbour_bits(fil: np.ndarray, fdr: np.ndarray, ncols: int, bg: float) -> tuple[np.ndarray, np.ndarray]:
    """For each cell of a window with its halo (flat, ``ncols`` wide): which neighbours drain into it (``inflow``),
    and which have data (``has_data``, not ``bg``).  The halo's cells have neither."""
    inflow = np.zeros(fdr.size, dtype=np.uint8)
    has_data = np.zeros(fdr.size, dtype=np.uint8)
    for row in range(1, fdr.size // ncols - 1):
        _neighbour_bits_of_row(fil, fdr, ncols, bg, row, inflow, has_data)
    return inflow, has_data

@njit(cache=True, nogil=True, inline="always")
def _flood(cell: int, in_flood: np.ndarray, wet: np.ndarray, offsets: np.ndarray):
    """Mark ``cell`` flooded, and so a wet neighbour of each of its neighbours (``wet``: bit ``pos`` for the
    neighbour at ``NEIGHBOR_DELTAS[pos]``, whose own neighbour at ``7 - pos`` is the cell)."""
    in_flood[cell] = True
    for pos in range(8):
        neighbor = cell + offsets[pos]
        wet[neighbor] = wet[neighbor] | (1 << (7 - pos))

@njit(cache=True, nogil=True)
def _backfill_from_source(source: int, base_dtf: float, max_wse: float,
                          fil: np.ndarray, inflow: np.ndarray, offsets: np.ndarray,
                          bg: float, skip_flooded: bool, wet: np.ndarray,
                          blocked: np.ndarray, blocked_stamp: int,
                          stack: np.ndarray, visited: np.ndarray, visit_stamp: int,
                          out_pixels: np.ndarray, out_dtf: np.ndarray) -> int:
    """BFA step 1: backfill against the FDR from ``source`` (chapter 3, p.113).

    ``DTF`` is measured from ``source``'s elevation, as the dissertation defines
    it; on a filled DEM the inverse-D8 walk is elevation-monotone, so that equals
    the accumulated positive rise.  ``inflow`` (``_neighbour_bits``) gives
    the neighbours that drain into each cell, and ``wet`` those flooded,
    skipped with ``skip_flooded``.  Returns the number of cells written.
    """
    source_elev = fil[source]
    visited[source] = visit_stamp
    stack[0] = source
    top = 1
    count = 0
    while top > 0:
        top -= 1
        center = stack[top]
        bits = inflow[center]
        if skip_flooded:
            bits &= ~wet[center]

        for pos in range(8):
            if not (bits >> pos) & 1:
                continue
            neighbor = center + offsets[pos]
            if visited[neighbor] == visit_stamp:
                continue
            if blocked_stamp != 0 and blocked[neighbor] == blocked_stamp:
                continue
            neighbor_elev = fil[neighbor]
            if neighbor_elev == bg or neighbor_elev > max_wse:
                continue

            visited[neighbor] = visit_stamp
            out_pixels[count] = neighbor
            out_dtf[count] = base_dtf + max(0.0, neighbor_elev - source_elev)
            count += 1
            stack[top] = neighbor
            top += 1

    return count

@njit(cache=True, nogil=True)
def _refresh_boundary(boundary: np.ndarray, n_boundary: int,
                      added: np.ndarray, n_added: int,
                      has_data: np.ndarray, wet: np.ndarray, on_boundary: np.ndarray) -> int:
    """Re-derive the full interior boundary ``dI F`` (algorithm steps 3 and 8):
    the flooded cells with a dry neighbour that has data.

    The floodplain only grows, so a flooded cell that is not on the boundary can
    never rejoin it.  Retesting the previous boundary plus the newly flooded
    cells therefore yields the exact ``dI F``, at ``O(|dI F| + |added|)`` instead
    of a rescan of the whole floodplain.
    """
    kept = 0
    for i in range(n_boundary):
        pixel = boundary[i]
        if has_data[pixel] & ~wet[pixel]:
            boundary[kept] = pixel
            kept += 1
        else:
            on_boundary[pixel] = False

    for i in range(n_added):
        pixel = added[i]
        if on_boundary[pixel]:
            continue

        if has_data[pixel] & ~wet[pixel]:
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

    Ties keep the order of ``keys``.  Past a few hundred keys they are sorted
    by a radix sort of both as one 64-bit key, which is several times faster.
    """
    n = keys.size
    if n < 512:
        by_elevation = keys[np.argsort(-elevation[keys], kind="mergesort")]
        return by_elevation[np.argsort(depth[by_elevation], kind="mergesort")]
    
    key = (_ordered_bits(depth[keys]) << np.uint64(32)) | _ordered_bits(-elevation[keys])
    # Stable least-significant-digit radix sort, 11 bits at a time
    order = keys.copy()
    spare_order = np.empty_like(order)
    spare_key = np.empty_like(key)
    counts = np.empty(2049, dtype=np.int64)
    for shift in range(0, 64, 11):
        counts[:] = 0
        for i in range(n):
            counts[((key[i] >> np.uint64(shift)) & np.uint64(2047)) + 1] += 1
        if counts.max() == n:
            continue  # every key has this digit
        for d in range(2048):
            counts[d + 1] += counts[d]
        for i in range(n):
            d = (key[i] >> np.uint64(shift)) & np.uint64(2047)
            at = counts[d]
            counts[d] = at + 1
            spare_order[at] = order[i]
            spare_key[at] = key[i]

        order, spare_order = spare_order, order
        key, spare_key = spare_key, key

    return order

@njit(cache=True, nogil=True)
def _ordered_bits(values: np.ndarray) -> np.ndarray:
    """float32 ``values`` as unsigned integers in the same order, ties as ties (-0.0 is 0.0); no NaN."""
    bits = (values + np.float32(0.0)).view(np.uint32)
    sign = np.uint32(0x80000000)
    out = np.empty(values.size, dtype=np.uint64)
    for i in range(values.size):
        b = bits[i]
        out[i] = np.uint64(~b if b & sign else b | sign)

    return out

@njit(cache=True, nogil=True, inline="always")
def _spilled_from(cell: int, j: int, spill_dtf: float, head: float, spill_decay: float, round_id: int,
                  spilled_round: np.ndarray, spilled_depth: np.ndarray, spilled_dtf: np.ndarray) -> bool:
    """Whether an earlier path of this spill round backfilled from ``cell`` from no more DTF, and at least as deep,
    than this path would as its ``j``-th cell."""
    return (spilled_round[cell] == round_id and head - j * spill_decay <= spilled_depth[cell]
            and np.float32(spill_dtf + j * spill_decay) >= spilled_dtf[cell])

@njit(cache=True, nogil=True)
def _forward_path(start: int, spill_dtf: float, head: float, fil: np.ndarray, fdr: np.ndarray,
                  offsets: np.ndarray, flood_depths: np.ndarray,
                  bg: float, excluded: np.ndarray, spill_decay: float,
                  path: np.ndarray, seen: np.ndarray, seen_stamp: int, round_id: int,
                  spilled_round: np.ndarray, spilled_depth: np.ndarray, spilled_dtf: np.ndarray) -> int:
    """Algorithm step 6a: the trajectory ``T(y_k)``, with its two halt criteria.

    The spill floods ``start`` from ``spill_dtf`` with ``head`` metres of water
    above it, and every step down the path costs ``spill_decay`` of both: the
    path's ``j``-th cell floods from ``spill_dtf + j * spill_decay``.  So with a
    decay the path also ends where its head would run out on the next step.

    ``flood_depths`` is the reference's ``flddat``: a snapshot of the floodplain
    taken at the top of the spill round and deliberately *not* refreshed while
    the round's candidates are processed.  Reading live depths here would halt
    trajectories a cell or two earlier and change the result.

    The path is cut short, before the cell, where it joins an earlier path of
    the round that backfilled from that cell from no more DTF, at least as deep
    (``_spilled_from``).  The rest of this path is the earlier one's, the flow
    directions being the same, and so it would lower no DTF and backfill from
    no cell: the earlier path gave each of its cells as little DTF, and
    backfilled from each from as little, as deep (spills run shallowest
    first), or met a still earlier one that did.
    """
    current = start
    length = 0
    while True:
        if seen[current] == seen_stamp:
            break

        if _spilled_from(current, length, spill_dtf, head, spill_decay, round_id, spilled_round, spilled_depth, spilled_dtf):
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
        if spill_decay > 0.0 and length * spill_decay > head:
            break
        if excluded[nxt]:
            if not _spilled_from(nxt, length, spill_dtf, head, spill_decay, round_id, spilled_round, spilled_depth,
                                 spilled_dtf):
                path[length] = nxt
                length += 1
            break
        
        if fil[nxt] == bg:
            break
        if flood_depths[nxt] > 0.0 and np.float32(spill_dtf + length * spill_decay) >= flood_depths[nxt]:
            break

        current = nxt

    return length


@njit(cache=True, nogil=True)
def _iterate(fil: np.ndarray, fdr: np.ndarray, inflow: np.ndarray, has_data: np.ndarray, ncols: int,
             sources: np.ndarray, excluded: np.ndarray,
             dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
             global_max_wse: float, bg: float, spill_decay: float, arrays, counters: np.ndarray) -> int:
    """Depth-stepped FLDPLN (algorithm steps 1-8) on ``arrays`` (``_iterative_arrays``).

    Returns how many cells were listed in ``flooded``: each cell of the
    floodplain, in the order it was flooded (a source given twice, twice).
    """
    (dtf, fsp, in_flood, on_boundary, flood_depths,
     dirty, boundary, added, stack, fill_pixels, fill_dtf, path, touched, flooded,
     cand_avail, cand_belev, cand_fsp, cand_dtf, spilled_depth, spilled_dtf,
     dirty_stamp, visited, seen, on_path, cand_stamp, spilled_round, wet) = arrays

    # ``flood_depths`` is the reference's ``flddat``.  It tracks ``dtf`` during
    # the backfill phase but is frozen while a spill round processes its
    # candidates, so writes made then are queued in ``dirty`` and folded in at
    # the start of the next round.
    n_dirty = 0
    round_id = counters[1]

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
    stamp = counters[0]
    n_flooded = 0

    n_added = 0
    for i in range(sources.size):
        source = sources[i]
        if fil[source] == bg:
            continue

        dtf[source] = 0.0
        fsp[source] = source
        _flood(source, in_flood, wet, offsets)
        flood_depths[source] = fldmn
        added[n_added] = source
        flooded[n_flooded] = source
        n_flooded += 1
        n_added += 1

    n_boundary = _refresh_boundary(boundary, 0, added, n_added, has_data, wet, on_boundary)

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
            if not inflow[pixel] & ~wet[pixel]:
                continue  # no dry neighbour drains into it: there is nothing to backfill

            max_wse = min(global_max_wse, fil[pixel] + fldht - pixel_dtf)
            stamp += 1
            count = _backfill_from_source(
                pixel, pixel_dtf, max_wse, fil, inflow, offsets, bg,
                True, wet, on_path, 0, stack, visited, stamp,
                fill_pixels, fill_dtf)
            for j in range(count):
                target = fill_pixels[j]
                value = fill_dtf[j]
                if value < (dtf[target] if in_flood[target] else _UNREACHED):
                    dtf[target] = value
                    fsp[target] = pixel_fsp
                    flood_depths[target] = max(fldmn, value)
                    if not in_flood[target]:
                        _flood(target, in_flood, wet, offsets)
                        added[n_added] = target
                        flooded[n_flooded] = target
                        n_flooded += 1
                        n_added += 1

        n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, has_data, wet, on_boundary)

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
                dry = has_data[pixel] & ~wet[pixel]

                for pos in range(8):
                    if not (dry >> pos) & 1:
                        continue
                    neighbor = pixel + offsets[pos]
                    if excluded[neighbor]:
                        continue
                    neighbor_elev = fil[neighbor]
                    if neighbor_elev > limit:
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
            # shallowest spill first.  Each step down a trajectory costs
            # spill_decay: its j-th cell floods from spill_dtf + j * spill_decay,
            # with path_depth - j * spill_decay of water to backfill with.
            ordered = _order_by_depth_then_elevation(touched[:n_touched],
                                                     cand_dtf, cand_belev)
            for i in range(n_touched):
                start = ordered[i]
                spill_dtf = cand_dtf[start]
                spill_fsp = cand_fsp[start]
                start_elev = fil[start]
                path_depth = min(global_max_wse, start_elev + fldht - spill_dtf) - start_elev
                stamp += 1
                length = _forward_path(start, spill_dtf, path_depth, fil, fdr, offsets,
                                       flood_depths, bg, excluded, spill_decay,
                                       path, seen, stamp, round_id, spilled_round, spilled_depth, spilled_dtf)
                if length == 0:
                    continue

                path_stamp = stamp
                for j in range(length):
                    on_path[path[j]] = path_stamp

                for j in range(length):
                    source = path[j]
                    source_dtf = np.float32(spill_dtf + j * spill_decay)
                    if source_dtf < (dtf[source] if in_flood[source] else _UNREACHED):
                        dtf[source] = source_dtf
                        fsp[source] = spill_fsp
                        if dirty_stamp[source] != round_id:
                            dirty_stamp[source] = round_id
                            dirty[n_dirty] = source
                            n_dirty += 1
                        if not in_flood[source]:
                            _flood(source, in_flood, wet, offsets)
                            added[n_added] = source
                            flooded[n_flooded] = source
                            n_flooded += 1
                            n_added += 1
                            n_new += 1

                # The path stops before any cell an earlier path of this round
                # backfilled from from no more DTF, at least as deep (see
                # _forward_path): that one gave every cell above it as little DTF
                # as this one would, its own path cells above it too, which
                # block this one's backfill and are no lower.
                for j in range(length):
                    source = path[j]
                    source_dtf = np.float32(spill_dtf + j * spill_decay)
                    source_depth = path_depth - j * spill_decay
                    spilled_round[source] = round_id
                    spilled_depth[source] = source_depth
                    spilled_dtf[source] = source_dtf
                    stamp += 1
                    count = _backfill_from_source(
                        source, 0.0, fil[source] + source_depth, fil, inflow, offsets, bg,
                        False, wet, on_path, path_stamp, stack, visited, stamp,
                        fill_pixels, fill_dtf)
                    
                    for k in range(count):
                        target = fill_pixels[k]
                        value = fill_dtf[k] + source_dtf
                        if value < (dtf[target] if in_flood[target] else _UNREACHED):
                            dtf[target] = value
                            fsp[target] = spill_fsp
                            if dirty_stamp[target] != round_id:
                                dirty_stamp[target] = round_id
                                dirty[n_dirty] = target
                                n_dirty += 1
                            if not in_flood[target]:
                                _flood(target, in_flood, wet, offsets)
                                added[n_added] = target
                                flooded[n_flooded] = target
                                n_flooded += 1
                                n_added += 1
                                n_new += 1

            n_boundary = _refresh_boundary(boundary, n_boundary, added, n_added, has_data, wet, on_boundary)

            # Step 7: keep spilling while the boundary can still afford to.
            spill = False
            if iterative_spill and n_new > 0:
                for i in range(n_boundary):
                    if dtf[boundary[i]] < fldht:
                        spill = True
                        break
            shallow_only = True

    counters[0] = stamp
    counters[1] = round_id
    return n_flooded

def _solve_iterative(fil: np.ndarray, fdr: np.ndarray, ncols: int,
                     sources: np.ndarray, excluded: np.ndarray,
                     dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
                     global_max_wse: float, bg: float, spill_decay: float):
    """Depth-stepped FLDPLN (algorithm steps 1-8) in a window of its own.

    Returns ``(cells, dtf, fsp)`` in window-local flat ids, ascending.
    """
    inflow, has_data = _neighbour_bits(fil, fdr, ncols, bg)
    return _solve_iterative_reusing(fil, fdr, inflow, has_data, ncols, sources, excluded, dh, fldmn, fldmx,
                                    iterative_spill, global_max_wse, bg, spill_decay, _iterative_arrays(fil.size),
                                    np.zeros(2, dtype=np.int64), 0, ncols)


@njit(cache=True, nogil=True)
def _solve_iterative_reusing(fil: np.ndarray, fdr: np.ndarray, inflow: np.ndarray, has_data: np.ndarray, ncols: int,
                             sources: np.ndarray, excluded: np.ndarray,
                             dh: float, fldmn: float, fldmx: float, iterative_spill: bool,
                             global_max_wse: float, bg: float, spill_decay: float, arrays, counters: np.ndarray,
                             halo: int, out_cols: int):
    """``_solve_iterative`` on ``arrays`` (``_iterative_arrays``), which it leaves as it found them, resetting only
    the cells it flooded: a thread can keep them from one segment to the next (see ``Workspace``).

    The cells come out in ascending order: the flooded cells' bounding box
    is scanned row by row, not the whole window.  A cell at ``(row, col)``
    of the window comes out as ``(row - halo) * out_cols + col - halo``: its
    window-local id with ``halo`` 0 and ``out_cols`` ``ncols``, or with 1
    and the raster's width, its raster id when the window is the raster.
    """
    n_flooded = _iterate(fil, fdr, inflow, has_data, ncols, sources, excluded, dh, fldmn, fldmx, iterative_spill,
                         global_max_wse, bg, spill_decay, arrays, counters)
    dtf, fsp, in_flood, on_boundary, flood_depths, flooded = (arrays[0], arrays[1], arrays[2], arrays[3], arrays[4],
                                                              arrays[13])
    cells = np.empty(n_flooded, dtype=np.int32)
    out_dtf = np.empty(n_flooded, dtype=np.float32)
    out_fsp = np.empty(n_flooded, dtype=np.int32)
    if n_flooded == 0:
        return cells, out_dtf, out_fsp
    r0, c0 = fil.size, fil.size
    r1, c1 = -1, -1
    for i in range(n_flooded):
        row = flooded[i] // ncols
        col = flooded[i] - row * ncols
        r0, r1, c0, c1 = min(r0, row), max(r1, row), min(c0, col), max(c1, col)
    k = 0
    for row in range(r0, r1 + 1):
        for pixel in range(row * ncols + c0, row * ncols + c1 + 1):
            if in_flood[pixel]:
                cells[k] = (row - halo) * out_cols + pixel - row * ncols - halo
                out_dtf[k] = dtf[pixel]  # as found, not raised to fldmn (see build_fldpln_library)
                source = fsp[pixel]
                source_row = source // ncols
                out_fsp[k] = (source_row - halo) * out_cols + source - source_row * ncols - halo
                k += 1
    wet = arrays[26]
    for i in range(n_flooded):
        pixel = flooded[i]
        dtf[pixel] = 0.0
        fsp[pixel] = 0
        in_flood[pixel] = False
        on_boundary[pixel] = False
        flood_depths[pixel] = 0.0
        for neighbor in (pixel - ncols - 1, pixel - ncols, pixel - ncols + 1, pixel - 1, pixel + 1,
                         pixel + ncols - 1, pixel + ncols, pixel + ncols + 1):
            wet[neighbor] = 0
    return cells[:k], out_dtf[:k], out_fsp[:k]
