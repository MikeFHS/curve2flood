from __future__ import annotations


import numpy as np
import pandas as pd
from numba import njit, prange

# Neighbor order used in the MATLAB code:
# 1 2 3
# 4 x 5
# 6 7 8
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

@njit(cache=True, nogil=True)
def _pixel_to_rc(pixel: int, ncols: int) -> tuple[int, int]:
    return pixel // ncols, pixel % ncols

@njit(cache=True, nogil=True)
def _rc_to_pixel(row: int, col: int, ncols: int) -> int:
    return np.int32(row * ncols + col)

@njit(cache=True, nogil=True)
def _valid_neighbors(pixel: int, nrows: int, ncols: int):
    row, col = _pixel_to_rc(pixel, ncols)
    for pos, (dr, dc) in enumerate(NEIGHBOR_DELTAS):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < nrows and 0 <= cc < ncols:
            yield (pos, _rc_to_pixel(rr, cc, ncols))

@njit(cache=True, nogil=True)
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

def _segment_pixels_matlab(seg_id: int, seg_info: np.ndarray, fdr: np.ndarray,
                    nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id. The seed-point path is not ported."""
    row_idx = int(seg_id)
    if row_idx < 0 or row_idx >= seg_info.shape[0]:
        raise IndexError("seg0 is outside seg_info.")

    start = round(seg_info[row_idx, 0])
    length = round(seg_info[row_idx, 4])

    if length <= 0:
        return []

    pixels = [start]
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        pixels.append(nxt)
        current = nxt
    return pixels


def _downstream_exclusion_matlab(seg_id: int, seg_info: np.ndarray, fdr: np.ndarray,
                          nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end, excluded from spillover candidates."""
    row_idx = int(seg_id)
    end_pixel = int(seg_info[row_idx, 1])

    excluded: set[int] = set()
    current = end_pixel
    seen = {current}
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1 or nxt in seen:
            break
        excluded.add(np.int32(nxt))
        seen.add(np.int32(nxt))
        current = nxt

@njit(cache=True, nogil=True, parallel=True)
def _segment_pixels(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id. The seed-point path is not ported."""
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

@njit(cache=True, nogil=True, parallel=True)
def _downstream_exclusion(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end, excluded from spillover candidates."""
    end_pixel = stream_info[stream_info[:, 3] == stream_id, 1][0]  # Assuming linkno is in the 4th column (index 3)

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

@njit(cache=True, nogil=True)
def _backfill_from_source(source: int, base_dtf: float, max_wse: float,
                          fil: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int,
                          bg: float, flood_members: set[int] | None = None,
                          excluded: set[int] | None = None) -> list[tuple[int, float]]:
    """
    Backfill opposite the D8 flow direction from `source`.

    Returned DTF values include `base_dtf`. When `flood_members` is supplied,
    pixels already in the floodplain are not returned; this matches the first
    backfill pass in the MATLAB code. Spillover backfill passes leave it as None
    so existing pixels can be overwritten when the new DTF is lower.
    """
    # Whitebox
    INFLOW = np.array([4, 8, 16, 2, 32, 1, 128, 64], dtype=np.int32)

    # MATLAB
    # INFLOW = np.array([2, 4, 8, 1, 16, 128, 64, 32], dtype=np.int32)

    source_elev = fil[source]
    result: list[tuple[int, float]] = []
    queue: list[int] = [source]
    visited = {source}
    if excluded is None:
        excluded = {np.int32(-1)} # This helps numba know what the set's type is

    while queue:
        center = queue.pop()
        for pos, nbr in _valid_neighbors(center, nrows, ncols):
            if nbr in visited or nbr in excluded:
                continue
            if flood_members is not None and nbr in flood_members:
                continue
            if fil[nbr] == bg or fil[nbr] > max_wse:
                continue
            if fdr[nbr] != INFLOW[pos]:
                continue
            dtf = base_dtf + max(0.0, fil[nbr] - source_elev)
            visited.add(np.int32(nbr))
            result.append((nbr, dtf))
            queue.append(nbr)

    return result

@njit(cache=True, nogil=True)
def sort_boundary(lst: list, fil: np.ndarray) -> list:
    # This is a numba-compatible sort, since numba cannot cache .sort(key=lambda x: (x[2], -fil[x[1]]
    def key_func(x):
        return x[2], -fil[x[1]]  # Sort by spillover, than elevation

    for i in range(1, len(lst)):
        current_item = lst[i]
        current_key = key_func(current_item)
        j = i - 1
        
        while j >= 0 and key_func(lst[j]) < current_key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = current_item
        
    return lst

@njit(cache=True, nogil=True)
def sort_candidates(lst: list) -> list:
    # This is a numba-compatible sort, since numba cannot cache .sort(key=lambda x: (x[2], -x[4]))
    def key_func(x):
        return x[2], -x[4]  # Sort by spillover, than elevation

    for i in range(1, len(lst)):
        current_item = lst[i]
        current_key = key_func(current_item)
        j = i - 1
        
        while j >= 0 and key_func(lst[j]) < current_key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = current_item
        
    return lst

@njit(cache=True, nogil=True)
def _initial_boundary(records: dict[int, tuple[int, float]], fil: np.ndarray,
              nrows: int, ncols: int, bg: float) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for pixel, (fsp, dtf) in records.items():
        if fil[pixel] == bg:
            continue
        for _, nbr in _valid_neighbors(pixel, nrows, ncols):
            if nbr not in records and fil[nbr] != bg:
                out.append((fsp, pixel, dtf))
                break
    out = sort_boundary(out, fil)
    return out

@njit(cache=True, nogil=True)
def _update_boundary(records: dict[int, tuple[int, float]], filled_dem: np.ndarray, new_boundary: set[int]) -> list[tuple[int, int, float]]:
    bdy = [
        (records[p][0], p, records[p][1])
        for p in new_boundary
    ]
    bdy = sort_boundary(bdy, filled_dem)
    return bdy

@njit(cache=True, nogil=True)
def _flood_map(records: dict[int, tuple[int, float]], flddat: np.ndarray, fldmn: float) -> None:
    for pixel, (_, dtf) in records.items():
        flddat[pixel] = max(fldmn, dtf)

@njit(cache=True, nogil=True)
def _spill_candidates(boundary: list[tuple[int, int, float]], records: dict[int, tuple[int, float]],
                      fil: np.ndarray, nrows: int, ncols: int,
                      fldht: float, mxht: float, bg: float, excluded: set[int]) -> list[tuple[int, float, int, float, float]]:
    best: dict[int, tuple[float, float, int, float, float]] = {}
    for fsp, bdy_pixel, bdy_dtf in boundary:
        bdy_elev = fil[bdy_pixel]
        if bdy_elev == bg:
            continue
        limit = min(mxht, bdy_elev + fldht - bdy_dtf)
        for _, nbr in _valid_neighbors(bdy_pixel, nrows, ncols):
            if nbr in records or nbr in excluded:
                continue
            nbr_elev = fil[nbr]
            if nbr_elev == bg or nbr_elev > limit:
                continue
            spill_dtf = bdy_dtf + max(0.0, nbr_elev - bdy_elev)
            available_depth = fldht - bdy_dtf - max(0.0, nbr_elev - bdy_elev)
            if nbr not in best:
                best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)
            else:
                previous = best[nbr]
                old_available = previous[0]
                old_bdy_elev = previous[1]
                if available_depth > old_available or (
                    available_depth == old_available and bdy_elev > old_bdy_elev
                ):
                    best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)

    candidates = [
        (fsp, spill_dtf, pixel, pixel_elev, bdy_elev)
        for pixel, (_, bdy_elev, fsp, spill_dtf, pixel_elev) in best.items()
    ]
    candidates = sort_candidates(candidates)
    return candidates

@njit(cache=True, nogil=True)
def _forward_path(start: int, spill_dtf: float, fil: np.ndarray, fdr: np.ndarray,
                  flddat: np.ndarray, nrows: int, ncols: int, bg: float,
                  excluded: set[int]) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    current = start
    while True:
        if current in seen:
            break
        seen.add(np.int32(current))
        path.append(np.int32(current))
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt == -1:
            break
        if nxt in excluded:
            path.append(np.int32(nxt))
            break
        if fil[nxt] == bg:
            break
        if flddat[nxt] > 0 and spill_dtf >= flddat[nxt]:
            break
        current = nxt
    return path

@njit(cache=True, nogil=True)
def _assimilate(records: dict[int, tuple[int, float]], fsp: int, pixel: int, dtf: float) -> bool:
    if pixel not in records:
        records[pixel] = (fsp, dtf)
        return True
    
    if records[pixel][1] > dtf:
        records[pixel] = (fsp, dtf)
        return True
    
    return False

@njit(cache=True, nogil=True, parallel=True)
def _fldpln_library_for_segment(shape: tuple[int, int],
                               dem: np.ndarray,
                               filled_dem: np.ndarray, 
                               flow_direction: np.ndarray, 
                               stream_id: int,
                               stream_info: np.ndarray,
                               dh: float,
                               fldmn: float,
                               fldmx: float,
                               iterative_spill: bool,
                               global_max_wse: float = 0.0,
                               bg: float = -9999) -> pd.DataFrame:
    """
    Build an FLDPLN floodplain table for a stream segment.
    """
    nrows, ncols = shape
    stream_pixels = _segment_pixels(stream_id, stream_info, flow_direction, nrows, ncols)
    excluded = _downstream_exclusion(stream_id, stream_info, flow_direction, nrows, ncols)

    records: dict[int, tuple[int, float]] = {}
    for pixel in stream_pixels:
        if filled_dem[pixel] != bg:
            records[pixel] = (pixel, 0.0)

    fldht = 0.0
    iterations = int(np.ceil(fldmx / dh))

    flood_depths = np.zeros(filled_dem.size, dtype=np.float32)

    for _ in range(iterations):
        fldht += min(dh, fldmx - fldht)

        boundary = _initial_boundary(records, filled_dem, nrows, ncols, bg)
        new_boundary = set()
        _flood_map(records, flood_depths, fldmn)
        flood_members = set(records)

        for fsp, boundary_pixel, boundary_dtf in boundary:
            boundary_elev = filled_dem[boundary_pixel]
            max_wse = min(global_max_wse, boundary_elev + fldht - boundary_dtf)
            additions = _backfill_from_source(
                boundary_pixel, boundary_dtf, max_wse, filled_dem, flow_direction, nrows, ncols,
                bg, flood_members=flood_members
            )
            for pixel, dtf in additions:
                if _assimilate(records, fsp, pixel, dtf):
                    flood_depths[pixel] = max(fldmn, dtf)
                    flood_members.add(np.int32(pixel))
                    new_boundary.add(np.int32(pixel))

        boundary = _update_boundary(records, filled_dem, new_boundary)
        spill = True
        while spill:
            before = len(records)
            _flood_map(records, flood_depths, fldmn)
            candidates = _spill_candidates(
                boundary, records, filled_dem, nrows, ncols, fldht, global_max_wse, bg, excluded
            )

            new_boundary.clear()

            for fsp, spill_dtf, pixel, pixel_elev, _ in candidates:
                path = _forward_path(pixel, spill_dtf, filled_dem, flow_direction, flood_depths, nrows, ncols, bg, excluded)
                if not path:
                    continue
                path_set = set(path)
                path_stage = min(global_max_wse, pixel_elev + fldht - spill_dtf)
                path_depth = path_stage - pixel_elev

                pending: list[tuple[int, float]] = [(p, 0.0) for p in path]
                for source in path:
                    source_wse = filled_dem[source] + path_depth
                    pending.extend(
                        _backfill_from_source(
                            source, 0.0, source_wse, filled_dem, flow_direction, nrows, ncols,
                            bg, flood_members=None, excluded=path_set
                        )
                    )

                for new_pixel, local_dtf in pending:
                    if _assimilate(records, fsp, new_pixel, local_dtf + spill_dtf):
                        new_boundary.add(np.int32(new_pixel))

            boundary = _update_boundary(records, filled_dem, new_boundary)
            if iterative_spill:
                has_shallow_boundary = False

                for _, _, dtf in boundary:
                    if dtf < fldht:
                        has_shallow_boundary = True
                        break

                if len(records) > before and has_shallow_boundary:
                    spill = len([row for row in boundary if row[2] < fldht]) > 0
                else:
                    spill = False
            else:
                spill = False

    rows: list[list[float]] = []
    for pixel, (fsp, dtf) in records.items():
        out_dtf = max(fldmn, dtf)
        sink_fill_depth = filled_dem[pixel] - max(0.0, dem[pixel])
        rows.append([fsp, pixel, out_dtf, sink_fill_depth])

    return rows

def fldpln_library_for_segment(dem: np.ndarray,
                               filled_dem: np.ndarray, 
                               flow_direction: np.ndarray, 
                               stream_id: int,
                               stream_info: np.ndarray,
                               dh: float,
                               fldmn: float,
                               fldmx: float,
                               iterative_spill: bool,
                               global_max_wse: float = 0.0,
                               bg: float = -9999):
    """Build an FLDPLN floodplain table for a single stream segment.

    Parameters
    ----------
    dem : np.ndarray
        Original elevation raster.
    filled_dem : np.ndarray
        Filled elevation raster.
    flow_direction : np.ndarray
        D8 flow-direction raster, assumes whitebox conventions.
    stream_id : int
        Stream segment identifier.
    stream_info : np.ndarray
        Segment metadata, as a 2D array with one row per segment and columns including start pixel (0), end pixel (1), length (2), and linkno (3).
    dh : float
        Flood increment step; must be positive.
    fldmn : float
        Minimum flood depth.
    fldmx : float
        Maximum flood depth to evaluate.
    iterative_spill : bool
        If True, continue spilling while shallow boundary conditions exist.
    global_max_wse : float, optional
        Upper bound on water-surface elevation. 0 means no bound. 
    bg : float, optional
        Background/no-data value in DEM.

    Returns
    -------
    pandas.DataFrame
        Table with FSP, FPP, DTF, and sink fill depth columns.
    """
    if dh <= 0:
        raise ValueError("dh must be positive.")

    nrows, ncols = dem.shape
    if flow_direction.shape != (nrows, ncols):
        raise ValueError("flow_direction shape must match filled_dem shape.")
    if dem.shape != (nrows, ncols):
        raise ValueError("dem shape must match filled_dem shape.")

    dem = dem.astype(np.float32, copy=False).ravel()
    filled_dem = filled_dem.astype(np.float32, copy=False).ravel()
    flow_direction = flow_direction.astype(np.uint8, copy=False).ravel()

    global_max_wse = np.finfo(np.float32).max if not global_max_wse else 0.99999 * global_max_wse

    rows = _fldpln_library_for_segment(
        (np.int32(nrows), np.int32(ncols)), dem, filled_dem, flow_direction, stream_id, stream_info, dh, fldmn, fldmx, iterative_spill, global_max_wse, bg
    )
    header = [
        "FSP",
        "FPP",
        "DTF",
        "sink fill depth",
    ]
    df = pd.DataFrame(rows, columns=header)
    return df