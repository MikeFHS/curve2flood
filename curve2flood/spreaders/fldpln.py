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

# Flow direction values in the MATLAB port's D8 convention.
INFLOW = np.asarray([4, 8, 16, 2, 32, 1, 128, 64], dtype=np.int32)
OUTFLOW = np.asarray([64, 128, 1, 32, 2, 16, 8, 4], dtype=np.int32)

# MATLAB
INFLOW = np.asarray([2, 4, 8, 1, 16, 128, 64, 32], dtype=np.int32)
# OUTFLOW = np.asarray([32, 64, 128, 16, 1, 8, 4, 2], dtype=np.int32)

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
    return row * ncols + col

@njit(cache=True, nogil=True)
def _valid_neighbors(pixel: int, nrows: int, ncols: int):
    row, col = _pixel_to_rc(pixel, ncols)
    for pos, (dr, dc) in enumerate(NEIGHBOR_DELTAS):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < nrows and 0 <= cc < ncols:
            yield (pos, _rc_to_pixel(rr, cc, ncols))

@njit(cache=True, nogil=True)
def _next_downstream(pixel: int, fdr: np.ndarray, nrows: int, ncols: int) -> int | None:
    OUTFLOW = {
        np.uint8(32): (-1, -1),
        np.uint8(64): (-1, 0),
        np.uint8(128): (-1, 1),
        np.uint8(16): (0, -1),
        np.uint8(1): (0, 1),
        np.uint8(8): (1, -1),
        np.uint8(4): (1, 0),
        np.uint8(2): (1, 1),
    }

    fd = fdr[pixel]
    if fd == 0:
        return None

    row, col = _pixel_to_rc(pixel, ncols)
    dr, dc = OUTFLOW[fd]
    rr = row + dr
    cc = col + dc
    if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
        return None
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
        if nxt is None:
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
        if nxt is None or nxt in seen:
            break
        excluded.add(nxt)
        seen.add(nxt)
        current = nxt

def _segment_pixels(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> list[int]:
    """Return stream pixels for a segment id. The seed-point path is not ported."""
    row = stream_info[stream_info[:, 3] == stream_id, [0, 2]]
    start = row[0]
    length = row[1]
    # row = stream_info.iloc[1]
    # if row.empty:
    #     raise ValueError("stream_id not found in stream_info.")

    # start = round(row.iat[0, 0])
    # length = round(row.iat[0, 2])
    # length = round(row.iat[0, 4])
    # start = round(row.iat[0])
    # length = round(row.iat[4])

    if length <= 0:
        return []

    pixels = [start]
    current = start
    for _ in range(1, length):
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt is None:
            break
        pixels.append(nxt)
        current = nxt
    return pixels


def _downstream_exclusion(stream_id: int, stream_info: np.ndarray, fdr: np.ndarray, nrows: int, ncols: int) -> set[int]:
    """Pixels downstream of the segment end, excluded from spillover candidates."""
    end_pixel = stream_info[stream_info[:, 3] == stream_id, 1][0]  # Assuming linkno is in the 4th column (index 3)
    # row = stream_info.iloc[1]
    # end_pixel = int(row[0, 1])
    # end_pixel = int(row.iat[1])

    excluded: set[int] = set()
    current = end_pixel
    seen = {current}
    while True:
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt is None or nxt in seen:
            break
        excluded.add(nxt)
        seen.add(nxt)
        current = nxt

    return excluded


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
    source_elev = fil[source]
    result: list[tuple[int, float]] = []
    queue: list[int] = [source]
    visited = {source}
    excluded = excluded or set()

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
            visited.add(nbr)
            result.append((nbr, dtf))
            queue.append(nbr)

    return result

# @njit(cache=True, nogil=True)
def _boundary(records: dict[int, tuple[int, float]], fil: np.ndarray,
              nrows: int, ncols: int, bg: float) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for pixel, (fsp, dtf) in records.items():
        if fil[pixel] == bg:
            continue
        for _, nbr in _valid_neighbors(pixel, nrows, ncols):
            if nbr not in records and fil[nbr] != bg:
                out.append((fsp, pixel, dtf))
                break
    out.sort(key=lambda x: (x[2], -fil[x[1]]))
    return out


def _flood_map(records: dict[int, tuple[int, float]], flddat: np.ndarray, fldmn: float) -> None:
    for pixel, (_, dtf) in records.items():
        flddat[pixel] = max(fldmn, dtf)


def _spill_candidates(boundary: list[tuple[int, int, float]], records: dict[int, tuple[int, float]],
                      flddat: np.ndarray, fil: np.ndarray, nrows: int, ncols: int,
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
            previous = best.get(nbr)
            if previous is None:
                best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)
            else:
                old_available, old_bdy_elev, *_ = previous
                if available_depth > old_available or (
                    available_depth == old_available and bdy_elev > old_bdy_elev
                ):
                    best[nbr] = (available_depth, bdy_elev, fsp, spill_dtf, nbr_elev)

    candidates = [
        (fsp, spill_dtf, pixel, pixel_elev, bdy_elev)
        for pixel, (_, bdy_elev, fsp, spill_dtf, pixel_elev) in best.items()
    ]
    candidates.sort(key=lambda x: (x[1], -x[4]))
    return candidates


def _forward_path(start: int, spill_dtf: float, fil: np.ndarray, fdr: np.ndarray,
                  flddat: np.ndarray, nrows: int, ncols: int, bg: float,
                  excluded: set[int]) -> list[int]:
    path: list[int] = []
    seen: set[int] = set()
    current = start
    while True:
        if current in seen:
            break
        seen.add(current)
        path.append(current)
        nxt = _next_downstream(current, fdr, nrows, ncols)
        if nxt is None:
            break
        if nxt in excluded:
            path.append(nxt)
            break
        if fil[nxt] == bg:
            break
        if flddat[nxt] > 0 and spill_dtf >= flddat[nxt]:
            break
        current = nxt
    return path


def _assimilate(records: dict[int, tuple[int, float]], fsp: int, pixel: int, dtf: float) -> bool:
    old = records.get(pixel)
    if old is None or old[1] > dtf:
        records[pixel] = (fsp, dtf)
        return True
    return False


import tqdm

# @njit(cache=True, nogil=True, parallel=True)
@profile
def fldpln_library_for_segment(dem: np.ndarray,
                               filled_dem: np.ndarray, 
                               flow_direction: np.ndarray, 
                               stream_id: int,
                               stream_info: np.ndarray,
                               dh: float,
                               fldmn: float,
                               fldmx: float,
                               ssflg: bool,
                               global_max_wse: float = 0.0,
                               bg: float = -9999):
    """
    This function is meant to mimic the FLDPLN model developed at the University of Kansas, translated iteratively
    using Codex-ChatGPT and this repository: https://github.com/AlabamaWaterInstitute/fldpln and this documentation: 
    https://services.kars.geoplatform.ku.edu/fldpln/AGU_2023_Operational_FIM_in_Kansas.pdf and 
    https://kuscholarworks.ku.edu/server/api/core/bitstreams/df102f13-5968-4e45-ad04-91b4d8086de0/content

    Propagate each seeded WSE upstream along reverse flow direction, then iteratively
    perform boundary spillover and upstream backfill to steady-state.
    A cell is inundated if its elevation + 0.1 m is lower than the seed WSE and it drains to
    that seed cell. If multiple seeds reach a cell, keep the maximum WSE.

    Inputs:
    - WSE_Initial: seeded WSE raster (nan/-9998 where dry); WSE for each stream cell
    - E: ground elevation raster
    - flowdir: D8 flow direction grid
    - streams: stream/segment ids for source labeling
    - nrows/ncols: raster dimensions
    
    Notes:
    - Spillover candidates are dry boundary cells adjacent to wet cells where WSE_Out > E.
    - Candidate depth is selected from wet neighbors using a minimum required depth
      (tie-breaker: highest boundary elevation).
    - Spillover floods the candidate point and then backfills upstream (reverse flowdir)
      to the spill depth until steady-state.

    Returns WSE_Out or the WSE Array for a one set of streamflow inputs
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
    strpts = _segment_pixels(stream_id, stream_info, flow_direction, nrows, ncols)
    excluded = _downstream_exclusion(stream_id, stream_info, flow_direction, nrows, ncols)

    records: dict[int, tuple[int, float]] = {}
    for pixel in strpts:
        if filled_dem[pixel] != bg:
            records[pixel] = (pixel, 0.0)

    fldht = 0.0
    iterations = int(np.ceil(fldmx / dh))

    flddat = np.zeros(filled_dem.size, dtype=np.float32)

    for _ in tqdm.tqdm(range(iterations), desc="Processing floodplain"):
        fldht += min(dh, fldmx - fldht)

        bdy = _boundary(records, filled_dem, nrows, ncols, bg)
        _flood_map(records, flddat, fldmn)
        flood_members = set(records)

        for fsp, boundary_pixel, boundary_dtf in bdy:
            boundary_elev = filled_dem[boundary_pixel]
            max_wse = min(global_max_wse, boundary_elev + fldht - boundary_dtf)
            additions = _backfill_from_source(
                boundary_pixel, boundary_dtf, max_wse, filled_dem, flow_direction, nrows, ncols,
                bg, flood_members=flood_members
            )
            for pixel, dtf in additions:
                if _assimilate(records, fsp, pixel, dtf):
                    flddat[pixel] = max(fldmn, dtf)
                    flood_members.add(pixel)

        bdy = _boundary(records, filled_dem, nrows, ncols, bg)
        spill = True
        while spill:
            before = len(records)
            _flood_map(records, flddat, fldmn)
            candidates = _spill_candidates(
                bdy, records, flddat, filled_dem, nrows, ncols, fldht, global_max_wse, bg, excluded
            )

            for fsp, spill_dtf, pixel, pixel_elev, _ in candidates:
                path = _forward_path(pixel, spill_dtf, filled_dem, flow_direction, flddat, nrows, ncols, bg, excluded)
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
                    _assimilate(records, fsp, new_pixel, local_dtf + spill_dtf)

            bdy = _boundary(records, filled_dem, nrows, ncols, bg)
            if ssflg:
                if len(records) > before and any(dtf < fldht for _, _, dtf in bdy):
                    bdy = [row for row in bdy if row[2] < fldht]
                    spill = len(bdy) > 0
                else:
                    spill = False
            else:
                spill = False


    rows: list[list[float]] = []
    for pixel, (fsp, dtf) in records.items():
        out_dtf = max(fldmn, dtf)
        # fill_adjusted = out_dtf + fil[pixel] - max(0.0, dem[pixel])
        sink_fill_depth = filled_dem[pixel] - max(0.0, dem[pixel])
        rows.append([fsp, pixel, out_dtf, sink_fill_depth])

    header = [
        "FSP",
        "FPP",
        "DTF",
        "sink fill depth",
    ]
    df = pd.DataFrame(rows, columns=header)
    # df = df.sort_values(by=['FSP', 'FPP', 'DTF'])
    return df